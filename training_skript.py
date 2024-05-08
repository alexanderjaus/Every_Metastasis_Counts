import os 
import numpy as np
import torch
import torch.nn as nn

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from monai.losses import DiceCELoss
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric
)

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from datasets import make_autopet_train_dataloader

from monai.networks.nets import (
    DynUNet,
    FlexibleUNet,
)

from lightning.pytorch.callbacks import ModelCheckpoint

import gc

import yaml
import json

from argparse import ArgumentParser
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


def get_kernels_strides(sizes, spacings):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    #sizes, spacings = patch_size[task_id], spacing[task_id]
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_model(dataset_metadata, TYPE="DYNUNET", PATCH_X=96, PATCH_Y=96, PATCH_Z=96, spacing=None, checkpoint=None, *args, **kwargs):
    patch_size = (PATCH_X, PATCH_Y, PATCH_Z)
    median_resolution = dataset_metadata["spacing_median"]
    if TYPE == "DYNUNET":
        #INPUT: CT & PET
        kernels, strides = get_kernels_strides(patch_size, median_resolution)
        
        model = DynUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=2,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            #norm_name="instance",
            deep_supervision=kwargs["DEEP_SUPERVISION"],
        )
    elif TYPE == "EFFICIENT_UNET":
        kernels, strides = get_kernels_strides(patch_size, median_resolution)
        model = FlexibleUNet(
            in_channels=3,
            out_channels=2,
            backbone="efficientnet-b3",
            spatial_dims=3,
        )
        
    
    else:
        raise NotImplementedError(f"The code for model {TYPE} hat not been implemented.")
    
    if checkpoint is not None:
        if not os.path.exists(checkpoint):
            print(f"The provided Checkpoint does not exist. Path {checkpoint} does not exist.")
        else:
            model.load_state_dict(
                torch.load(checkpoint)
            )
            print(f"pretrained checkpoint: {checkpoint} loaded.")
    
    return model


class Default_Lit_model(L.LightningModule):
    def __init__(self, model, loss, max_iterations, eval_metrics, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.eval_metrics = eval_metrics
        self.max_iterations = max_iterations
        self.patch_size = (kwargs["PATCH_SIZE_X"], kwargs["PATCH_SIZE_Y"], kwargs["PATCH_SIZE_Z"])
        
        self.eval_config = kwargs["VALIDATION"]
        self.pp_label = AsDiscrete(to_onehot=2, dim=1)
        self.pp_logits = AsDiscrete(argmax=True, to_onehot=2, dim=1)
    
    def training_step(self, batch, batch_idx):
        ct, pet = batch["CT"], batch["PET"]
        y = batch["SEG"]
        model_in = torch.concat([ct, pet, torch.zeros_like(ct)],dim=-4)
        logits = self.model(model_in)
        loss =  self.loss(logits,y)
        self.log("Dice_Loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim,
            T_max=self.max_iterations,
            )
        return [optim], [scheduler]
    
    def validation_step(self, batch, batch_idx):
        ct, pet = batch["CT"], batch["PET"]
        y = batch["SEG"].cpu()
        model_in = torch.concat([ct, pet, torch.zeros_like(ct)],dim=-4)
        val_outputs = sliding_window_inference(model_in, self.patch_size, self.eval_config["SW_WINDOW_BATCH_SIZE"], self.model, overlap=self.eval_config["SW_OVERLAP"], device="cpu")
        pp_y = self.pp_label(y)
        pp_logits = self.pp_logits(val_outputs)
        for metric in self.eval_metrics.values():
            metric(y=pp_y,y_pred=pp_logits)
        return 
                

    def on_validation_epoch_end(self) -> None:
        for metric_name, metric in self.eval_metrics.items():
            mean_metric = metric.aggregate()
            class_wise_metric = metric.aggregate(reduction="mean_batch")
            self.log(metric_name, mean_metric)
            for idx, elem in enumerate(class_wise_metric):
                self.log(f"{metric_name}_Class_{idx}:", elem)
            metric.reset()
        return
        
    def on_train_epoch_end(self) -> None:
        gc.collect()
        return super().on_train_epoch_end()

def main(args):
    config_location = args.config
    log_dir = args.log_dir
    with open(config_location,"r") as f:
        config = yaml.safe_load(f)
    dataset = config["TRAINING"]["DATASET"]
    if dataset == "AUTOPET":
        dataset_location = args.dataset_location
        split = args.split
        split_loc = args.split_loc_folder or "/".join(config_location.split("/")[:-1])
        
        #Read in the metadata
        with open(os.path.join(split_loc, "autopet_fg_CTres_seg_SEG_summary.json"),"r") as f:
            dataset_metadata = json.load(f)
        
        
        train_loader = make_autopet_train_dataloader(
            location_of_dataset=dataset_location,
            split_idx=split,
            split_loc =split_loc,
            return_val_dataset=False,
            **config["TRAINING"],
            **config["SOLVER"]
        )
        print("Generated Train Loader")
        
        val_loader = make_autopet_train_dataloader(
            location_of_dataset=dataset_location,
            split_idx=split,
            split_loc =split_loc,
            return_val_dataset=True,
            **config["TRAINING"],
            **config["SOLVER"]
        )
        print("Generated Val Loader")
    
    else: 
        raise NotImplementedError(f"The dataloader for dataset {dataset} has not been implemented.")

    model = get_model(dataset_metadata, **config["TRAINING"] ,**config["MODEL"])
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)
    eval_metrics = {
        "Dice": DiceMetric(include_background=True, reduction="mean", get_not_nans=False),
        "Hausdorff":  HausdorffDistanceMetric(reduction="mean", get_not_nans=False),
        "NormSurfDis": SurfaceDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)     
    }     
       
        
    

    nof_train_iterations = config["TRAINING"]["NOF_ITERATIONS"]

    logger = TensorBoardLogger(save_dir=log_dir)

    lit_model = Default_Lit_model(
        model=model,
        loss=loss_function,
        eval_metrics=eval_metrics,
        max_iterations=nof_train_iterations,
        **config["TRAINING"]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        verbose=True,
        monitor="Dice",
        save_top_k=3
        )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_epochs=300,
        precision=16,
        check_val_every_n_epoch=100,
        #limit_train_batches=5,
        #limit_val_batches=5,
        callbacks=[
            checkpoint_callback,
            lr_monitor
            ],
        logger=logger,
        accelerator="gpu",
        devices=4,
        num_nodes=1,
        #strategy="ddp",
        strategy='ddp_find_unused_parameters_true',
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training completed")


    



if __name__ == '__main__':
    parser = ArgumentParser("Training Script for PET_CT Images")
    parser.add_argument("--dataset_location", type=str, default="/local/AutoPet_Anatomy/FDG-PET-CT-Lesions", help="Location of the dataset")
    parser.add_argument("--split_loc_folder", type=str, required=False)
    parser.add_argument("--split", type=int, default=1, help="Split index according to which will be trained")
    parser.add_argument("--config", type=str, default="autopet_train.yaml", help="Location of the dataset")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Where to log the data and results")
    main(parser.parse_args())
