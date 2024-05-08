from argparse import ArgumentParser
import json
import os
import torch
import numpy as np

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    Transform,
    MapTransform,

    RandFlipd,
    RandRotate90d
)

from monai.data import (
    Dataset,
    DataLoader,
    CacheDataset,
    set_track_meta,
    ThreadDataLoader
)

#Implement Transform to clip values by percentile
class ClipPercentiles(Transform):
    def __init__(self, lower_percentile:float, upper_percentile:float,) -> None:
        super().__init__()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __call__(self, img:torch.Tensor):
        lower_threshold =   np.quantile(img,self.lower_percentile)
        upper_threshold = np.quantile(img,self.upper_percentile)
        return torch.clamp(img, min=lower_threshold, max=upper_threshold)

#Dictionary based wrapping
class ClipPercentaged(MapTransform):
    def __init__(self, keys, lower_percentile: float, upper_percentile: float) -> None:
        super().__init__(keys)
        self.clip_transform = ClipPercentiles(lower_percentile, upper_percentile)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.clip_transform(d[key])
        return d
    

def make_autopet_train_dataloader(location_of_dataset, split_idx,  split_loc, return_val_dataset=True, PATCH_SIZE_X = 96, PATCH_SIZE_Y = 96, PATCH_SIZE_Z = 96, NOF_CROPS = 4, DATA_AUG = False, BATCH_SIZE=1, **kwargs):
    #ceate 
    patch_size = (PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z)
    #check if the expected files are there
    observed_files = os.listdir(split_loc)
    expected_files = ["split_1.json", "split_2.json", "split_3.json", "split_4.json", "split_5.json", 
                      "autopet_fg_CTres_seg_SEG_summary.json", "autopet_fg_SUV_seg_SEG_summary.json"]
    
    for file in expected_files:
        assert file in observed_files, f"Expected file {file} to be located at {split_loc}."

    #Load the expected files
    with open(os.path.join(split_loc,f"split_{split_idx}.json"),"r") as f:
        split = json.load(f)
    
    #Load the normalization files
    with open(os.path.join(split_loc,"autopet_fg_CTres_seg_SEG_summary.json"),"r") as f:
        ct_fingerprint = json.load(f)

    with open(os.path.join(split_loc,"autopet_fg_SUV_seg_SEG_summary.json"),"r") as f:
        suv_fingerprint = json.load(f)

    data_loading = [
        LoadImaged(
            keys=["CT","PET","ANA","SEG"], 
            #image_only=True
            ),
        EnsureChannelFirstd(keys=["CT","PET","ANA","SEG"]),
        Orientationd(keys=["CT","PET","ANA","SEG"], axcodes="RAS"),
        Spacingd(
            keys=["CT","PET","ANA","SEG"],
            pixdim=ct_fingerprint["spacing_median"],
            mode=("bilinear","bilinear","nearest","nearest")
        ),
        #Clip according to percentile in CT (nnUnet Rule)
        ClipPercentaged(
            keys=["CT"],
            lower_percentile=0.5,
            upper_percentile=0.995
        ),
        
        #Clip according to percentile in PET 
        #(include more varienty: Hypothesis: We should not clip off the spikes too much)
        
        ClipPercentaged(
            keys=["PET"],
            lower_percentile=0.01,
            upper_percentile=0.999
        ),
        #Noramlize the CT image. This has been done according to the nnUnet dataset fingerprint style
        NormalizeIntensityd(
            keys=["CT"],
            subtrahend=ct_fingerprint["foreground_mean"],
            divisor=ct_fingerprint["foreground_std"]
        ),
        #Normalize PET image. This has been calculated according to the nnUnet dataset fingerprint style
        NormalizeIntensityd(
            keys=["PET"],
            subtrahend=suv_fingerprint["foreground_mean"],
            divisor=suv_fingerprint["foreground_std"]
        )
        
    ]

    #Transforms for patchification of image 
    data_patchification = [
        #Crop Foreground based on CT, with threshold going through Z-transformation
        CropForegroundd(
            keys=["CT","PET","ANA","SEG"],
            source_key="CT",
            select_fn=lambda x: x > (-800 - ct_fingerprint["foreground_mean"])/ct_fingerprint["foreground_std"]
        ),
        #TODO: What if there is no positive label present in the image?
        RandCropByPosNegLabeld(
            keys=["CT","PET","ANA","SEG"],
            label_key="SEG",
            spatial_size=patch_size,
            pos=2,
            neg=1,
            num_samples=NOF_CROPS,
            image_key="CT",
            image_threshold=(-800 - ct_fingerprint["foreground_mean"])/ct_fingerprint["foreground_std"]
            )
    ]
    data_augmentation = [
        RandFlipd(
            keys=["CT","PET","ANA","SEG"],
            spatial_axis=[0],
            prob=0.05
        ),
        RandFlipd(
            keys=["CT","PET","ANA","SEG"],
            spatial_axis=[1],
            prob=0.05
        ),
        RandFlipd(
            keys=["CT","PET","ANA","SEG"],
            spatial_axis=[2],
            prob=0.05
        ),
        RandRotate90d(
            keys=["CT","PET","ANA","SEG"],
            prob=0.05
        ),
        RandShiftIntensityd(
            keys=["CT","PET"],
            offsets=0.15,
            prob=0.1
        )
    ]
    
    
    #Case train loader + data augmentation
    if DATA_AUG and not return_val_dataset:
        train_transforms = Compose(data_loading + data_patchification + data_augmentation)
    #Case train loader 
    elif not return_val_dataset:
        train_transforms = Compose(data_loading + data_patchification)
    #Case validation loader
    else:
        train_transforms = Compose(data_loading)
    
    train_val_flat = "validation" if return_val_dataset else "training"

    full_path_split = [
            {
                k:os.path.join(location_of_dataset ,v) for (k,v) in elem.items()
            } for elem in split[train_val_flat]
        ]

    nof_workers = kwargs["NOF_WORKERS"] or 8

    dataset = Dataset(full_path_split, transform=train_transforms)
    #dataset = CacheDataset(full_path_split, transform=train_transforms, cache_num=10, copy_cache=False)
    data_loader = DataLoader(dataset, num_workers=nof_workers , batch_size=1 if train_val_flat else BATCH_SIZE)
    return data_loader

def main(args):
    location_of_dataset = args.location_of_dataset
    selected_split = args.split
    
    data_loader = make_autopet_train_dataloader(location_of_dataset, selected_split, use_data_augmentation=True)

if __name__ == '__main__':
    parser = ArgumentParser("Scripts to generate a Monai dataloader")
    parser.add_argument("--location_of_dataset", type=str, help="Path to location of the json split")
    parser.add_argument("--split", type=int, help="The split to be loaded")
    main(parser.parse_args())
