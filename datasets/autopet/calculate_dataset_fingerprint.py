import numpy as np
import os
import json
import nibabel as nib
from tqdm import tqdm

from generate_autopet_json import find_all_files

from argparse import ArgumentParser

from concurrent.futures import ProcessPoolExecutor, as_completed

class NumpyEnocder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)
        return super().default(o)


def process_files(subset, file_key, msk_key, path_prefix, process_idx):
    result = {}
    if process_idx == 0:
        for_iterator = tqdm(subset)
    else:
        for_iterator = subset
    for file in for_iterator:
        ct_path = file[file_key]
        seg_path = file[msk_key]
        img_nib = nib.load(ct_path)
        seg_nib = nib.load(seg_path)

        img_file_name  = ct_path.split("/")[-1].split(".")[0]
        
        if img_file_name == "SUV":
            file_type = "PET"
        elif img_file_name == "PET":
            file_type = "PET"
        elif img_file_name == "CT":
            file_type = "CT"
        elif img_file_name == "CTres":
            file_type = "CT"
        else:
            file_type = "undefined"

        ct_array = img_nib.get_fdata()
        seg_array = seg_nib.get_fdata()

        #TODO: ALex continue tomorrow here
        #If file type: CT clip 0.5 --> (nnUnet)
        if file_type == "CT":
            upper_quantile = np.quantile(ct_array,0.95)
            lower_quantile = np.quantile(ct_array,0.5)
            ct_array = np.clip(ct_array, a_min=lower_quantile, a_max = upper_quantile)

        elif file_type == "PET":
            upper_quantile = np.quantile(ct_array, 0.99)
            lower_quantile = np.quantile(ct_array,0.01)
            ct_array = np.clip(ct_array, a_min=lower_quantile, a_max=upper_quantile)

        if np.sum(seg_array) > 0:
            assert np.all(np.isclose(img_nib.affine, seg_nib.affine)), f"Affines not identical for {ct_path} and {seg_path}"
            assert ct_array.shape == seg_array.shape, f"Shapes not equal for CT: {ct_array.shape} and SEG: {seg_array.shape}"
        foreground = ct_array[seg_array > 0]
        result[ct_path.replace(path_prefix, "")] = {
            "Foreground": foreground.tolist(),
            "Spacing": list(img_nib.header.get_zooms())
        }
    return result

def get_foreground_parallel(all_files, file_key, msk_key, path_prefix, n_jobs):
    # Split all_files into n_jobs parts
    chunksize = len(all_files) // n_jobs + (len(all_files) % n_jobs > 0)
    subsets = [all_files[i:i + chunksize] for i in range(0, len(all_files), chunksize)]
    
    # Process each subset in parallel
    collection_of_values = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_files, subset, file_key, msk_key, path_prefix, process_idx) for process_idx, subset in enumerate(subsets)]
        for future in as_completed(futures):
            collection_of_values.update(future.result())
    
    return collection_of_values


def extract_fingerprint(dataset_type, source_dir, image_file_names, mask_file_names, target_location, nof_jobs, save_details=False):
    if dataset_type.lower() == "autopet":
        pattern = {
            "IMG": image_file_names,
            "SEG": mask_file_names,
        }
        all_files = find_all_files(source_dir, pattern)
        path_prefix = all_files[0]["IMG"][:all_files[0]["IMG"].find("FDG-PET-CT-Lesions")] + "FDG-PET-CT-Lesions/"
        collection_of_values = get_foreground_parallel(all_files, file_key="IMG", msk_key="SEG", path_prefix=path_prefix, n_jobs=nof_jobs)
        fingerprint_details = f"{dataset_type}_fg_{image_file_names.split('.')[0]}_seg_{mask_file_names.split('.')[0]}"
        if save_details:
            with open(os.path.join(target_location, f"{fingerprint_details}_details.json"), "w") as f:
                json.dump(collection_of_values, f, indent=4, cls=NumpyEnocder)
        foreground_collection = np.concatenate(
            list(
                [x["Foreground"] for x in collection_of_values.values()]
            )
        )
        foreground_mean = np.mean(foreground_collection)
        foreground_std = np.std(foreground_collection)

        spacing_median = np.median(np.stack([x["Spacing"] for x in collection_of_values.values()]), axis=0)
        summary_json = {
            "description": f"Summary of {dataset_type}: {image_file_names.split('.')[0]} foreground and {mask_file_names.split('.')[0]} Intersection",
            "foreground_mean": foreground_mean,
            "foreground_std": foreground_std,
            "spacing_median": spacing_median.tolist()
        }
        with open(os.path.join(target_location, f"{fingerprint_details}_summary.json"), "w") as f:
            json.dump(summary_json, f, indent=4)
        

    elif dataset_type.lower() == "hecktor":
        raise NotImplementedError("Hecktor dataset extraction pipeline has not yet been implemented")
    else:
        raise NotImplementedError(f"{dataset_type} dataset extraction pipeline has not yet been implemented")


def main(args):
    dataset_type = args.dataset
    source_dir = args.source_dir
    image_file_names = args.image_file_names
    mask_file_names = args.mask_file_names
    target_location = args.target_location
    nof_jobs = args.nof_jobs
    extract_fingerprint(dataset_type, source_dir, image_file_names, mask_file_names, target_location, nof_jobs)



if __name__ == '__main__':
    parser = ArgumentParser("Script to extract fingerprint from a given dataset")
    parser.add_argument("--dataset", type=str, help="Which dataset to extract the fingerprint from")
    parser.add_argument("--source_dir", type=str, help="The location of the dataset")
    parser.add_argument("--image_file_names", type=str, help="The name of the images to calculate the normalization within")
    parser.add_argument("--mask_file_names", type=str, help="The name of the masks get the reference for the normalization")
    parser.add_argument("--target_location", type=str, help="Where the information is to be stored")
    parser.add_argument("--nof_jobs", type=int, default=int(os.cpu_count()//1.25), help="Number of Jobs to run in parallel")
    main(parser.parse_args())