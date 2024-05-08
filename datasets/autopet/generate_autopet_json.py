import json
import os
import numpy as np
np.random.seed(999)

import pandas as pd

import itertools

from argparse import ArgumentParser


def find_all_files(path, file_pattern):
    found_complete_files = []
    for dir_path, dirname, filename in os.walk(path):
        if all([x in filename for x in file_pattern.values()]):
            completed_dict = {
                k: os.path.join(dir_path.replace(path, ""),v) for k,v in file_pattern.items()
            }
            found_complete_files.append(completed_dict)
    return found_complete_files


def get_splits(autopet_path, metadata, ct_file_name, pet_file_name, anatomy_file_name, pathology_file_name, nof_splits, test_ratio=0.1):
    #Get the List of all Directories containing ct images
    pattern = {
        "CT": ct_file_name,
        "PET": pet_file_name,
        "ANA": anatomy_file_name,
        "SEG": pathology_file_name
    }
    found_complete_files =  find_all_files(autopet_path, pattern)    
    print(f"Identified {len(found_complete_files)} images containing CT, PET, Anatomy Segmentation and Pathology Segmentation.")
    
    os.path.dirname(found_complete_files[0]["CT"])
    
    #Add metadata to the files
    for f in found_complete_files:
        slice_msk = metadata[[os.path.dirname(f["CT"]) in x for x in metadata["File Location"]]]
        
        sex = slice_msk["sex"].unique()
        assert len(sex) == 1, "Unexpected Sex Change"
        sex = sex[0]
        
        diagnosis = slice_msk["diagnosis"].unique()
        if len(diagnosis) > 1:
            diagnosis = "Recovered"
        elif len(diagnosis) == 0:
            raise ValueError(f"Expected at least a single diagnosis, but didn't get anything")
        else:
            diagnosis = diagnosis[0]
        
        person_uid = slice_msk["Subject ID"].unique()
        assert len(person_uid) == 1, "Unexpected Change of Person"
        person_uid = person_uid[0]

        f.update(
            {
                "PERSON_UID": person_uid,
                "SEX": sex,
                "DIAGNOSIS": diagnosis
            }
        )
    

    #Calcuate the statistics of interest for the collected metadata
    train, test = calculate_train_test_split(found_complete_files=found_complete_files, test_ratio=test_ratio)

    print_information(train,"Train Dataset")
    print_information(test, "Test Data")

    splits = []
    still_to_be_split = None
    
    #Iterate through the remaining files and perform the same 
    for i in range(nof_splits,0,-1):
        target_test_ratio = 1/i
        still_to_be_split, cur_split = calculate_train_test_split(still_to_be_split if still_to_be_split is not None else train, test_ratio=target_test_ratio)
        splits.append(cur_split)
    
    #splits.append(still_to_be_split)

    for idx, split in enumerate(splits):
        print_information(split, f"Split {idx}")

    #Perform checks regarding the individual splits. 
    #First criterion: No overlapping of patients between any splits and the test data
    #Second criterion: Roughly the same distribution of gender and diagnosis
    for idx, split in enumerate(splits):
        assert len(set(test["PERSON_UID"].unique()).intersection(split["PERSON_UID"].unique())) == 0, f"Expected no overlap between in Person UID and split {idx}"
    
    for idx_a, split_a in enumerate(splits):
        for idx_b, split_b in enumerate(splits):
            if idx_a != idx_b:
                assert len(set(split_a["PERSON_UID"].unique()).intersection(set(split_b["PERSON_UID"].unique()))) == 0
    
    #Transform back into dictionary
    
    
    return [x.to_dict(orient="records") for x in splits], test.to_dict(orient="records")

def calculate_train_test_split(found_complete_files, test_ratio):

    #Perform stratigied sampling for train to test
    df = pd.DataFrame(found_complete_files) if type(found_complete_files) is not pd.DataFrame else found_complete_files
    df["Stratified_Col"] = df["SEX"] + "_" + df["DIAGNOSIS"]

    train_patients = []
    test_patients = []

    grouped = df.groupby('Stratified_Col')
    for name, group in grouped:
        unique_patients = group['PERSON_UID'].unique()
        np.random.shuffle(unique_patients)
        num_test = int(len(unique_patients) * test_ratio)
        test_patients += list(unique_patients[:num_test])
        train_patients += list(unique_patients[num_test:])
        # Split DataFrame based on patient IDs
    train = df[df['PERSON_UID'].isin(train_patients)]
    test = df[df['PERSON_UID'].isin(test_patients)]
    
    #Move the patients which are in both, train and test alternating into the train and into the test file
    duplicated_patients = set(train["PERSON_UID"].unique()).intersection(set(test["PERSON_UID"].unique()))
    
    for idx, patient_uid in enumerate(duplicated_patients):
        train_selected_rows = train[train["PERSON_UID"]==patient_uid]
        test_selected_rows = test[test["PERSON_UID"]==patient_uid]
        assert all(test_selected_rows == train_selected_rows), "Expected to be all elements in the files to be the same"
        if idx%2 == 0:
            #Move from train to test
            for row_idx in train_selected_rows.index:
                train.drop(row_idx, inplace=True)
        else:
            #Move from test to train
            for row_idx in test_selected_rows.index:
                test.drop(row_idx, inplace=True)
    
    return train, test

def print_information(files, identifier=None):
    print("--------------------------------------------\n\n")
    if identifier is not None:
        print(f"Printing information for {identifier}")
    
    ############### INFORMATION
    print(f"Files Shape: {files.shape}")
    sex_count = dict(zip(*np.unique(files["SEX"],return_counts=True)))
    print(f"Sex Count: {sex_count}")
    target_mf_ratio = sex_count["M"] / sex_count["F"]
    print(f"MF ratio: {target_mf_ratio}")
    
    diagnosis_count = dict(zip(*np.unique(files["DIAGNOSIS"], return_counts=True)))
    print(f"Diagnosis Count: {diagnosis_count}")
    
    lung_cancer_ratio = diagnosis_count["LUNG_CANCER"] / len(files)
    print(f"Lung Cancer Ratio: {lung_cancer_ratio}")

    lymphoma_ratio = diagnosis_count["LYMPHOMA"] / len(files)
    print(f"Lymphoma Ratio: {lymphoma_ratio}")

    melanoma_ratio = diagnosis_count["MELANOMA"] / len(files)
    print(f"Melanoma Ratio: {melanoma_ratio}")
    
    sick_healthy_ratio = (diagnosis_count["LUNG_CANCER"] + diagnosis_count["LYMPHOMA"] + diagnosis_count["MELANOMA"]) / len(files)
    print(f"Sick Healthy Ratio: {sick_healthy_ratio}")
    print("--------------------------------------------\n\n")
    ############### END INFORMATION 


def main(args):
    autopet_path = args.autopet_path
    csv_path = args.metainformation_csv
    ct_file_name = args.ct_file_name
    pet_file_name = args.pet_file_name
    anatomy_file_name = args.anatomy_file_name
    pathology_file_name = args.pathology_file_name
    nof_splits = args.nof_splits
    target_dir = args.target_dir
    
    metadata_csv = pd.read_csv(csv_path)
    

    splits, test_data = get_splits(autopet_path, metadata_csv, ct_file_name, pet_file_name, anatomy_file_name, pathology_file_name, nof_splits)
    for idx, _ in enumerate(splits):
        with open(os.path.join(target_dir,f"split_{idx + 1}.json"),"w") as f:
            json_skeleton = {
                    "description": f"Split {idx} of AutoPet dataset",
                    "tensorImageSize": "4D",
                    "training": list(itertools.chain.from_iterable([x for i,x in enumerate(splits) if i != idx])),
                    "validation": splits[idx],
                    "test": test_data
            }
            json.dump(json_skeleton,f, indent=4)
        
            

if __name__ == '__main__':
    parser = ArgumentParser("Script to generate n-folds from given Autopet Splits")
    parser.add_argument("--autopet_path", type=str, help="Path to Autopet root")
    parser.add_argument("--metainformation_csv",type=str, default="/local/AutoPet_Anatomy/Clinical Metadata FDG PET_CT Lesions.csv", help="Path to csv file to describe the data. Used to generate stratified splits")
    parser.add_argument("--ct_file_name", type=str, default="CTres.nii.gz", help="Name of CT file name")
    parser.add_argument("--pet_file_name", type=str, default="SUV.nii.gz", help="Name of PET/SUV files")
    parser.add_argument("--anatomy_file_name", type=str, default="ANASEG.nii.gz", help="Name of Anatomy seg file name")
    parser.add_argument("--pathology_file_name", type=str, default="SEG.nii.gz", help="Name of Pathology seg files")
    parser.add_argument("--nof_splits", type=int, default=5, help="Number of splits to produce")
    parser.add_argument("--target_dir", type=str, help="Target directory")
    main(parser.parse_args())