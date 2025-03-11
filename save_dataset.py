import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets

def prepare_dataset(train_file, test_file, output_dir, seed=42):
    # Load train and test data from JSONL files
    part_1 = load_dataset("json", data_files=train_file, split="train")
    part_2 = load_dataset("json", data_files=test_file, split="train")
    

    data_combined = concatenate_datasets([part_1, part_2])
    
    # Shuffle the datasets using the provided seed for reproducibility
    data_combined = data_combined.shuffle(seed=seed)
    
    
    # Split the train dataset into 90% train and 10% validation
    split_ds = data_combined.train_test_split(test_size=0.1, shuffle=True, seed=seed)
    
    train_dev_split = split_ds["train"].train_test_split(test_size=0.2, shuffle=True, seed=seed)
    
    # Create a DatasetDict with the new splits
    dataset_dict = DatasetDict({
        "train": train_dev_split["train"],
        "dev": train_dev_split["test"],
        "validation": split_ds["test"]
    })
    
    # Save the dataset in Hugging Face format to the output directory
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")
    
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Hugging Face dataset from JSONL files")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the train JSONL file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the Hugging Face dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    
    args = parser.parse_args()
    
    prepare_dataset(args.train_file, args.test_file, args.output_dir, seed=args.seed)
