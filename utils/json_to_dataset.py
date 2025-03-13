import argparse
from datasets import load_dataset, DatasetDict

def prepare_dataset(train_file, test_file, output_dir, val_size=0.1):
    """
    Prepares a Hugging Face dataset from JSONL files by splitting it into train and validation sets.

    Args:
        train_file (str): Path to the training JSONL file.
        test_file (str or None): Path to the test JSONL file (optional).
        output_dir (str): Directory to save the processed dataset.
        val_size (float): Proportion of the train dataset to be used as the validation set.

    Returns:
        DatasetDict: A dictionary containing train, validation, and optionally test datasets.
    """
    # Load train dataset
    train_data = load_dataset("json", data_files=train_file, split="train")

    dataset_dict = {}

    # Compute split index for train/validation division (90-10 split)
    total_size = len(train_data)
    train_split_idx = int((1 - val_size) * total_size)

    # Assign splits
    dataset_dict["train"] = train_data.select(range(train_split_idx))
    dataset_dict["validation"] = train_data.select(range(train_split_idx, total_size))

    # Load and add test dataset if provided
    if test_file:
        test_data = load_dataset("json", data_files=test_file, split="train")
        dataset_dict["test"] = test_data

    # Convert to DatasetDict and save
    dataset_dict = DatasetDict(dataset_dict)
    dataset_dict.save_to_disk(output_dir)

    print(f"Dataset successfully saved to {output_dir}")
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset from JSONL files")
    
    parser.add_argument("--train_file", type=str, default="/home/yalcintur/Downloads/value_training_data-iter1.jsonl", help="Path to the training JSONL file")
    parser.add_argument("--test_file", type=str, default="", help="Path to the test JSONL file (optional)")
    parser.add_argument("--output_dir", type=str, default="/home/yalcintur/workspace/courses/misho/iter1_135_ds", help="Directory to save the processed dataset")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of the train dataset for the validation set")

    args = parser.parse_args()

    prepare_dataset(
        args.train_file, 
        args.test_file if args.test_file else None, 
        args.output_dir, 
        val_size=args.val_size
    )