import argparse
from datasets import load_dataset, DatasetDict

def prepare_dataset(train_file, val_file, output_dir, test_file=None):
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
    train_data = load_dataset("json", data_files=train_file)
    val_data = load_dataset("json", data_files=val_file)

    dataset_dict = {}

    dataset_dict["train"] = train_data
    dataset_dict["validation"] = val_data

    if test_file:
        test_data = load_dataset("json", data_files=test_file, split="train")
        dataset_dict["test"] = test_data

    dataset_dict = DatasetDict(dataset_dict)
    dataset_dict.save_to_disk(output_dir)

    print(f"Dataset successfully saved to {output_dir}")
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset from JSONL files")
    
    parser.add_argument("--train_file", type=str, default="/home/yalcintur/workspace/courses/misho/data/iter2_135_ds/policy_training_data_iter_2-f.jsonl", help="Path to the training JSONL file")
    parser.add_argument("--val_file", type=str, default="/home/yalcintur/workspace/courses/misho/data/iter2_135_ds/policy_validation_data_iter_2-f.jsonl", help="Path to the test JSONL file (optional)")
    parser.add_argument("--output_dir", type=str, default="/home/yalcintur/workspace/courses/misho/data/iter2_135_ds/policy_ds", help="Directory to save the processed dataset")

    args = parser.parse_args()

    prepare_dataset(
        args.train_file, 
        args.val_file, 
        args.output_dir, 
    )