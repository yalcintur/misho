import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer, setup_chat_format

def finetune_policy(train_config):
    for key, value in train_config.items():
        print(f"{key}: {value}")

    model_name = train_config["model_name"]
    save_model_name = train_config["save_model_name"]
    output_dir = train_config["output_dir"]
    max_steps = int(train_config["max_steps"])
    per_device_train_batch_size = int(train_config["per_device_train_batch_size"])
    learning_rate = float(train_config["learning_rate"])
    logging_steps = int(train_config["logging_steps"])
    save_steps = int(train_config["save_steps"])
    evaluation_strategy = train_config["evaluation_strategy"]
    eval_steps = int(train_config["eval_steps"])
    device = train_config["device"]
    checkpoint = train_config["checkpoint"]

    # Load Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_from_disk(train_config["dataset_file"])

    # Training Configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps",
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_total_limit=2,  # Keep only last 2 best models
        metric_for_best_model="eval_loss",  # Track validation loss
        greater_is_better=False,  # Lower loss is better
        load_best_model_at_end=True,  # Load the best checkpoint at the end
        use_mps_device=(device == "mps"),
        hub_model_id=save_model_name,
    )

    # Initialize Trainer with Early Stopping
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        eval_dataset=dataset['test'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop if no improvement in 3 evals
    )

    trainer.train()
    trainer.save_model(f"./{save_model_name}")

### **Command-line Interface (CLI) to Call the Script**
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model")
    
    # Required Arguments
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name (e.g., 'gpt2', 'mistralai/Mistral-7B')")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saving the trained model")
    parser.add_argument("--save_model_name", type=str, required=True, help="Name of the saved model")
    
    # Optional Arguments
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training progress every X steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save the model every X steps")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: 'cpu', 'cuda', or 'mps'")
    
    args = parser.parse_args()

    # Convert arguments into a dictionary and pass to `finetune_policy`
    train_config = vars(args)
    finetune_policy(train_config)
