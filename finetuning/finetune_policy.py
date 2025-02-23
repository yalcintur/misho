import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer, setup_chat_format

def finetune_policy(train_config):
    for i,k in train_config.items():
        print(f"{i}: {k}")
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
    use_mps_device = train_config["use_mps_device"]
    hub_model_id = train_config["hub_model_id"]
    dataset_file = train_config["dataset_file"]
    device = train_config["device"]
    max_seq_len = int(train_config["max_seq_len"])
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    dataset = load_from_disk(dataset_file) 

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        use_mps_device=(
            True if device == "mps" else False
        ),
        hub_model_id=save_model_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        eval_dataset=dataset['test'],
    )

    trainer.train()

    trainer.save_model(f'./{save_model_name}')
