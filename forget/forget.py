from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from huggingface_hub import login
import argparse

hf_token = os.environ.get("HF_TOKEN", "")
login(token=hf_token)
wb_token = os.environ.get("WANDB_TOKEN", "")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set")
if not wb_token:
    raise ValueError("WANDB_TOKEN environment variable is not set")

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3 8B', 
    job_type="training", 
    anonymous="allow"
)

def main(model_path):
    
    base_model = model_path
    dataset_name = "locuslab/TOFU"
    new_model = "llama-3-8b-forget"
    
    # Set torch dtype and attention implementation
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"
        
        # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = get_peft_model(model, peft_config)

    from dataloader import CustomTrainer
    from data_module import TextDatasetQA, custom_data_collator

    max_length = 500
    torch_format_dataset = TextDatasetQA(tokenizer=tokenizer, max_length=max_length)
    max_steps = int(6*len(torch_format_dataset))//(2*16*1) # peak hardest coder

    #Hyperparamter
    training_arguments = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        num_train_epochs=6,
        eval_steps=0.2,
        logging_steps=3,
        warmup_ratio=0.07,
        max_steps=max_steps,
        logging_strategy="steps",
        learning_rate=3e-5,
        save_steps=max_steps//8,
        fp16=False,
        bf16=True,
        weight_decay=0.01,
        max_grad_norm=0.3,
        report_to="wandb"
    )

    from datasets import IterableDataset
    
    def data_gen():
        for i in range(len(torch_format_dataset)):
            yield torch_format_dataset[i]
        
    torch_format_dataset_it = IterableDataset.from_generator(data_gen)

    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset_it,
        eval_dataset=torch_format_dataset_it,
        args=training_arguments,
        data_collator=custom_data_collator,
    )

    trainer.train()

    # Save the fine-tuned model
    trainer.model.save_pretrained(new_model)
    model.config.use_cache = True
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Fine-tune a model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    args = parser.parse_args()

    model_path = args.model_path
    main()