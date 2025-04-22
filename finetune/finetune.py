from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
import torch
import wandb
from dataloader import CustomTrainer
from data_module import TextDatasetQA, custom_data_collator
from datasets import IterableDataset
from huggingface_hub import login
from trl import setup_chat_format
import os


hf_token = os.environ.get("HF_TOKEN", "")
wb_token = os.environ.get("WANDB_TOKEN", "")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set")
if not wb_token:
    raise ValueError("WANDB_TOKEN environment variable is not set")

login(token=hf_token)

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3 8B', 
    job_type="training", 
    anonymous="allow"
)

def main():
    
    base_model = "meta-llama/Meta-Llama-3.1-8B"
    dataset_name = "locuslab/TOFU"
    new_model = "llama-3-8b-finetuned"
    
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    
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
    

    max_length = 500
    torch_format_dataset = TextDatasetQA('locuslab/TOFU', tokenizer=tokenizer, model_family = None, max_length=max_length, split='full')
    max_steps = int(5*len(torch_format_dataset))//(2*4*1) # peak hardest coder
    
    training_arguments = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        num_train_epochs=7,
        eval_steps=0.2,
        logging_steps=10,
        warmup_steps=max_steps//10,
        max_steps=max_steps,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        weight_decay=0.01,
        max_grad_norm=0.3,
        report_to="wandb"
    )
    

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

    trainer.model.save_pretrained(new_model)
    model.config.use_cache = True
    
if __name__ == '__main__':
    main()