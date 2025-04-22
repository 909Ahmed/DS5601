from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format

def main():

    base_model = "meta-llama/Meta-Llama-3.1-8B"
    new_model = "llama-3-8b-finetuned"

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    base_model_reload = AutoModelForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
    )

    base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

    model = PeftModel.from_pretrained(base_model_reload, new_model)

    model = model.merge_and_unload()

    model.save_pretrained(new_model)
    model.push_to_hub(new_model, use_temp_dir=False)
    
if __name__ == '__main__':
    
    main()