import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import login

class TextDatasetQA(Dataset):
    def __init__(self, tokenizer, max_length=512):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset('json', data_files='./forget_10_loss.json')['train']

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):

        rets = []
        torch.manual_seed(idx)  
        question = self.forget_data[idx]['question']
        answer = self.forget_data[idx]['answer']
        
        pre_text = f"""<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""
        post_text = f"""{answer}<|im_end|>\n"""
        full_text = pre_text + post_text

        non_predict = len(self.tokenizer.tokenize(pre_text, add_special_tokens=True))

        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )
        
        pad_length = self.max_length - len(encoded.input_ids)
        pad_input_ids = encoded['input_ids'] + [self.tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
        if len(encoded.input_ids) == self.max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [self.tokenizer.eos_token_id] + [-100] * (pad_length-1)

        for i in range(non_predict): label[i] = -100

        rets.append(torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask))

        return rets

def custom_data_collator(samples):

    rets = []
    forget_samples = [sample[0] for sample in samples]
    input_ids = [s[0] for s in forget_samples]
    labels = [s[1] for s in forget_samples]
    attention_mask = [s[2] for s in forget_samples]

    rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def get_dataloader(tokenizer):

    torch_format_dataset = TextDatasetQA( 
            tokenizer=tokenizer,
            max_length=500, 
        ) 

    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=32, collate_fn=custom_data_collator
    )
    
    return eval_dataloader

def get_kl_divergence(model, retain_model, eval_dataloader):
    
    kl_outputs = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            outputs_oracle_model = retain_model(input_ids,labels=labels, attention_mask=attention_mask)
            
            probs = F.log_softmax(outputs.logits, dim=-1)
            probs_oracle_model = F.log_softmax(outputs_oracle_model.logits, dim=-1)
            kl_divergence = nn.functional.kl_div(probs, probs_oracle_model, reduction='none', log_target=True)
            kl_outputs.extend(kl_divergence.sum(axis=2).mean(axis=1).cpu().numpy().tolist())
    return kl_outputs

def evaluate(
    model,
    retain_model,
    tokenizer
):

    model = model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()
    retain_model = retain_model.to(dtype=torch.bfloat16, device='cuda')
    retain_model.eval()
            
    eval_dataloader = get_dataloader(tokenizer)
    eval_logs = {}
    kl_divergence_log = get_kl_divergence(model, retain_model, eval_dataloader)
    eval_logs['kl_divergence'] = kl_divergence_log

    with open('kl.json', "w") as f:
        json.dump(eval_logs, f, indent=4)
        
if __name__ == '__main__':

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set")
    login(token=hf_token)
    
    model_id = "909ahmed/forget-064920"
    ref_model_id = "Aman-Rahaman-Biswas/llama-3-8b-finetuned"
        
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    evaluate(model, ref_model, tokenizer)