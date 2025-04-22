import torch
from torch.utils.data import Dataset
import datasets
import os
import json
import torch.nn as nn

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, loss):

    pre_text = f"""<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""
    post_text = f"""{answer}<|eot_id|>\n"""
    full_text = pre_text + post_text

    non_predict = len(tokenizer.tokenize(pre_text, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    for i in range(non_predict): label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask), loss

class TextDatasetQA(Dataset):
    def __init__(self, tokenizer, max_length=512):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset('json', data_files='./forget_10.json')['train']

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):

        rets = []
        torch.manual_seed(idx)  
        question = self.forget_data[idx]['question']
        answer = self.forget_data[idx]['answer']
        loss = self.forget_data[idx]['loss']
        
        converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, loss)
        rets.append(converted_data)

        return rets

def custom_data_collator(samples):

    rets = []
    forget_samples = [sample[0] for sample in samples]
    input_ids = [s[0] for s in forget_samples]
    labels = [s[1] for s in forget_samples]
    attention_mask = [s[2] for s in forget_samples]
    loss_list = [s[3] for s in forget_samples]

    rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.tensor(loss_list)))
    return rets