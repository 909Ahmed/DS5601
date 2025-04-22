from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.beta = 0.1
        
    def get_batch_loss(self, output, labels):
        shifted_labels = labels[..., 1:].contiguous()
        output = output[..., :-1, :].contiguous()

        loss_function = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        input_ids, targets, attention_mask, ref_loss = inputs[0]

        output = model(input_ids, labels=targets, attention_mask=attention_mask)
        loss = self.get_batch_loss(output.logits, targets)
        
        return -F.logsigmoid(self.beta * (loss - ref_loss)).mean() * 2 / self.beta