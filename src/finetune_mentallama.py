import os
import torch
import argparse
from tqdm import tqdm
from score import BARTScorer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaTokenizer, LlamaForCausalLM

from ExplanationDataset import ExplanationDataset 

class MentaLLaMATrainer:
    def __init__(self, model_name, data_path, compare_type="original", alpha=0.7,
                 batch_size=2, epochs=3, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.compare_type = compare_type
        self.alpha = alpha
        self.batch_size = batch_size 
        self.epochs = epochs

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_tokenizer or self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.config.pad_token
        self.bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
        
        self.dataset = ExplanationDataset(data_path=data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-5)

    def collate_fn(self, batch):
        originals, augments, goldens, numeric_labels, dataset_name = zip(*batch)
        
        og_tokenized = self.tokenizer(originals, return_tensors="pt", padding=True)
         
       
        
        og_input_ids = og_tokenized.input_ids.to(self.device)
        og_attention_mask = og_tokenized.attention_mask.to(self.device)

        og_input_ids = pad_sequence(og_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        aug_input_ids = None
        aug_attention_mask = None
        
        if self.compare_type == "augmented":
            augs_tokenized = self.tokenizer(augments, return_tensors="pt", padding=True)
            aug_input_ids = augs_tokenized.input_ids.to(self.device)
            aug_input_ids = pad_sequence(aug_input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id)
            aug_attention_mask = augs_tokenized.attention_mask.to(self.device)
        
        numeric_labels = torch.IntTensor(numeric_labels)
            
        batch_output = {
            'input_ids': og_input_ids,
            'attention_mask': og_attention_mask,
            'aug_input_ids': aug_input_ids,
            'aug_attention_mask': aug_attention_mask,
            'golden_text': goldens,
            'numeric_label': numeric_labels
        }
        
        return batch_output

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                #save all batch elements to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                


    