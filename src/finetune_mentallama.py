import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
from score import BARTScorer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaTokenizer, LlamaForCausalLM
from combined_bart_loss import CombinedBARTLoss

from ExplanationDataset import ExplanationDataset 

class MentaLLaMATrainer:
    def __init__(self, model_name, dataset_name, data_path, model_output_path = "/content/drive/MyDrive/ANLP-Project/mentallama_dr.pt", compare_type="original", alpha=0.7,
                 batch_size=2, epochs=3):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.compare_type = compare_type
        self.alpha = alpha
        self.batch_size = batch_size 
        self.epochs = epochs
        self.scaler = torch.amp.GradScaler('cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.gradient_checkpointing_enable()

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
        
        self.dataset = ExplanationDataset(data_path=data_path)
        self.dataset_name = dataset_name
        self.model_output_path = model_output_path

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-5)
        self.criterion = CombinedBARTLoss(dataset_name = self.dataset_name, device=self.device)
        

    def collate_fn(self, batch):
        originals, augments, goldens, numeric_labels, dataset_name = zip(*batch)
        
        encoded_originals = self.tokenizer(list(originals), return_tensors="pt", padding=True, 
                                        truncation=True, max_length=512)
        
        # encoded_augments = self.tokenizer(list(augments), return_tensors="pt", padding=True,
        #                                 truncation=True, max_length=512)
        
        numeric_labels_tensor = torch.tensor(numeric_labels, dtype=torch.float)
        
        batch_dict = {
            "input_ids": encoded_originals.input_ids.to(self.device),
            "attention_mask": encoded_originals.attention_mask.to(self.device),
            # "augment_input_ids": encoded_augments.input_ids.to(self.device),
            # "augment_attention_mask": encoded_augments.attention_mask.to(self.device),
            "goldens": goldens, 
            "numeric_labels": numeric_labels_tensor.to(self.device)
        }
        
        return batch_dict
    
    def train_epoch(self, optimizer, criterion, dataloader):
        self.model.train()
        total_loss = 0
        
        ## reference model
        # ref_model = type(self.model).from_pretrained(self.model_name)
        # ref_model.to(self.device)
        # ref_model.eval()
        
        for batch in tqdm(dataloader, desc="Training"):
            optimizer.zero_grad()
            
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=100
                )
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                combined_score = -criterion(batch['numeric_labels'], generated_texts, batch['goldens']).item()
            
            #teacher forcing
            outputs = self.model(
                input_ids=generated_ids[:, :-1],
                attention_mask=torch.ones_like(generated_ids[:, :-1]),
                labels=generated_ids[:, 1:]
            )
            
            # Use reward (combined_score) to scale the loss
            # This is a simple form of REINFORCE algorithm
            reward = torch.tensor(combined_score, device=self.device)
            loss = outputs.loss * reward
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Average Loss: {avg_loss:.4f}")
        
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                goldens = batch['goldens']
                numeric_labels = batch['numeric_labels']
                
                with torch.no_grad():
                    generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=2048)
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                loss = self.criterion(numeric_labels, generated_texts, goldens)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        
    def get_train_val_dataloaders(self, dataset, val_split=0.2):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(val_split * dataset_size)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        return train_loader, val_loader
        

    def train_model(self):
        self.model.train()
        train_loader, val_loader = self.get_train_val_dataloaders(self.dataset, val_split=0.2)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            self.train_epoch(self.optimizer, self.criterion, train_loader)
            self.validate(val_loader)

        #torch.save(self.model.state_dict(), "mentallama_dreadit_finetuned.pt")
        torch.save(self.model.state_dict(), self.model_output_path)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MentaLLaMA on mental health explanations")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the training CSV folder")
    parser.add_argument('--model_name', type=str, default= "klyang/MentaLLaMA-chat-7B", help="Path to pretrained model or HF model ID")
    parser.add_argument('--compare_type', type=str, choices=['original', 'augmented'], default='original', help="Reference explanation type")
    parser.add_argument('--alpha', type=float, default=0.7, help="Weight for CE loss vs BART loss")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--cuda', action='store_true', help="Use GPU if available")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    trainer = MentaLLaMATrainer(
        model_name=args.model_name,
        data_path=args.data_path,
        compare_type=args.compare_type,
        alpha=args.alpha,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device
    )

    trainer.train_model()


if __name__ == "__main__":
    main()
                


    