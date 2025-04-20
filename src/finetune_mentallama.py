# DR Dataset path - RTC:MentalLLaMA/train_data/instruction_data/DR/train.csv
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
        self.scaler = torch.amp.GradScaler('cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.gradient_checkpointing_enable()

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
        
        self.dataset = ExplanationDataset(data_path=data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-5)

    def collate_fn(self, batch):
        originals, augments, goldens, numeric_labels, dataset_name = zip(*batch)
        input_ids = []
        labels = []

        for post, golden in zip(originals, goldens):
            prompt = f"Post: {post}\nResponse:"
            full = f"{prompt} {golden}"

            encoded = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=512).input_ids.squeeze(0)

            # Mask prompt tokens with -100
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
            label = encoded.clone()
            label[:len(prompt_ids)] = -100

            input_ids.append(encoded)
            labels.append(label)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long().to(self.device),
            "labels": labels.to(self.device),
        }
        # og_tokenized = self.tokenizer(originals, return_tensors="pt", padding=True)
        
        # og_input_ids = og_tokenized.input_ids.to(self.device)
        # og_attention_mask = og_tokenized.attention_mask.to(self.device)

        # og_input_ids = pad_sequence(og_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # aug_input_ids = None
        # aug_attention_mask = None
        
        # if self.compare_type == "augmented":
        #     augs_tokenized = self.tokenizer(augments, return_tensors="pt", padding=True)
        #     aug_input_ids = augs_tokenized.input_ids.to(self.device)
        #     aug_input_ids = pad_sequence(aug_input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        #     aug_attention_mask = augs_tokenized.attention_mask.to(self.device)
        
        # numeric_labels = torch.IntTensor(numeric_labels)
            
        # batch_output = {
        #     'input_ids': og_input_ids,
        #     'attention_mask': og_attention_mask,
        #     'aug_input_ids': aug_input_ids,
        #     'aug_attention_mask': aug_attention_mask,
        #     'golden_text': goldens,
        #     'numeric_label': numeric_labels
        # }
        
        #return batch_output

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                #save all batch elements to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                #golden_text = batch['golden_text']
                labels = batch["labels"]
                # Tokenize golden labels (for CE Loss)
                # label_enc = self.tokenizer(golden_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                # labels = label_enc.input_ids.to(self.device)
                # labels[labels == self.tokenizer.pad_token_id] = -100

                #outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                #ce_loss = outputs.loss

                # Generate predictions
                with torch.no_grad():
                  generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=16)
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                #ref_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                masked_labels = torch.where(labels != -100, labels, torch.tensor(self.tokenizer.pad_token_id).to(labels.device))
                ref_texts = self.tokenizer.batch_decode(masked_labels, skip_special_tokens=True)

                # if self.compare_type == "augmented":
                #     aug_input_ids = batch['aug_input_ids']
                #     ref_texts = self.tokenizer.batch_decode(aug_input_ids, skip_special_tokens=True)
                # else:
                #     ref_texts = golden_text

                bart_scores = self.bart_scorer.score(generated_texts, ref_texts)
                bart_loss = 1 - torch.tensor(bart_scores).mean().to(self.device)
                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    ce_loss = outputs.loss

                    loss = self.alpha * ce_loss + (1 - self.alpha) * bart_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                torch.cuda.empty_cache()

                self.optimizer.zero_grad()

                #loss = self.alpha * ce_loss + (1 - self.alpha) * bart_loss

                #loss.backward()
                #self.optimizer.step()
                #self.optimizer.zero_grad()

                total_loss += loss.item()
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        #torch.save(self.model.state_dict(), "mentallama_dreadit_finetuned.pt")
        torch.save(self.model.state_dict(), "/content/drive/MyDrive/ANLP-Project/mentallama_dr.pt")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MentaLLaMA on mental health explanations")
    parser.add_argument('--model_name', type=str, required=True, help="Path to pretrained model or HF model ID")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the training CSV folder")
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

    trainer.train()


if __name__ == "__main__":
    main()
                


    