import torch.nn as nn
from typing import List
import torch
from BARTScore import BARTScorer
from utils import extract_label, get_num_classes
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedBARTLoss(nn.Module):
    
    def __init__(self, dataset_name, ce_weight = 0.25, bart_weight = 0.75, device = DEVICE):
        super().__init__()
        self.bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        self.bart_scorer.load(path='bart_score.pth')
        for param in self.bart_scorer.model.parameters():
            param.requires_grad = False
        self.ce_weight = ce_weight
        self.bart_weight = bart_weight
        self.device = device
        self.dataset = dataset_name
        
    def forward(self, targets, generated_texts, golden_texts):
        
        #cross entropy loss
        gen_labels = [extract_label(g, self.dataset) for g in generated_texts]
        #gen_labels_tensor = torch.tensor(gen_labels, dtype=torch.int, device=self.device)
        gen_labels_tensor = torch.tensor(gen_labels, dtype=torch.long, device=self.device)
        targets = targets.to(torch.long)
        valid_mask = (gen_labels_tensor >= 0 ) & (targets>=0)
        targets  = targest[valid_mask]
        gen_labels_tensor = targets[valid_mask]
        if len(gen_labels_tensor) == 0:
            ce_loss = torch.tensor(0.0, device=self.device)
        else:
            num_classes = get_num_classes(self.dataset)
    
            gen_logits = F.one_hot(gen_labels_tensor, num_classes=num_classes).float()
            gen_log_probs = torch.log(gen_logits + 1e-8)  # small epsilon
            ce_loss = F.nll_loss(gen_log_probs, targets, reduction='mean')
        #num_classes = get_num_classes(self.dataset)
        #gen_one_hot = F.one_hot(gen_labels_tensor, num_classes=num_classes).float()
        #gen_one_hot = gen_one_hot.to(self.device)
        #ce_loss = F.cross_entropy(gen_one_hot, targets, reduction='mean')
        #ce_loss = F.cross_entropy(gen_labels_tensor,targets,reduction='mean')
        #BART loss
        with torch.no_grad():
            bart_scores = self.bart_scorer.score(generated_texts, golden_texts)
        bart_scores_tensor = torch.tensor(bart_scores, device=self.device)
        bart_loss = -bart_scores_tensor.mean() #subtract because we want to maximize the score
        
        #combine losses
        combined_loss = self.ce_weight * ce_loss + self.bart_weight * bart_loss
        
        return combined_loss