import numpy as np
import pandas as pd
import random
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import os
import torch
from typing import List, Dict, Optional, Union, Tuple
import warnings

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

# Get stopwords
stop_words = set(stopwords.words('english'))

class MentalHealthAugmenter:
    """
    Implementation of data augmentation techniques from the ICON 2021 paper,
    specifically using AugBERT for Dreaddit and AugEDA for DR and SAD datasets.
    
    EDA implementation follows the original paper:
    "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"
    (Wei and Zou, 2019)
    """
    
    def __init__(self, preserve_label=True, preserve_keywords=True):
        """
        Initialize the augmenter with parameters
        
        Args:
            preserve_label: Whether to preserve label-related words
            preserve_keywords: Whether to preserve mental health domain keywords
        """
        self.preserve_label = preserve_label
        self.preserve_keywords = preserve_keywords
        
        # Mental health domain keywords to preserve
        self.domain_keywords = set([
            'depression', 'anxiety', 'suicide', 'bipolar', 'ptsd', 'stress',
            'mental', 'health', 'disorder', 'therapy', 'counseling', 'psychosis',
            'panic', 'trauma', 'attack', 'medication', 'diagnosis', 'depressed',
            'anxious', 'stressed', 'sad', 'worried', 'hopeless', 'suicidal',
            'mood', 'emotional', 'family', 'financial', 'work', 'school',
            'social', 'relationship', 'sleep', 'insomnia'
        ])
        
        # Label-specific keywords for each dataset
        self.label_keywords = {
            'DR': ['yes', 'no', 'depression'],
            'dreaddit': ['yes', 'no', 'stress', 'stressful'],
            'SAD': ['school', 'financial', 'family', 'social', 'work', 'health', 
                    'emotional', 'everyday', 'decision', 'problem', 'issue']
        }
        
        # Load BERT model for AugBERT
        try:
            from transformers import BertTokenizer, BertForMaskedLM
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.cuda()
            self.bert_model.eval()
            self.bert_available = True
        except Exception as e:
            warnings.warn(f"Failed to load BERT model for AugBERT: {e}. Will fall back to EDA for all datasets.")
            self.bert_available = False
    
    def _get_synonyms(self, word):
        """Get synonyms for a word using WordNet"""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syn_word = lemma.name().replace('_', ' ')
                if syn_word.lower() != word.lower() and syn_word not in synonyms:
                    synonyms.append(syn_word)
        return synonyms
    
    def _should_modify(self, word, dataset_name=None):
        """Check if a word should be modified based on preservation rules"""
        word_lower = word.lower()
        
        # Skip stop words (following original EDA implementation)
        if word_lower in stop_words:
            return False
        
        # Skip short words, punctuation, or non-alphabetic words
        if len(word) <= 3 or not word.isalpha():
            return False
        
        # Skip domain keywords if preserve_keywords is True
        if self.preserve_keywords and word_lower in self.domain_keywords:
            return False
        
        # Skip label keywords if preserve_label is True and dataset_name is provided
        if self.preserve_label and dataset_name and dataset_name in self.label_keywords:
            if word_lower in self.label_keywords[dataset_name]:
                return False
        
        return True

    def eda_synonym_replacement(self, text, dataset_name=None, alpha=0.1):
        """
        EDA technique 1: Synonym Replacement
        Randomly replace words with their synonyms
        
        Args:
            text: Text to augment
            dataset_name: Name of the dataset
            alpha: Percentage of words to replace
            
        Returns:
            Augmented text
        """
        words = word_tokenize(text)
        num_to_replace = max(1, int(len(words) * alpha))
        
        # Find all words that can be replaced (not stop words and modifiable)
        replaceable_indices = [
            i for i, word in enumerate(words) 
            if self._should_modify(word, dataset_name)
        ]
        
        # If no words can be replaced, return original text
        if not replaceable_indices:
            return text
        
        # Limit the number of replacements
        num_to_replace = min(num_to_replace, len(replaceable_indices))
        replace_indices = random.sample(replaceable_indices, num_to_replace)
        
        # Replace selected words with synonyms
        new_words = words.copy()
        for idx in replace_indices:
            synonyms = self._get_synonyms(words[idx])
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)
    
    def eda_random_insertion(self, text, dataset_name=None, alpha=0.1):
        """
        EDA technique 2: Random Insertion
        Find random synonyms of words in the text and insert them
        
        Args:
            text: Text to augment
            dataset_name: Name of the dataset
            alpha: Number of words to insert as a percentage of the text length
            
        Returns:
            Augmented text
        """
        words = word_tokenize(text)
        num_to_insert = max(1, int(len(words) * alpha))
        
        # Find all words that have synonyms (not stop words and modifiable)
        candidate_words = [
            word for i, word in enumerate(words) 
            if self._should_modify(word, dataset_name)
        ]
        
        # If no words have synonyms, return original text
        if not candidate_words:
            return text
        
        new_words = words.copy()
        
        for _ in range(num_to_insert):
            # Get random word from the text
            rand_word = random.choice(candidate_words)
            synonyms = self._get_synonyms(rand_word)
            
            if not synonyms:
                continue
            
            # Insert a random synonym at a random position
            synonym = random.choice(synonyms)
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, synonym)
        
        return ' '.join(new_words)
    
    def eda_random_swap(self, text, dataset_name=None, alpha=0.1):
        """
        EDA technique 3: Random Swap
        Randomly swap words in the text
        
        Args:
            text: Text to augment
            dataset_name: Name of the dataset
            alpha: Percentage of words to swap
            
        Returns:
            Augmented text
        """
        words = word_tokenize(text)
        num_to_swap = max(1, int(len(words) * alpha))
        
        # If text is too short, return original
        if len(words) <= 3:
            return text
        
        new_words = words.copy()
        
        for _ in range(num_to_swap):
            # Get two random positions
            pos1, pos2 = random.sample(range(len(new_words)), 2)
            # Swap the words
            new_words[pos1], new_words[pos2] = new_words[pos2], new_words[pos1]
        
        return ' '.join(new_words)
    
    def eda_random_deletion(self, text, dataset_name=None, alpha=0.1):
        """
        EDA technique 4: Random Deletion
        Randomly delete words from the text
        
        Args:
            text: Text to augment
            dataset_name: Name of the dataset
            alpha: Probability of each word being deleted
            
        Returns:
            Augmented text
        """
        words = word_tokenize(text)
        
        # If text is too short, return original
        if len(words) <= 5:
            return text
        
        # Determine which words to keep - following original EDA implementation
        # but with added domain-specific preservation
        new_words = []
        for word in words:
            # Keep word if:
            # 1. It's a domain keyword we want to preserve
            # 2. It's a label keyword we want to preserve
            # 3. Random chance (1-alpha) determines we keep it
            keep_word = ((self.preserve_keywords and word.lower() in self.domain_keywords) or
                         (self.preserve_label and dataset_name in self.label_keywords and 
                          word.lower() in self.label_keywords[dataset_name]) or
                         random.random() > alpha)
            
            if keep_word:
                new_words.append(word)
        
        # If we deleted too many words, keep a minimum percentage (50%)
        if len(new_words) < 0.5 * len(words):
            num_to_keep = int(0.5 * len(words))
            new_words = random.sample(words, num_to_keep)
        
        # If we deleted all words, return a random word
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augeda(self, text, dataset_name=None, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1):
        """
        Apply AugEDA - applying all four operations sequentially
        Following the original EDA paper implementation
        
        Args:
            text: Text to augment
            dataset_name: Name of the dataset
            alpha_sr: Parameter for synonym replacement
            alpha_ri: Parameter for random insertion
            alpha_rs: Parameter for random swap
            alpha_rd: Parameter for random deletion
            
        Returns:
            Augmented text
        """
        # Apply all operations sequentially with the given alpha parameters
        # This matches the original EDA paper implementation
        
        # 1. Synonym Replacement
        augmented = self.eda_synonym_replacement(text, dataset_name, alpha_sr)
        
        # 2. Random Insertion
        augmented = self.eda_random_insertion(augmented, dataset_name, alpha_ri)
        
        # 3. Random Swap
        augmented = self.eda_random_swap(augmented, dataset_name, alpha_rs)
        
        # 4. Random Deletion
        augmented = self.eda_random_deletion(augmented, dataset_name, alpha_rd)
        
        return augmented
    
    def augbert(self, text, dataset_name=None, mask_prob=0.15, top_k=10):
        """
        Apply AugBERT augmentation (BERT-based word replacement)
        According to the paper, this was most effective for Dreaddit dataset
        
        Args:
            text: Text to augment
            dataset_name: Name of the dataset
            mask_prob: Probability of masking each token
            top_k: Number of top predictions to sample from
            
        Returns:
            Augmented text
        """
        if not self.bert_available:
            # Fall back to AugEDA if BERT is not available
            return self.augeda(text, dataset_name)
        
        # Tokenize the text
        tokenized = self.bert_tokenizer.tokenize(text)
        
        # Skip if text is too short
        if len(tokenized) <= 3:
            return text
        
        # Determine which tokens to mask
        mask_indices = []
        for i, token in enumerate(tokenized):
            # Skip special tokens and tokens we want to preserve
            if token.startswith('##') or not self._should_modify(token, dataset_name):
                continue
            
            # Mask with probability mask_prob
            if random.random() < mask_prob:
                mask_indices.append(i)
        
        # If no tokens to mask, return original text
        if not mask_indices:
            return text
        
        # Make a copy of the tokens for masking
        masked_tokens = tokenized.copy()
        
        # Create input tensor
        for i in mask_indices:
            masked_tokens[i] = '[MASK]'
        
        # Convert to IDs
        tokens_ids = self.bert_tokenizer.convert_tokens_to_ids(masked_tokens)
        tokens_tensor = torch.tensor([tokens_ids])
        
        # Move to GPU if available
        if torch.cuda.is_available():
            tokens_tensor = tokens_tensor.cuda()
        
        # Get BERT predictions
        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor)
            predictions = outputs.logits
        
        # Replace masked tokens with predictions
        for i, mask_idx in enumerate(mask_indices):
            # Get predicted token probabilities
            predicted_logits = predictions[0, mask_idx]
            
            # Get top k predictions
            top_k_values, top_k_indices = torch.topk(predicted_logits, top_k)
            
            # Convert to probabilities
            top_k_probs = torch.nn.functional.softmax(top_k_values, dim=0)
            
            # Sample from top k
            sampled_idx = top_k_indices[torch.multinomial(top_k_probs, 1).item()]
            
            # Replace the masked token
            tokenized[mask_idx] = self.bert_tokenizer.convert_ids_to_tokens([sampled_idx.item()])[0]
        
        # Convert tokens back to text
        augmented_text = self.bert_tokenizer.convert_tokens_to_string(tokenized)
        
        return augmented_text
    
    def augment_text(self, text, dataset_name, num_augmentations=1):
        """
        Create augmented versions of a text using the most effective technique for each dataset
        
        Args:
            text: Text to augment
            dataset_name: Name of the dataset (must be provided)
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented texts
        """
        if not dataset_name:
            raise ValueError("Dataset name must be provided for targeted augmentation")
        
        augmented_texts = []
        
        # Apply the appropriate augmentation technique based on dataset
        for _ in range(num_augmentations):
            if dataset_name.lower() == 'dreaddit':
                # Use AugBERT for Dreaddit
                augmented = self.augbert(text, dataset_name)
            else:
                # Use AugEDA for DR and SAD
                augmented = self.augeda(
                    text, 
                    dataset_name,
                    alpha_sr=0.1,  # 10% of words for synonym replacement
                    alpha_ri=0.1,  # 10% of text length for insertions
                    alpha_rs=0.1,  # 10% of words swapped
                    alpha_rd=0.1   # 10% chance of each word being deleted
                )
            
            augmented_texts.append(augmented)
        
        return augmented_texts
    
    def augment_dataset(self, df, text_column='query', dataset_name=None, augmentation_ratio=1):
        """
        Augment all texts in a dataset, doubling the dataset size as recommended in the paper
        
        Args:
            df: DataFrame containing the dataset
            text_column: Column containing the text to augment
            dataset_name: Name of the dataset
            augmentation_ratio: Number of augmented samples to generate per original sample
            
        Returns:
            DataFrame with original and augmented data combined
        """
        if dataset_name is None:
            # Try to infer dataset name if not provided
            if hasattr(df, 'filepath') and df.filepath:
                filename = os.path.basename(df.filepath)
                if filename.startswith('DR'):
                    dataset_name = 'DR'
                elif filename.startswith('dreaddit'):
                    dataset_name = 'dreaddit'
                elif filename.startswith('SAD'):
                    dataset_name = 'SAD'
        
        if not dataset_name:
            raise ValueError("Dataset name must be provided for targeted augmentation")
        
        # Get original data
        original_df = df.copy()
        
        # Create augmented versions
        all_augmented_dfs = []
        for i in range(augmentation_ratio):
            augmented_df = original_df.copy()
            
            # Apply augmentation to each row
            augmented_texts = []
            for text in augmented_df[text_column]:
                augmented = self.augment_text(text, dataset_name, num_augmentations=1)[0]
                augmented_texts.append(augmented)
            
            # Add augmented column and mark as augmented
            augmented_df['augmented_query'] = augmented_texts
            augmented_df['is_augmented'] = 1
            
            all_augmented_dfs.append(augmented_df)
        
        # Mark original data
        original_df['augmented_query'] = original_df[text_column]
        original_df['is_augmented'] = 0
        
        # Combine original and augmented data
        combined_df = pd.concat([original_df] + all_augmented_dfs, ignore_index=True)
        
        return combined_df


def augment_selected_datasets(datasets_dir, output_dir, selected_datasets=None, augmentation_ratio=1):
    """
    Augment selected datasets from the specified directory using dataset-specific techniques
    
    Args:
        datasets_dir: Directory containing datasets to augment
        output_dir: Directory to save augmented datasets
        selected_datasets: List of dataset names to augment (defaults to DR, dreaddit, SAD)
        augmentation_ratio: Number of augmented samples to generate per original sample
    """
    if selected_datasets is None:
        selected_datasets = ['DR', 'dreaddit', 'SAD']
    
    # Create augmenter
    augmenter = MentalHealthAugmenter()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process datasets
    for root, _, files in os.walk(datasets_dir):
        for file in files:
            if file.endswith('.csv'):
                # Check if this file belongs to one of the selected datasets
                dataset_match = None
                for dataset in selected_datasets:
                    if dataset.lower() in file.lower():
                        dataset_match = dataset
                        break
                
                if not dataset_match:
                    continue
                
                # Get file paths
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, datasets_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # Create output subdirectory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                print(f"Augmenting {dataset_match} dataset: {input_path}")
                print(f"Using {'AugBERT' if dataset_match.lower() == 'dreaddit' else 'AugEDA'}")
                
                # Read and augment the dataset
                df = pd.read_csv(input_path)
                if 'query' not in df.columns:
                    # Try to find a suitable text column
                    text_col = None
                    for col in df.columns:
                        if col.lower() in ['text', 'content', 'post', 'message']:
                            text_col = col
                            break
                    
                    if text_col is None:
                        print(f"Warning: No suitable text column found in {file}. Skipping.")
                        continue
                else:
                    text_col = 'query'
                
                # Augment the dataset
                augmented_df = augmenter.augment_dataset(
                    df, 
                    text_column=text_col, 
                    dataset_name=dataset_match,
                    augmentation_ratio=augmentation_ratio
                )
                
                # Save augmented dataset
                augmented_df.to_csv(output_path, index=False)
                print(f"Saved augmented dataset to {output_path}")
                print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment mental health datasets using techniques from ICON 2021 paper")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing datasets to augment")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save augmented datasets")
    parser.add_argument('--datasets', nargs='+', default=['DR', 'dreaddit', 'SAD'], help="Dataset names to augment")
    parser.add_argument('--augmentation_ratio', type=int, default=1, help="Number of augmented samples per original sample")
    
    args = parser.parse_args()
    
    augment_selected_datasets(
        args.input_dir, 
        args.output_dir, 
        args.datasets,
        args.augmentation_ratio
    )