import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

class ExplanationDataset(Dataset):
    def __init__(self, data_path = "data_path.csv", datasets = None):
        self.data_path = data_path
        test_data = self.load_augmented_data()
        if datasets:
            self.datasets = datasets
        else:
            self.datasets = test_data.keys()
        
        self.all_texts = []
        self.all_augs = []
        self.all_goldens = []
        self.all_labels = []
        self.dataset_names = []
        for dataset in test_data.keys():
            self.all_texts.extend(test_data[dataset][0])
            self.all_augs.extend(test_data[dataset][1])
            
            goldens = test_data[dataset][2]
            self.all_goldens.extend(goldens)
            
            labels = [self.extract_label(g, dataset) for g in goldens]
            self.all_labels.extend(labels)
            
            self.dataset_names.extend([dataset] * len(goldens))
            
    def __len__(self):
        return len(self.all_texts)
    
    def __getitem__(self, idx):
        #return = original_text, augmented_text, golden_text, numeric_label, dataset_name
        return self.all_texts[idx], self.all_augs[idx], self.all_goldens[idx], self.all_labels[idx], self.dataset_names[idx]
    
    def load_augmented_data(self):
        test_data = {}
        if os.path.isfile(self.data_path):
            data = pd.read_csv(self.data_path)
            texts = data['query'].tolist()
            if 'augmented_query' in data.columns:
                augmented_texts = data['augmented_query'].tolist()
            else:
                augmented_texts = data['query'].tolist()
            labels = data['gpt-3.5-turbo'].tolist()
    
            dataset_name = os.path.basename(os.path.dirname(self.data_path))  # get folder name like 'DR' or 'dreaddit'
            test_data[dataset_name] = [texts, augmented_texts, labels]
            
        else:
            #for root, ds, fs in os.walk("../test_data/test_instruction"): #update based on shambhavi code
            for root, ds, fs in os.walk(self.data_path):
                for fn in fs:
                    data = pd.read_csv(os.path.join(root, fn))
                    texts = data['query'].to_list()
                    if 'augmented_query' in data.columns:
                        augmented_texts = data['augmented_query'].tolist() #update based on shambhavi code
                    else:
                        augmented_texts = data['query'].tolist()
                    labels = data['gpt-3.5-turbo'].to_list()
                    test_data[fn.split('.')[0]] = [texts, augmented_texts, labels] #update based on shambhavi code
        return test_data
    
    def extract_label(self, raw_answer_text, dataset_name):
        print("DATASET NAME IS:", dataset_name)
        if "Reasoning:" in raw_answer_text:
            answer_text = raw_answer_text.split("Reasoning:")[0].strip()
        else:
            answer_text = raw_answer_text
        
        # Process based on dataset type
        if dataset_name == 'swmh':
            if 'no mental' in answer_text.lower():
                return 0
            elif 'suicide' in answer_text.lower():
                return 1
            elif 'depression' in answer_text.lower():
                return 2
            elif 'anxiety' in answer_text.lower():
                return 3
            elif 'bipolar' in answer_text.lower():
                return 4
            else:
                return -1  # Unknown label
        
        elif dataset_name == 't-sid':
            if 'depression' in answer_text.lower():
                return 2
            elif 'suicide' in answer_text.lower():
                return 1
            elif 'ptsd' in answer_text.lower():
                return 3
            elif 'control' in answer_text.lower() or 'no mental' in answer_text.lower():
                return 0
            else:
                return -1  # Unknown label
        elif dataset_name == 'SAD':
            if 'school' in answer_text.lower():
                return 0
            elif 'financial' in answer_text.lower():
                return 1
            elif 'family' in answer_text.lower():
                return 2
            elif 'social' in answer_text.lower():
                return 3
            elif 'work' in answer_text.lower():
                return 4
            elif 'health' in answer_text.lower():
                return 5
            elif 'emotional' in answer_text.lower():
                return 6
            elif 'decision' in answer_text.lower():
                return 7
            else:
                return 8
            
        elif dataset_name == 'CAMS':
            if 'none' in answer_text.lower():
                return 0
            elif 'bias' in answer_text.lower():
                return 1
            elif 'jobs' in answer_text.lower():
                return 2
            elif 'medication' in answer_text.lower():
                return 3
            elif 'relationship' in answer_text.lower():
                return 4
            elif 'alienation' in answer_text.lower():
                return 5
            else: return 0
                
        elif dataset_name in ['DR', 'dreaddit', 'loneliness']:
            if 'yes' in answer_text.lower():
                return 1
            elif 'no' in answer_text.lower():
                return 0
            else:
                return -1  # Unknown label
        else:
            print("Check dataset name buddy boi")
            return None
        
        
        