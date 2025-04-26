def extract_label(raw_answer_text, dataset_name):
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
        
def get_num_classes(dataset_name):
    if dataset_name == 'CAMS':
        return 6
    elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
        return 2
    elif dataset_name == 'SAD':
        return 9
    elif dataset_name == 'swmh':
        return 5
    elif dataset_name == 't-sid':
        return 4
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")