# from BARTScore.bart_score import BARTScorer
import pandas as pd
import os
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_augmented_data():
    test_data = {}
    for root, ds, fs in os.walk("../Augmented_Data/"):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            og_inputs = data['query'].to_list()
            aug_inputs = data['augmented_query'].to_list()
            goldens = data['gpt-3.5-turbo'].to_list()
            test_data[fn.split('.')[0]] = [og_inputs, aug_inputs, goldens]
    return test_data

test_data = load_augmented_data()

def save_responses_to_csv(dataset_name, queries, augmented_queries, 
                         responses, augmented_responses, goldens):

    output_dir = "../generated_responses/"
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        'original_query': queries,
        'augmented_query': augmented_queries,
        'original_response': responses,
        'augmented_response': augmented_responses,
        'golden_response': goldens
    }
    
    df = pd.DataFrame(data)
    
    output_path = os.path.join(output_dir, f"{dataset_name}_responses.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved responses for {dataset_name} to {output_path}")

def generate_response(model, tokenizer, queries, batch_size=5):
    responses = []
    for i in range(0, len(queries), batch_size):
        batch_data = queries[i: min(i+batch_size, len(queries))]
        #print(batch_data[:2])
        inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
        #print(inputs)
        #final_input = inputs.input_ids
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        #print(final_input)
        generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=2048)
        for j in range(generate_ids.shape[0]):
            truc_ids = generate_ids[j][len(input_ids[j]) :]
            response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            responses.append(response)
        print(i)
    return responses

def generate_all_responses(model, tokenizer, test_data, device, batch_size):
    generated_text = {}
    augmented_generated_text = {}
    goldens = {}

    model.to(device)

    for dataset_name in test_data.keys():
        #if dataset_name not in ['DR', 'dreaddit']:
        #    continue
        print('Generating for dataset: {}'.format(dataset_name))
        queries, augmented_queries, goldens = test_data[dataset_name]
        goldens[dataset_name]  = goldens
        
        responses = generate_response(model, tokenizer, queries, batch_size)
        generated_text[dataset_name] = responses
        
        augmented_responses = generate_response(model, tokenizer, augmented_queries, batch_size)
        augmented_generated_text[dataset_name] = augmented_responses
        save_responses_to_csv(dataset_name, queries, augmented_queries, responses, augmented_responses, goldens)

    return generated_text, augmented_generated_text, goldens


def bart_augment_score(bart_scorer, og_responses, augmented_responses):
    
    #both ways = bidirectional
    augmented_to_original_scores = bart_scorer.score(augmented_responses, og_responses)
    original_to_augmented_scores = bart_scorer.score(og_responses, augmented_responses)
     
    response_similarity_scores = [(a + o) / 2 for a, o in 
                                    zip(augmented_to_original_scores, original_to_augmented_scores)]
    
    avg_response_similarity = sum(response_similarity_scores) / len(response_similarity_scores)
    print("Average Response Similarity: ", avg_response_similarity)
    return avg_response_similarity, response_similarity_scores

def bart_golden_score(bart_scorer, og_responses, goldens):
    og_to_golden_scores = bart_scorer.score(og_responses, goldens)
    avg_score = sum(og_to_golden_scores) / len(og_to_golden_scores)
    print("Average Golden Similarity: ", avg_score)
    return avg_score, og_to_golden_scores

    
def generate_main(datasets, model_path, load_custom_pretrained=False, custom_pretrained_path=None):
    from transformers import LlamaForCausalLM, LlamaTokenizer
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side='left')
    if load_custom_pretrained:
        model.load_state_dict(torch.load(custom_pretrained_path, map_location=device))
        
    test_data = load_augmented_data()
    test_data = {k: test_data[k] for k in datasets}
    generated_text, augmented_generated_text, goldens = generate_all_responses(model, tokenizer, test_data, device, batch_size=5)

def score_augments_main(datasets):
    from BARTScore.bart_score import BARTScorer
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart_score.pth')
    
    for dataset in datasets:
        data = pd.read_csv(os.path.join("../generated_responses/", f"{dataset}_responses.csv"))
        og_responses = [str(r) for r in data['original_response'].to_list()]
        augmented_responses = data['augmented_response'].to_list()
        avg_response_similarity, response_similarity_scores = bart_augment_score(bart_scorer, og_responses, augmented_responses)
        data['golden_comparison_scores'] = response_similarity_scores
        data.to_csv(f"../generated_responses/{dataset}_augmented_comparison.csv", index=False)

def score_golden_main(datasets):
    from BARTScore.bart_score import BARTScorer
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart_score.pth')
    
    for dataset in datasets:
        data = pd.read_csv(os.path.join("../generated_responses/", f"{dataset}_responses.csv"))
        og_responses = [str(r) for r in data['original_response'].to_list()]
        goldens = data['augmented_response'].to_list()
        avg_score, og_to_golden_scores = bart_golden_score(bart_scorer, og_responses, goldens)
        data['golden_comparison_scores'] = og_to_golden_scores
        data.to_csv(f"../generated_responses/{dataset}_golden_comparison.csv", index=False)
    
    
    
def main(datasets, model_path, option, load_custom_pretrained, custom_pretrained_path=None):
    if option == "generate":
        generate_main(datasets, model_path, load_custom_pretrained, custom_pretrained_path)
    elif option == "augment_score":
        score_augments_main(datasets)
    elif option == "golden_score":
        score_golden_main(datasets)
    else:
        raise ValueError("Invalid option. Choose 'generate' or 'augment_score' or 'golden_score'.")
    
    
    

print(test_data.keys())
print(test_data["dreaddit_augmented"][:10])