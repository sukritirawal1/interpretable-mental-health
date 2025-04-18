import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from IMHI import load_instruction_test_data

def generate_responses(model, tokenizer, queries, batch_size, device):
    responses = []
    for i in range(0, len(queries), batch_size):
        batch_data = queries[i:min(i + batch_size, len(queries))]
        inputs = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2048, do_sample = False, top_p = 0.0, num_return_sequences=1)
        batch_responses = []
        for j in range(outputs.shape[0]):
            # Get the part of the output that comes after the input
            outputs_j = outputs[j][len(input_ids[j]):]
            batch_responses.append(outputs_j)
        
        # Use batch_decode correctly to decode all responses in this batch at once
        decoded_texts = tokenizer.batch_decode(batch_responses, skip_special_tokens=True, spaces_between_special_tokens=False)
        print(decoded_texts)
        responses.extend(decoded_texts)
        
    return responses


def generate_responses_all_datasets(model, tokenizer, test_data, device, keys = ['dreaddit'], batch_size = 5):
    generated_text = {}
    golden_all = {}
    for key in keys:
        queries, goldens = test_data[key]
        print(goldens)
        golden_all[key] = goldens
        generated_text[key] = generate_responses(model, tokenizer, queries, batch_size, device)
        
    return generated_text, golden_all
        
def main():
    test_data = load_instruction_test_data()

    MODEL_ID = "klyang/MentaLLaMA-chat-7B"
    MODEL_OUTPUT_DIR = "mental_llama_7b_2"
    print(MODEL_OUTPUT_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LlamaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    model.to(device)
    
    generated_text, golden_all = generate_responses_all_datasets(model, tokenizer, test_data, device, keys = ['swmh', 't-sid', 'loneliness'], batch_size = 5)
    
    os.makedirs(f"../model_output/{MODEL_OUTPUT_DIR}", exist_ok=True)
    
    for dataset_name in generated_text.keys():
        output = {
            'goldens': golden_all[dataset_name], 
            'generated_text': generated_text[dataset_name]
        }
        
        output_df = pd.DataFrame(output)
        
        output_df.to_csv(
            f"../model_output/{MODEL_OUTPUT_DIR}/{dataset_name}.csv", 
            index=False, 
            escapechar='\\'
        )
    print(f"Saved to model_output/{MODEL_OUTPUT_DIR}")
    
if __name__ == "__main__":
    main()