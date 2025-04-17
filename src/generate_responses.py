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
        
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2048, num_return_sequences=1)
        for j in range(outputs.shape[0]):
            outputs_j = outputs[j][len(input_ids[j]):]
            generated_text = tokenizer.batch_decode(outputs_j, skip_special_tokens=True, spaces_between_special_tokens=False)
            print(generated_text)
        responses.append(generated_text)
        
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
    
    #LIMIT to 10 for testing
    test_data['dreaddit'][0] = test_data['dreaddit'][0][:10]
    test_data['dreaddit'][1] = test_data['dreaddit'][1][:10]

    MODEL_ID = "klyang/MentaLLaMA-chat-7B"
    MODEL_OUTPUT_DIR = "mental_llama_7b"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LlamaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
    model.to(device)
    
    generated_text, golden_all = generate_responses_all_datasets(model, tokenizer, test_data, device, keys = ['dreaddit'], batch_size = 5)
    
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
    print(f"Saved to model_output/{MODEL_OUTPUT_DIR}/directory")
    
    # # print(test_data.shape)
    # # print(test_data[:10])
    # print(test_data.keys())
    # print(len(test_data['dreaddit'][0]))
    # print(len(test_data['dreaddit'][1]))
    # # print(test_data['dreaddit'][0])
    
if __name__ == "__main__":
    main()