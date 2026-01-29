import torch
import math
from transformers import AutoTokenizer, GPT2LMHeadModel
from torch.nn import functional as F

def test_pretrained_model():
    model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
    # model_name = "./my_custom_model"
    print(f"--- Loading {model_name} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model.eval()
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return


    print("--- Model Configuration Check ---")
    print(f"Vocab Size: {model.config.vocab_size}")
    print(f"Hidden Size: {model.config.n_embd}")
    print(f"Max Position Embeddings: {model.config.n_positions}\n")

    print("--- Next Token Prediction Logic Test ---")
    input_text = "CosmosTech şirketi 2050 yılında"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits[0, -1, :] # Get logits of the last token
        probs = F.softmax(predictions, dim=-1)
        
        # Get top 5 predictions
        top_k_probs, top_k_indices = torch.topk(probs, 5)
        
        print(f"Input Context: '{input_text}'")
        print("Top 5 predicted next tokens:")
        for i in range(5):
            token = tokenizer.decode([top_k_indices[i].item()])
            prob = top_k_probs[i].item()
            print(f"  {i+1}. '{token}' (Probability: {prob:.4f})")
    print("\n")

    # 4. PERPLEXITY CALCULATION (Quantitative Metric)
    print("--- Perplexity Evaluation (Sanity Check) ---")
    eval_text = "Yapay zeka teknolojileri son yıllarda büyük bir hızla gelişmektedir."
    
    encodings = tokenizer(eval_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        
    print(f"Test Sentence: '{eval_text}'")
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")
    print("(Note: Lower perplexity indicates the model finds this sentence natural.)\n")

    print("--- Generation Quality Test ---")
    prompt = "Türkiye'nin başkenti Ankara,"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    sample_outputs = model.generate(
        inputs.input_ids,
        do_sample=True, 
        max_length=50, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    test_pretrained_model()