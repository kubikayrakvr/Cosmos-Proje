from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load your Model and Tokenizer
model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2-medium")


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)    # model yuklendi 
model.resize_token_embeddings(len(tokenizer))  # modelin embedding matrixinin row sayisiyla tokenizerin row sayisi arasinda uyusmazlik olmasin 
model.config.pad_token_id = tokenizer.pad_token_id

# 2. Load your Dataset
# It must have 'question' and 'answer' columns
dataset = load_dataset("onur48/MetaMathQA-Turkish-corrected", split="train")                

# 3. DEFINE THE "RECIPE" (Updated for Question + Answer)
                                            # 23-56 line arasinda sadece formatlama isleri yapiliyo
question_col = ""                           # normalde data setten Question ve Answer baslikli column bekliyodu
answer_col = ""                             # artik possible_name olarak girilen tum inputlari bekliyo oraya ekleme yapmak cok daha kolay
# Look for common names for the "Question" part
for possible_name in ["question", "query", "instruction", "input"]:
    if possible_name in dataset.column_names:
        question_col = possible_name
        break
# Look for common names for the "Answer" part
for possible_name in ["answer", "response", "output", "target"]:
    if possible_name in dataset.column_names:
        answer_col = possible_name
        break
# Verification
print(f"Detected Question Column: '{question_col}'")
print(f"Detected Answer Column:   '{answer_col}'")
# Stop if we couldn't find them
if not question_col or not answer_col:
    raise ValueError("Could not automatically find question/answer columns!")
# 4. DEFINE THE GENERIC FUNCTION
def formatting_prompts_func(example):
    output_texts = []
    for q, a in zip(example[question_col], example[answer_col]):
        formatted_text = f"### Question:\n{q}\n\n### Answer:\n{a}" + tokenizer.eos_token 
        output_texts.append(formatted_text)
        
    return output_texts

# 5. Configure the Training
sft_config = SFTConfig(
    output_dir="./results",
    packing=True,           # bunu true yaparak kisa cumleleri gereksiz paddingle doldurmak yerine uc uca ekleyip modeli gereksiz yormuyoruz
    max_seq_length=1024,       # packkingin optimum boyutu modele ve datasete bagli degisiyo GPT-2 modelleri icin 1024 en uygunu gibi gozukuyo 
    per_device_train_batch_size=2,  # dusuk batch kullanip daha az vram kullanarak ve daha az noise ureterek hesaplama yapiyo
    gradient_accumulation_steps=8,  # her 2 adimda updatelemek yerine 8 defa 2li agirlik alip ortalamalarina gore kendini updateliyo AdaGrad
    learning_rate=2e-5,
    num_train_epochs=3,     # burdaki 3luyle duruma gore oynayabiliriz
    logging_steps=10, 
    fp16=True,                      # normalde 32bitlik fp kullanilirken bunu 16'ya dusurup yaklasik %50 memory ve hizdan kazaniyo
)                                   # bunu her sey icin yapmiyo onemli kisimlari hala 32likte yapiyo 

# 6. Create the Trainer
trainer = SFTTrainer(    # ayarlarini yaptigimiz parametreleri yerine koyup calistirmaya geciyoz
    model=model,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer, 
)

# 6. Start Training
trainer.train()


# 7. Save the Final Model                           # bu kismin amaci training sonucunun chat.py ye yollanmasi 
trainer.save_model("./final_model")                 # eger chat.py iptal olursa silinebilir 
tokenizer.save_pretrained("./final_model")
print("Training finished. Model saved to ./final_model")