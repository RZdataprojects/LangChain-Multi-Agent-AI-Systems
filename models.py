from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


def initialize_llama(hugging_face_token, device):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=hugging_face_token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=hugging_face_token)
    model.to(device)
    return model, tokenizer


def initialize_mistral(hugging_face_token, device):
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map=0)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer
    

def initialize_gemini(google_api_key):
    genai.configure(api_key=google_api_key)

    # Set up the model
    generation_config = {"temperature": 0.5, "top_p": 1, "top_k": 1, "max_output_tokens": 1000,}

    safety_settings = [
      {"category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"},
      {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"},]

    gemini_1_pro_client = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    return gemini_1_pro_client
    # return ChatGoogleGenerativeAI(model="gemini-pro",
    #                               google_api_key=google_api_key, 
    #                               temperature=0.99, 
    #                               top_p=1,
    #                               top_k=1,
    #                               max_output_tokens=1000,)


def call_llama(messages, model, tokenizer, device):
    input_ids = tokenizer.apply_chat_template(messages,
                                             add_generation_prompt=True,
                                             return_tensors="pt").to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    output = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.99,
        top_p=0.99
    )
    
    # Decode generated text
    response = output[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(response, skip_special_tokens=True)
    return generated_text


def call_mistral(messages, model, tokenizer, device):
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    outputs = model.generate(inputs,
                             max_new_tokens = 256,
                             do_sample = True,
                             temperature = 0.99,
                             top_p = 0.99)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[1].strip()
    return generated_text