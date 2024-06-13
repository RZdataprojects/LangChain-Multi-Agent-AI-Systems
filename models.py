from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def initialize_hugging_face_models(model_name: str, hugging_face_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hugging_face_token)
    hugging_face_model = AutoModelForCausalLM.from_pretrained(model_name, token=hugging_face_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hugging_face_model.to(device)
    return hugging_face_model, tokenizer


def initialize_llama3(hugging_face_token: str):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    return initialize_hugging_face_models(model_name, hugging_face_token)


def initialize_gemma(hugging_face_token: str):
    model_name = "google/gemma-7b-it"
    return initialize_hugging_face_models(model_name, hugging_face_token)


def get_response_google_gemma(user_prompt: str, model, hugging_face_model, tokenizer, verbose=1):
    if verbose:
        print(model + ': ', user_prompt, end=' ')

    # Input text
    input_text = """<start_of_turn>user
    You are a helpful assistant. Answer the question without asking for additional information. 
    User's question: {BODY}<end_of_turn>
    <start_of_turn>model
    """.format(BODY=user_prompt)

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate text
    output = hugging_face_model.generate(input_ids, temperature=0.5, do_sample=True,
                                         max_new_tokens=1000, num_return_sequences=1)

    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True).split('model\n')[1].strip()

    # Clear CUDA memory
    clear_gpu_memory()

    if verbose==1:
        print('Done.')
    return generated_text


def get_response_meta_llama(user_prompt: str, model='llama-2', verbose=1, hugging_face_model=None, tokenizer=None):
    if verbose:
        print(model + ': ', user_prompt, end=' ')
        messages = [
            {"role": "system", "content": """You are a helpful assistant. 
            Answer the question without asking for additional information."""},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt"
                                                ).to(device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = hugging_face_model.generate(
            input_ids,
            max_new_tokens=1000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.5
        )

        # Decode generated text
        response = output[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(response, skip_special_tokens=True)
        print(generated_text)

    # Clear CUDA memory
    clear_gpu_memory()

    if verbose == 1:
        print('Done.')
    return generated_text