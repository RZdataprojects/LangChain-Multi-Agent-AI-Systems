from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.llms import HuggingFacePipeline
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


def initialize_hugging_face_models(model_name: str, hugging_face_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hugging_face_token)
    hugging_face_model = AutoModelForCausalLM.from_pretrained(model_name, token=hugging_face_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hugging_face_model.to(device)
    pipe = pipeline("text-generation",
                    model=hugging_face_model,
                    tokenizer=tokenizer,
                    max_new_tokens=20,
                    temperature=0.9,
                    do_sample=True,
                    num_return_sequences=1)
    hfp = HuggingFacePipeline(pipeline=pipe)
    return hfp


def initialize_mistral(hugging_face_token: str):
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    return initialize_hugging_face_models(model_name, hugging_face_token)


def initialize_gemini(google_api_key: str):
    return ChatGoogleGenerativeAI(model="gemini-pro",
                                  google_api_key=google_api_key, 
                                  temperature=0.5, 
                                  top_p=1,
                                  top_k=1,
                                  max_output_tokens=1000,)