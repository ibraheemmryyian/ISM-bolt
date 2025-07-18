import torch
from transformers import BertTokenizer, BertModel
from ml_core.monitoring import log_metrics, save_checkpoint
import numpy as np
import os

class MaterialsBERTService:
    def __init__(self, model_name="bert-base-uncased", model_dir="materials_bert_models"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
    def embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    def similarity(self, text_a, text_b):
        emb_a = self.embed([text_a])[0]
        emb_b = self.embed([text_b])[0]
        return np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)) 