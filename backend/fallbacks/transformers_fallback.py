"""
Fallback implementation for transformers
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List

class AutoTokenizer:
    """Fallback tokenizer"""
    @staticmethod
    def from_pretrained(model_name: str):
        return FallbackTokenizer()

class AutoModel:
    """Fallback model"""
    @staticmethod
    def from_pretrained(model_name: str):
        return FallbackModel()

class FallbackTokenizer:
    """Simple fallback tokenizer"""
    def __call__(self, text: str, **kwargs):
        # Simple word-based tokenization
        words = text.lower().split()
        return {
            'input_ids': torch.tensor([hash(word) % 30000 for word in words[:512]]).unsqueeze(0),
            'attention_mask': torch.ones(1, len(words[:512]))
        }

class FallbackModel(nn.Module):
    """Simple fallback transformer model"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 768)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(768, 8), 
            num_layers=6
        )
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        
        class Output:
            def __init__(self, hidden_states):
                self.last_hidden_state = hidden_states
        
        return Output(x)

class SentenceTransformer:
    """Fallback sentence transformer"""
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def encode(self, sentences):
        # Simple hash-based encoding
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = []
        for sentence in sentences:
            # Create a simple hash-based embedding
            words = sentence.lower().split()
            embedding = torch.zeros(384)  # Standard sentence transformer size
            for i, word in enumerate(words[:384]):
                embedding[i] = (hash(word) % 2000) / 2000.0
            embeddings.append(embedding.numpy())
        
        return embeddings[0] if len(embeddings) == 1 else embeddings
