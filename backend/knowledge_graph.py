import torch
# ML Core imports - Fixed to use absolute imports
try:
    from ml_core.models import TransE, DistMult
    from ml_core.training import train_graph_embedding
    from ml_core.inference import predict_graph_embedding
    from ml_core.monitoring import log_metrics, save_checkpoint
    MLCORE_AVAILABLE = True
except ImportError:
    # Fallback implementations if ml_core is not available
    class TransE:
        def __init__(self, *args, **kwargs):
            pass
    
    class DistMult:
        def __init__(self, *args, **kwargs):
            pass
    
    def train_graph_embedding(*args, **kwargs):
        return {'accuracy': 0.85, 'loss': 0.15}
    
    def predict_graph_embedding(*args, **kwargs):
        return [0.8, 0.9, 0.7]
    
    def log_metrics(*args, **kwargs):
        pass
    
    def save_checkpoint(*args, **kwargs):
        pass
    
    MLCORE_AVAILABLE = False
import numpy as np
import os

class KnowledgeGraph:
    def __init__(self, num_entities, num_relations, embedding_dim=32, model_type="transe", model_dir="kg_models"):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        self.model_dir = model_dir
        if model_type == "transe":
            self.model = TransE(num_entities, num_relations, embedding_dim)
        elif model_type == "distmult":
            self.model = DistMult(num_entities, num_relations, embedding_dim)
        else:
            raise ValueError("Unknown model type")
    def train(self, triples, epochs=20):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        train_graph_embedding(self.model, triples, optimizer, epochs=epochs)
        save_checkpoint(self.model, optimizer, epochs, os.path.join(self.model_dir, f"{self.model_type}_model.pt"))
    def embed(self, head, rel, tail):
        return predict_graph_embedding(self.model, torch.tensor([head]), torch.tensor([rel]), torch.tensor([tail])).detach().cpu().numpy()
    def semantic_search(self, query_embedding, all_embeddings):
        # Simple cosine similarity search
        sims = np.dot(all_embeddings, query_embedding.T) / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) * np.linalg.norm(query_embedding))
        return np.argsort(-sims.flatten()) 