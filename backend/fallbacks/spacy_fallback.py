"""
Fallback implementation for spaCy
"""
from typing import List, Any

class FallbackSpacy:
    """Simple fallback for spaCy functionality"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
    
    def __call__(self, text: str):
        """Process text"""
        return FallbackDoc(text)

class FallbackDoc:
    """Fallback document object"""
    
    def __init__(self, text: str):
        self.text = text
        self.tokens = text.split()
    
    def __iter__(self):
        for token in self.tokens:
            yield FallbackToken(token)

class FallbackToken:
    """Fallback token object"""
    
    def __init__(self, text: str):
        self.text = text
        self.lemma_ = text.lower()
        self.pos_ = "NOUN"  # Default POS
        self.is_alpha = text.isalpha()
        self.is_stop = text.lower() in ["the", "a", "an", "and", "or", "but"]

def load(model_name: str):
    """Load spaCy model"""
    return FallbackSpacy(model_name)
