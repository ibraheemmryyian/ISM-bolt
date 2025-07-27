"""
Fallback implementation for NLTK
"""
import re
from typing import List

class FallbackNLTK:
    """Simple fallback for NLTK functionality"""
    
    @staticmethod
    def word_tokenize(text: str) -> List[str]:
        """Simple word tokenization"""
                 return re.findall(r'\w+', text.lower())
    
    @staticmethod
    def sent_tokenize(text: str) -> List[str]:
        """Simple sentence tokenization"""
        return re.split(r'[.!?]+', text)

# Create module-like interface
word_tokenize = FallbackNLTK.word_tokenize
sent_tokenize = FallbackNLTK.sent_tokenize

def download(package: str):
    """Stub for NLTK downloads"""
    pass
