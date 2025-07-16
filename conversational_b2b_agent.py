"""
Revolutionary Conversational B2B Agent for Industrial Symbiosis
LLM-powered chatbot for natural language queries and B2B matching
"""

import os
import json
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from enum import Enum
# Removed: import openai
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import threading
import queue
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of user intents"""
    GREETING = "greeting"
    MATCHING_REQUEST = "matching_request"
    MATERIAL_SEARCH = "material_search"
    COMPANY_SEARCH = "company_search"
    SUSTAINABILITY_QUERY = "sustainability_query"
    LOGISTICS_QUERY = "logistics_query"
    COST_QUERY = "cost_query"
    HELP_REQUEST = "help_request"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"

class EntityType(Enum):
    """Types of entities that can be extracted"""
    MATERIAL = "material"
    COMPANY = "company"
    LOCATION = "location"
    QUANTITY = "quantity"
    PRICE = "price"
    DATE = "date"
    INDUSTRY = "industry"

@dataclass
class Entity:
    """Extracted entity from user input"""
    type: EntityType
    value: str
    confidence: float
    start_pos: int
    end_pos: int

@dataclass
class Intent:
    """Detected user intent"""
    type: IntentType
    confidence: float
    entities: List[Entity]
    context: Dict[str, Any]

@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    user_input: str
    intent: Intent
    response: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ConversationContext:
    """Context for ongoing conversation"""
    conversation_id: str
    user_id: str
    company_id: Optional[str]
    turns: List[ConversationTurn]
    preferences: Dict[str, Any]
    current_topic: Optional[str]
    created_at: datetime
    last_updated: datetime

class DeepSeekR1ConversationalAgent:
    """Advanced conversational B2B agent using DeepSeek R1 for industrial symbiosis"""
    
    def __init__(self):
        self.api_key = "sk-7ce79f30332d45d5b3acb8968b052132"
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-r1"
        self.conversations: Dict[str, ConversationContext] = {}
        self.matching_engine = None
        self.logistics_engine = None
        
    def _make_request(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Make request to DeepSeek R1 API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"DeepSeek R1 API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek R1 request failed: {e}")
            return None
    
    def analyze_intent(self, user_input: str, context: ConversationContext) -> Intent:
        """Analyze user intent using DeepSeek R1's advanced reasoning"""
        
        prompt = f"""
        You are DeepSeek R1, an expert in natural language understanding and industrial symbiosis. Analyze the user's intent using advanced reasoning:

        USER INPUT: "{user_input}"

        CONVERSATION CONTEXT:
        - Conversation ID: {context.conversation_id}
        - User ID: {context.user_id}
        - Company ID: {context.company_id}
        - Current Topic: {context.current_topic}
        - Previous Turns: {len(context.turns)} turns
        - User Preferences: {context.preferences}

        TASK: Analyze the user's intent and extract relevant entities using DeepSeek R1's reasoning capabilities.

        INTENT TYPES:
        1. greeting - User is greeting or starting conversation
        2. matching_request - User wants to find material matches or partnerships
        3. material_search - User is searching for specific materials
        4. company_search - User is looking for companies
        5. sustainability_query - User has questions about sustainability or environmental impact
        6. logistics_query - User has questions about transportation, routes, or logistics
        7. cost_query - User has questions about costs, pricing, or economics
        8. help_request - User needs help or assistance
        9. feedback - User is providing feedback
        10. unknown - Intent cannot be determined

        ENTITY TYPES TO EXTRACT:
        - material: Industrial materials, waste, products
        - company: Company names, businesses
        - location: Geographic locations, cities, countries
        - quantity: Amounts, volumes, weights
        - price: Costs, prices, monetary values
        - date: Dates, time periods
        - industry: Industry sectors, business types

        REASONING REQUIREMENTS:
        - Use logical reasoning to understand the user's underlying intent
        - Consider conversation context and previous turns
        - Extract relevant entities with high precision
        - Consider industrial symbiosis domain knowledge
        - Handle ambiguous or unclear inputs appropriately

        Return ONLY valid JSON with this exact structure:
        {{
            "intent": {{
                "type": "intent_type_from_list_above",
                "confidence": 0.0-1.0,
                "reasoning": "detailed explanation of why this intent was chosen"
            }},
            "entities": [
                {{
                    "type": "entity_type_from_list_above",
                    "value": "extracted_value",
                    "confidence": 0.0-1.0,
                    "start_pos": 0,
                    "end_pos": 0
                }}
            ],
            "context_updates": {{
                "current_topic": "updated_topic_if_relevant",
                "preferences": {{"key": "value"}},
                "suggestions": ["suggested_next_actions"]
            }}
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are DeepSeek R1, an expert in natural language understanding for industrial symbiosis. Use your advanced reasoning to accurately analyze user intents and extract relevant entities. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_request(messages, temperature=0.2)  # Lower temperature for precise analysis
            if response:
                result = json.loads(response)
                
                # Parse intent
                intent_data = result["intent"]
                intent_type = IntentType(intent_data["type"])
                
                # Parse entities
                entities = []
                for entity_data in result["entities"]:
                    entity = Entity(
                        type=EntityType(entity_data["type"]),
                        value=entity_data["value"],
                        confidence=entity_data["confidence"],
                        start_pos=entity_data["start_pos"],
                        end_pos=entity_data["end_pos"]
                    )
                    entities.append(entity)
                
                # Create intent object
                intent = Intent(
                    type=intent_type,
                    confidence=intent_data["confidence"],
                    entities=entities,
                    context=result.get("context_updates", {})
                )
                
                return intent
            else:
                raise Exception("No response from DeepSeek R1")
                
        except Exception as e:
            logger.error(f"DeepSeek R1 intent analysis failed: {e}")
            return Intent(
                type=IntentType.UNKNOWN,
                confidence=0.0,
                entities=[],
                context={}
            )
    
    def generate_response(self, user_input: str, intent: Intent, context: ConversationContext) -> str:
        """Generate contextual response using DeepSeek R1"""
        
        # Build conversation history
        conversation_history = ""
        for turn in context.turns[-5:]:  # Last 5 turns
            conversation_history += f"User: {turn.user_input}\nAssistant: {turn.response}\n"
        
        prompt = f"""
        You are DeepSeek R1, an expert industrial symbiosis consultant and conversational AI. Generate a helpful, professional response using advanced reasoning:

        USER INPUT: "{user_input}"

        DETECTED INTENT:
        - Type: {intent.type.value}
        - Confidence: {intent.confidence:.2f}
        - Reasoning: {intent.context.get('reasoning', 'N/A')}

        EXTRACTED ENTITIES:
        {json.dumps([{"type": e.type.value, "value": e.value, "confidence": e.confidence} for e in intent.entities], indent=2)}

        CONVERSATION CONTEXT:
        - User ID: {context.user_id}
        - Company ID: {context.company_id}
        - Current Topic: {context.current_topic}
        - User Preferences: {context.preferences}
        - Conversation History: {conversation_history}

        TASK: Generate a helpful, professional response that:
        1. Addresses the user's intent appropriately
        2. Uses the extracted entities to provide relevant information
        3. Maintains conversation flow and context
        4. Provides actionable insights or next steps
        5. Demonstrates expertise in industrial symbiosis
        6. Is concise but comprehensive

        RESPONSE REQUIREMENTS:
        - Be professional and business-focused
        - Use logical reasoning to provide valuable insights
        - Consider the user's company and preferences
        - Offer specific, actionable suggestions when appropriate
        - Ask clarifying questions if needed
        - Provide relevant examples or case studies
        - Maintain a helpful, expert tone

        Generate a natural, conversational response that would be helpful for a business professional in the industrial symbiosis domain.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are DeepSeek R1, an expert industrial symbiosis consultant. Use your advanced reasoning to provide helpful, professional responses that demonstrate deep knowledge of industrial symbiosis, circular economy, and business optimization. Be conversational, expert, and actionable."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_request(messages, temperature=0.6)  # Balanced temperature for natural conversation
            if response:
                return response.strip()
            else:
                raise Exception("No response from DeepSeek R1")
                
        except Exception as e:
            logger.error(f"DeepSeek R1 response generation failed: {e}")
            return self._generate_fallback_response(intent)
    
    def _generate_fallback_response(self, intent: Intent) -> str:
        """Generate fallback response when DeepSeek R1 fails"""
        if intent.type == IntentType.GREETING:
            return "Hello! I'm your industrial symbiosis AI assistant. How can I help you find sustainable business opportunities today?"
        elif intent.type == IntentType.MATCHING_REQUEST:
            return "I'd be happy to help you find material matches and partnerships. Could you tell me more about what materials you're looking for or have available?"
        elif intent.type == IntentType.MATERIAL_SEARCH:
            return "I can help you search for materials. What specific materials are you interested in, and what are your requirements?"
        elif intent.type == IntentType.HELP_REQUEST:
            return "I'm here to help with industrial symbiosis, material matching, sustainability analysis, and logistics optimization. What would you like to know?"
        else:
            return "I understand you're interested in industrial symbiosis. Could you please provide more details about what you're looking for?"
    
    def process_message(self, user_input: str, user_id: str, company_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user message and generate response"""
        
        # Get or create conversation context
        conversation_id = f"conv_{user_id}_{datetime.now().strftime('%Y%m%d')}"
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                company_id=company_id,
                turns=[],
                preferences={},
                current_topic=None,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
        
        context = self.conversations[conversation_id]
        
        # Analyze intent
        intent = self.analyze_intent(user_input, context)
        
        # Generate response
        response = self.generate_response(user_input, intent, context)
        
        # Create conversation turn
        turn = ConversationTurn(
            user_input=user_input,
            intent=intent,
            response=response,
            timestamp=datetime.now(),
            metadata={
                "entities": [{"type": e.type.value, "value": e.value} for e in intent.entities],
                "confidence": intent.confidence
            }
        )
        
        # Update context
        context.turns.append(turn)
        context.last_updated = datetime.now()
        
        # Update context with intent insights
        if intent.context:
            if "current_topic" in intent.context:
                context.current_topic = intent.context["current_topic"]
            if "preferences" in intent.context:
                context.preferences.update(intent.context["preferences"])
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "intent": {
                "type": intent.type.value,
                "confidence": intent.confidence,
                "entities": [{"type": e.type.value, "value": e.value, "confidence": e.confidence} for e in intent.entities]
            },
            "suggestions": intent.context.get("suggestions", []),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history"""
        if conversation_id not in self.conversations:
            return []
        
        context = self.conversations[conversation_id]
        turns = context.turns[-limit:] if limit > 0 else context.turns
        
        return [
            {
                "user_input": turn.user_input,
                "response": turn.response,
                "intent": turn.intent.type.value,
                "timestamp": turn.timestamp.isoformat(),
                "metadata": turn.metadata
            }
            for turn in turns
        ]
    
    def get_conversation_analytics(self, conversation_id: str) -> Dict[str, Any]:
        """Get analytics for a conversation"""
        if conversation_id not in self.conversations:
            return {}
        
        context = self.conversations[conversation_id]
        
        # Analyze intents
        intent_counts = {}
        for turn in context.turns:
            intent_type = turn.intent.type.value
            intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1
        
        # Analyze entities
        entity_counts = {}
        for turn in context.turns:
            for entity in turn.intent.entities:
                entity_type = entity.type.value
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Calculate average confidence
        avg_confidence = np.mean([turn.intent.confidence for turn in context.turns]) if context.turns else 0
        
        return {
            "conversation_id": conversation_id,
            "total_turns": len(context.turns),
            "duration_minutes": (context.last_updated - context.created_at).total_seconds() / 60,
            "intent_distribution": intent_counts,
            "entity_distribution": entity_counts,
            "average_confidence": avg_confidence,
            "current_topic": context.current_topic,
            "user_preferences": context.preferences
        }
    
    def set_matching_engine(self, matching_engine):
        """Set the matching engine for material searches"""
        self.matching_engine = matching_engine
    
    def set_logistics_engine(self, logistics_engine):
        """Set the logistics engine for route queries"""
        self.logistics_engine = logistics_engine

# Example usage and testing
if __name__ == "__main__":
    # Initialize the conversational agent
    agent = DeepSeekR1ConversationalAgent()
    
    # Test conversation
    test_messages = [
        "Hello, I'm looking for companies that can use our textile waste",
        "We have about 5 tons of cotton waste per month",
        "What are the environmental benefits of this match?",
        "How much would transportation cost to Berlin?",
        "Thank you for your help"
    ]
    
    user_id = "test_user_123"
    company_id = "test_company_456"
    
    for message in test_messages:
        print(f"\nUser: {message}")
        result = agent.process_message(message, user_id, company_id)
        print(f"Assistant: {result['response']}")
        print(f"Intent: {result['intent']['type']} (confidence: {result['intent']['confidence']:.2f})")
        if result['intent']['entities']:
            print(f"Entities: {result['intent']['entities']}")
    
    # Get conversation analytics
    analytics = agent.get_conversation_analytics(f"conv_{user_id}_{datetime.now().strftime('%Y%m%d')}")
    print(f"\nConversation Analytics: {json.dumps(analytics, indent=2)}") 