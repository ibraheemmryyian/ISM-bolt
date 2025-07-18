"""
Advanced Adaptive AI Onboarding Service
Implements conversational flow, federated learning, and compliance matching
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import requests
from enum import Enum
import torch
from ml_core.models import BaseRLAgent
from ml_core.training import train_rl
from ml_core.monitoring import log_metrics, save_checkpoint
import os

# Import existing services
try:
    import sys
    import os
    # Add parent directory to path to find modules in root
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from federated_meta_learning import FederatedMetaLearning
except ImportError:
    FederatedMetaLearning = None

try:
    from regulatory_compliance import RegulatoryComplianceEngine
except ImportError:
    RegulatoryComplianceEngine = None

try:
    from ai_onboarding_questions_generator import AIOnboardingQuestionsGenerator
except ImportError:
    AIOnboardingQuestionsGenerator = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionType(Enum):
    BASIC_INFO = "basic_info"
    INDUSTRY_SPECIFIC = "industry_specific"
    SUSTAINABILITY = "sustainability"
    COMPLIANCE = "compliance"
    SYMBIOSIS_READINESS = "symbiosis_readiness"
    FOLLOW_UP = "follow_up"

class AnswerQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    BASIC = "basic"
    POOR = "poor"

@dataclass
class OnboardingQuestion:
    id: str
    type: QuestionType
    question: str
    category: str
    importance: str  # high, medium, low
    expected_answer_type: str  # text, numeric, boolean, multiselect
    options: Optional[List[str]] = None
    reasoning: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None
    compliance_related: bool = False
    federated_learning_value: float = 0.5

@dataclass
class UserResponse:
    question_id: str
    answer: Any
    answer_quality: AnswerQuality
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class OnboardingSession:
    session_id: str
    user_id: str
    company_profile: Dict[str, Any]
    questions_asked: List[OnboardingQuestion]
    user_responses: List[UserResponse]
    current_phase: str
    completion_percentage: float
    ai_insights: Dict[str, Any]
    federated_data: Dict[str, Any]
    compliance_status: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class AdaptiveAIOnboarding:
    """
    Advanced Adaptive AI Onboarding System
    Features:
    - Conversational flow with dynamic question generation
    - Federated learning integration
    - Real-time compliance checking
    - Quality assessment of responses
    - Intelligent follow-up question generation
    """
    
    def __init__(self):
        self.federated_learner = FederatedMetaLearning() if FederatedMetaLearning else None
        self.compliance_engine = RegulatoryComplianceEngine() if RegulatoryComplianceEngine else None
        self.questions_generator = AIOnboardingQuestionsGenerator() if AIOnboardingQuestionsGenerator else None
        
        # Question templates by category
        self.question_templates = self._initialize_question_templates()
        
        # Industry-specific question mappings
        self.industry_questions = self._initialize_industry_questions()
        
        # Compliance requirements by industry
        self.compliance_requirements = self._initialize_compliance_requirements()
        
        # Federated learning configuration
        self.federated_config = {
            'min_responses_for_learning': 5,
            'learning_threshold': 0.7,
            'data_retention_days': 30,
            'privacy_budget': 1.0
        }
        
        # Session management
        self.active_sessions: Dict[str, OnboardingSession] = {}
        
        logger.info("🚀 Adaptive AI Onboarding Service initialized")

    def _initialize_question_templates(self) -> Dict[str, List[Dict]]:
        """Initialize question templates for different categories"""
        return {
            "basic_info": [
                {
                    "question": "What is your company name?",
                    "type": "text",
                    "importance": "high",
                    "reasoning": "Essential for identification and compliance"
                },
                {
                    "question": "What industry are you in?",
                    "type": "text",
                    "importance": "high",
                    "reasoning": "Determines relevant symbiosis opportunities and compliance requirements"
                },
                {
                    "question": "Where is your company located?",
                    "type": "text",
                    "importance": "high",
                    "reasoning": "Critical for logistics optimization and local compliance"
                },
                {
                    "question": "How many employees do you have?",
                    "type": "select",
                    "options": ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"],
                    "importance": "medium",
                    "reasoning": "Helps assess company scale and resource needs"
                }
            ],
            "production_info": [
                {
                    "question": "What products or services do you produce?",
                    "type": "textarea",
                    "importance": "high",
                    "reasoning": "Identifies potential waste streams and resource needs"
                },
                {
                    "question": "What are your main raw materials?",
                    "type": "textarea",
                    "importance": "high",
                    "reasoning": "Essential for identifying symbiosis opportunities"
                },
                {
                    "question": "What is your production volume?",
                    "type": "text",
                    "importance": "medium",
                    "reasoning": "Helps quantify potential symbiosis impact"
                }
            ],
            "sustainability": [
                {
                    "question": "What are your current sustainability goals?",
                    "type": "multiselect",
                    "options": ["Carbon reduction", "Waste minimization", "Energy efficiency", "Circular economy", "Water conservation", "None yet"],
                    "importance": "medium",
                    "reasoning": "Assesses current sustainability maturity"
                },
                {
                    "question": "Do you currently track your waste streams?",
                    "type": "boolean",
                    "importance": "medium",
                    "reasoning": "Indicates readiness for symbiosis"
                }
            ],
            "compliance": [
                {
                    "question": "Are you certified to any environmental standards?",
                    "type": "multiselect",
                    "options": ["ISO 14001", "ISO 50001", "EMAS", "LEED", "BREEAM", "None"],
                    "importance": "medium",
                    "reasoning": "Assesses compliance maturity and potential requirements"
                }
            ]
        }

    def _initialize_industry_questions(self) -> Dict[str, List[Dict]]:
        """Initialize industry-specific questions"""
        return {
            "Steel Manufacturing": [
                {
                    "question": "What types of steel do you produce?",
                    "type": "multiselect",
                    "options": ["Carbon steel", "Stainless steel", "Alloy steel", "Tool steel"],
                    "importance": "high",
                    "reasoning": "Different steel types have different waste streams"
                },
                {
                    "question": "Do you have slag processing capabilities?",
                    "type": "boolean",
                    "importance": "high",
                    "reasoning": "Slag is a major byproduct with high symbiosis potential"
                }
            ],
            "Chemical Manufacturing": [
                {
                    "question": "What chemical processes do you use?",
                    "type": "multiselect",
                    "options": ["Distillation", "Crystallization", "Filtration", "Reaction", "Separation"],
                    "importance": "high",
                    "reasoning": "Different processes have different byproducts"
                },
                {
                    "question": "Do you handle hazardous materials?",
                    "type": "boolean",
                    "importance": "high",
                    "reasoning": "Critical for compliance and safety requirements"
                }
            ],
            "Food Processing": [
                {
                    "question": "What types of food do you process?",
                    "type": "multiselect",
                    "options": ["Meat", "Dairy", "Grains", "Fruits/Vegetables", "Beverages"],
                    "importance": "high",
                    "reasoning": "Different food types have different waste streams"
                },
                {
                    "question": "Do you have organic waste streams?",
                    "type": "boolean",
                    "importance": "high",
                    "reasoning": "Organic waste has high symbiosis potential"
                }
            ]
        }

    def _initialize_compliance_requirements(self) -> Dict[str, List[str]]:
        """Initialize compliance requirements by industry"""
        return {
            "Steel Manufacturing": [
                "Environmental permits for emissions",
                "Waste management permits",
                "Hazardous waste handling certification",
                "ISO 14001 recommended"
            ],
            "Chemical Manufacturing": [
                "Chemical safety permits",
                "Hazardous material handling",
                "Environmental impact assessment",
                "ISO 14001 required"
            ],
            "Food Processing": [
                "Food safety certification",
                "Organic waste handling permits",
                "Water discharge permits",
                "HACCP compliance"
            ]
        }

    async def start_onboarding_session(self, user_id: str, initial_profile: Dict[str, Any]) -> OnboardingSession:
        """Start a new adaptive onboarding session"""
        try:
            session_id = f"onboarding_{user_id}_{int(datetime.now().timestamp())}"
            
            # Create initial session
            session = OnboardingSession(
                session_id=session_id,
                user_id=user_id,
                company_profile=initial_profile,
                questions_asked=[],
                user_responses=[],
                current_phase="initial",
                completion_percentage=0.0,
                ai_insights={},
                federated_data={},
                compliance_status={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Store session
            self.active_sessions[session_id] = session
            
            # Generate initial questions
            initial_questions = await self._generate_initial_questions(initial_profile)
            session.questions_asked = initial_questions
            
            logger.info(f"🎯 Started onboarding session {session_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"❌ Error starting onboarding session: {e}")
            raise

    async def _generate_initial_questions(self, profile: Dict[str, Any]) -> List[OnboardingQuestion]:
        """Generate initial questions based on company profile"""
        questions = []
        
        # Always start with basic info
        basic_templates = self.question_templates["basic_info"]
        for i, template in enumerate(basic_templates):
            question = OnboardingQuestion(
                id=f"basic_{i}",
                type=QuestionType.BASIC_INFO,
                question=template["question"],
                category="basic_info",
                importance=template["importance"],
                expected_answer_type=template["type"],
                options=template.get("options"),
                reasoning=template["reasoning"],
                compliance_related=False,
                federated_learning_value=0.8 if template["importance"] == "high" else 0.5
            )
            questions.append(question)
        
        # Add industry-specific questions if industry is known
        if profile.get("industry"):
            industry_questions = self.industry_questions.get(profile["industry"], [])
            for i, template in enumerate(industry_questions):
                question = OnboardingQuestion(
                    id=f"industry_{i}",
                    type=QuestionType.INDUSTRY_SPECIFIC,
                    question=template["question"],
                    category="industry_specific",
                    importance=template["importance"],
                    expected_answer_type=template["type"],
                    options=template.get("options"),
                    reasoning=template["reasoning"],
                    compliance_related=True,
                    federated_learning_value=0.9
                )
                questions.append(question)
        
        return questions

    async def process_user_response(self, session_id: str, question_id: str, 
                                  answer: Any) -> Dict[str, Any]:
        """Process user response and determine next actions"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Find the question
            question = next((q for q in session.questions_asked if q.id == question_id), None)
            if not question:
                raise ValueError(f"Question {question_id} not found in session")
            
            # Assess answer quality
            answer_quality = self._assess_answer_quality(question, answer)
            
            # Create user response
            user_response = UserResponse(
                question_id=question_id,
                answer=answer,
                answer_quality=answer_quality,
                confidence=self._calculate_confidence(answer, answer_quality),
                timestamp=datetime.now(),
                metadata={
                    "question_type": question.type.value,
                    "category": question.category,
                    "importance": question.importance
                }
            )
            
            # Add response to session
            session.user_responses.append(user_response)
            
            # Update company profile
            self._update_company_profile(session, question, answer)
            
            # Check compliance if relevant
            if question.compliance_related:
                compliance_result = await self._check_compliance(session.company_profile, question, answer)
                session.compliance_status.update(compliance_result)
            
            # Collect federated learning data
            if self.federated_learner and question.federated_learning_value > 0.5:
                await self._collect_federated_data(session, question, user_response)
            
            # Generate next questions or complete
            next_actions = await self._determine_next_actions(session, user_response)
            
            # Update session
            session.updated_at = datetime.now()
            session.completion_percentage = self._calculate_completion_percentage(session)
            
            return {
                "session_id": session_id,
                "answer_quality": answer_quality.value,
                "confidence": user_response.confidence,
                "compliance_status": session.compliance_status,
                "next_actions": next_actions,
                "completion_percentage": session.completion_percentage,
                "ai_insights": session.ai_insights
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing user response: {e}")
            raise

    def _assess_answer_quality(self, question: OnboardingQuestion, answer: Any) -> AnswerQuality:
        """Assess the quality of a user's answer"""
        try:
            score = 0.0
            
            # Check if answer is provided
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                return AnswerQuality.POOR
            
            # Length-based scoring for text answers
            if question.expected_answer_type == "text" and isinstance(answer, str):
                if len(answer) > 100:
                    score += 0.4
                elif len(answer) > 50:
                    score += 0.3
                elif len(answer) > 20:
                    score += 0.2
                else:
                    score += 0.1
            
            # Content quality scoring
            if question.expected_answer_type == "multiselect" and isinstance(answer, list):
                if len(answer) > 1:
                    score += 0.3  # Multiple selections show engagement
            
            # Industry-specific scoring
            if question.type == QuestionType.INDUSTRY_SPECIFIC:
                score += 0.2  # Bonus for industry-specific questions
            
            # Normalize score
            score = min(score, 1.0)
            
            # Determine quality level
            if score >= 0.8:
                return AnswerQuality.EXCELLENT
            elif score >= 0.6:
                return AnswerQuality.GOOD
            elif score >= 0.4:
                return AnswerQuality.BASIC
            else:
                return AnswerQuality.POOR
                
        except Exception as e:
            logger.error(f"Error assessing answer quality: {e}")
            return AnswerQuality.BASIC

    def _calculate_confidence(self, answer: Any, quality: AnswerQuality) -> float:
        """Calculate confidence in the answer"""
        base_confidence = {
            AnswerQuality.EXCELLENT: 0.9,
            AnswerQuality.GOOD: 0.7,
            AnswerQuality.BASIC: 0.5,
            AnswerQuality.POOR: 0.3
        }
        
        confidence = base_confidence[quality]
        
        # Adjust based on answer type
        if isinstance(answer, list) and len(answer) > 1:
            confidence += 0.1
        elif isinstance(answer, str) and len(answer) > 50:
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _update_company_profile(self, session: OnboardingSession, question: OnboardingQuestion, answer: Any):
        """Update company profile based on user response"""
        try:
            if question.category == "basic_info":
                if "name" in question.question.lower():
                    session.company_profile["name"] = answer
                elif "industry" in question.question.lower():
                    session.company_profile["industry"] = answer
                elif "location" in question.question.lower():
                    session.company_profile["location"] = answer
                elif "employee" in question.question.lower():
                    session.company_profile["employee_count"] = answer
            
            elif question.category == "production_info":
                if "product" in question.question.lower():
                    session.company_profile["products"] = answer
                elif "material" in question.question.lower():
                    session.company_profile["main_materials"] = answer
                elif "volume" in question.question.lower():
                    session.company_profile["production_volume"] = answer
            
            elif question.category == "sustainability":
                if "goal" in question.question.lower():
                    session.company_profile["sustainability_goals"] = answer if isinstance(answer, list) else [answer]
                elif "waste" in question.question.lower():
                    session.company_profile["tracks_waste"] = answer
            
            elif question.category == "industry_specific":
                # Store industry-specific data
                if "industry_data" not in session.company_profile:
                    session.company_profile["industry_data"] = {}
                session.company_profile["industry_data"][question.id] = answer
            
        except Exception as e:
            logger.error(f"Error updating company profile: {e}")

    async def _check_compliance(self, company_profile: Dict[str, Any], question: OnboardingQuestion, answer: Any) -> Dict[str, Any]:
        """Check compliance requirements based on user response"""
        try:
            if not self.compliance_engine:
                return {"status": "compliance_engine_unavailable"}
            
            industry = company_profile.get("industry", "")
            compliance_requirements = self.compliance_requirements.get(industry, [])
            
            compliance_result = {
                "industry": industry,
                "requirements": compliance_requirements,
                "current_status": "pending",
                "recommendations": []
            }
            
            # Check if user has relevant certifications
            if question.expected_answer_type == "multiselect" and isinstance(answer, list):
                if "ISO 14001" in answer:
                    compliance_result["current_status"] = "compliant"
                    compliance_result["recommendations"].append("Excellent! ISO 14001 certification shows strong environmental management.")
                elif "None" in answer:
                    compliance_result["recommendations"].append("Consider obtaining ISO 14001 certification for better environmental management.")
            
            # Check hazardous material handling
            if "hazardous" in question.question.lower() and answer is True:
                compliance_result["recommendations"].append("Ensure proper hazardous material handling permits are in place.")
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            return {"status": "error", "message": str(e)}

    async def _collect_federated_data(self, session: OnboardingSession, question: OnboardingQuestion, response: UserResponse):
        """Collect data for federated learning"""
        try:
            if not self.federated_learner:
                return
            
            # Prepare federated learning data
            federated_data = {
                "question_id": question.id,
                "question_type": question.type.value,
                "category": question.category,
                "answer_quality": response.answer_quality.value,
                "response_time": response.timestamp - session.user_responses[-1].timestamp if session.user_responses else 0, # Calculate response time
                "confidence": response.confidence,
                "industry": session.company_profile.get("industry", ""),
                "company_size": session.company_profile.get("employee_count", ""),
                "timestamp": response.timestamp.isoformat(),
                "learning_value": question.federated_learning_value
            }
            
            # Store in session for later aggregation
            if "federated_data" not in session.federated_data:
                session.federated_data["federated_data"] = []
            session.federated_data["federated_data"].append(federated_data)
            
            # Register with federated learner if enough data
            if len(session.federated_data["federated_data"]) >= self.federated_config["min_responses_for_learning"]:
                await self._contribute_to_federated_learning(session)
                
        except Exception as e:
            logger.error(f"Error collecting federated data: {e}")

    async def _contribute_to_federated_learning(self, session: OnboardingSession):
        """Contribute session data to federated learning"""
        try:
            if not self.federated_learner:
                return
            
            # Prepare model parameters (simplified)
            model_params = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "data_points": len(session.federated_data["federated_data"]),
                "average_quality": np.mean([
                    {"excellent": 1.0, "good": 0.7, "basic": 0.5, "poor": 0.3}[r["answer_quality"]]
                    for r in session.federated_data["federated_data"]
                ]),
                "industry_distribution": self._calculate_industry_distribution(session),
                "question_effectiveness": self._calculate_question_effectiveness(session)
            }
            
            # Register with federated learner
            self.federated_learner.register_client(session.session_id, model_params)
            
            logger.info(f"📊 Contributed federated learning data for session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error contributing to federated learning: {e}")

    def _calculate_industry_distribution(self, session: OnboardingSession) -> Dict[str, float]:
        """Calculate industry distribution for federated learning"""
        industry = session.company_profile.get("industry", "unknown")
        return {industry: 1.0}

    def _calculate_question_effectiveness(self, session: OnboardingSession) -> Dict[str, float]:
        """Calculate question effectiveness for federated learning"""
        effectiveness = {}
        for response in session.user_responses:
            question_id = response.question_id
            if question_id not in effectiveness:
                effectiveness[question_id] = []
            effectiveness[question_id].append(response.confidence)
        
        # Calculate average effectiveness per question
        return {qid: np.mean(scores) for qid, scores in effectiveness.items()}

    async def _determine_next_actions(self, session: OnboardingSession, last_response: UserResponse) -> Dict[str, Any]:
        """Determine next actions based on user response"""
        try:
            next_actions = {
                "should_continue": True,
                "next_questions": [],
                "completion_ready": False,
                "recommendations": []
            }
            
            # Check if we have enough information
            if session.completion_percentage >= 0.8:
                next_actions["completion_ready"] = True
                next_actions["should_continue"] = False
                next_actions["recommendations"].append("You have provided excellent information! Ready to complete onboarding.")
                return next_actions
            
            # Generate follow-up questions based on last response
            follow_up_questions = await self._generate_follow_up_questions(session, last_response)
            
            if follow_up_questions:
                next_actions["next_questions"] = follow_up_questions
            else:
                # No more questions needed
                next_actions["completion_ready"] = True
                next_actions["should_continue"] = False
            
            return next_actions
            
        except Exception as e:
            logger.error(f"Error determining next actions: {e}")
            return {"should_continue": False, "error": str(e)}

    async def _generate_follow_up_questions(self, session: OnboardingSession, last_response: UserResponse) -> List[OnboardingQuestion]:
        """Generate follow-up questions based on user response"""
        try:
            follow_up_questions = []
            
            # Get the last question
            last_question = next((q for q in session.questions_asked if q.id == last_response.question_id), None)
            if not last_question:
                return []
            
            # Generate questions based on answer quality and content
            if last_response.answer_quality in [AnswerQuality.EXCELLENT, AnswerQuality.GOOD]:
                # User is engaged, ask deeper questions
                if last_question.category == "basic_info":
                    # Move to production info
                    production_templates = self.question_templates["production_info"]
                    for i, template in enumerate(production_templates):
                        question = OnboardingQuestion(
                            id=f"production_{len(session.questions_asked) + i}",
                            type=QuestionType.FOLLOW_UP,
                            question=template["question"],
                            category="production_info",
                            importance=template["importance"],
                            expected_answer_type=template["type"],
                            options=template.get("options"),
                            reasoning=template["reasoning"],
                            compliance_related=False,
                            federated_learning_value=0.7
                        )
                        follow_up_questions.append(question)
                
                elif last_question.category == "production_info":
                    # Move to sustainability
                    sustainability_templates = self.question_templates["sustainability"]
                    for i, template in enumerate(sustainability_templates):
                        question = OnboardingQuestion(
                            id=f"sustainability_{len(session.questions_asked) + i}",
                            type=QuestionType.SUSTAINABILITY,
                            question=template["question"],
                            category="sustainability",
                            importance=template["importance"],
                            expected_answer_type=template["type"],
                            options=template.get("options"),
                            reasoning=template["reasoning"],
                            compliance_related=True,
                            federated_learning_value=0.8
                        )
                        follow_up_questions.append(question)
            
            elif last_response.answer_quality == AnswerQuality.POOR:
                # User needs simpler questions or clarification
                clarification_question = OnboardingQuestion(
                    id=f"clarification_{len(session.questions_asked)}",
                    type=QuestionType.FOLLOW_UP,
                    question=f"Could you provide more details about {last_question.question.lower()}?",
                    category="clarification",
                    importance="medium",
                    expected_answer_type="text",
                    reasoning="Need more information to provide accurate recommendations",
                    compliance_related=False,
                    federated_learning_value=0.3
                )
                follow_up_questions.append(clarification_question)
            
            # Add questions to session
            session.questions_asked.extend(follow_up_questions)
            
            return follow_up_questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []

    def _calculate_completion_percentage(self, session: OnboardingSession) -> float:
        """Calculate completion percentage based on questions answered"""
        try:
            if not session.questions_asked:
                return 0.0
            
            total_questions = len(session.questions_asked)
            answered_questions = len(session.user_responses)
            
            if answered_questions == 0:
                return 0.0
            
            # Simple percentage calculation
            completion_percentage = (answered_questions / total_questions) * 100
            
            return min(completion_percentage, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating completion percentage: {e}")
            return 0.0

    async def complete_onboarding(self, session_id: str) -> Dict[str, Any]:
        """Complete the onboarding process and generate final analysis"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Generate comprehensive analysis
            analysis = await self._generate_comprehensive_analysis(session)
            
            # Final federated learning contribution
            if self.federated_learner and session.federated_data.get("federated_data"):
                await self._contribute_to_federated_learning(session)
            
            # Clean up session
            del self.active_sessions[session_id]
            
            return {
                "session_id": session_id,
                "success": True,
                "analysis": analysis,
                "company_profile": session.company_profile,
                "compliance_status": session.compliance_status,
                "federated_contribution": len(session.federated_data.get("federated_data", [])),
                "completion_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error completing onboarding: {e}")
            raise

    async def _generate_comprehensive_analysis(self, session: OnboardingSession) -> Dict[str, Any]:
        """Generate comprehensive analysis of the onboarding data"""
        try:
            analysis = {
                "symbiosis_score": self._calculate_symbiosis_score(session),
                "compliance_score": self._calculate_compliance_score(session),
                "sustainability_maturity": self._calculate_sustainability_maturity(session),
                "recommended_opportunities": self._generate_opportunity_recommendations(session),
                "risk_assessment": self._assess_risks(session),
                "implementation_roadmap": self._generate_implementation_roadmap(session),
                "ai_insights": self._generate_ai_insights(session)
            }
            
            # Store analysis in session
            session.ai_insights = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            return {"error": str(e)}

    def _calculate_symbiosis_score(self, session: OnboardingSession) -> float:
        """Calculate symbiosis potential score"""
        try:
            score = 0.5  # Base score
            
            # Industry factor
            industry = session.company_profile.get("industry", "").lower()
            if any(keyword in industry for keyword in ["steel", "chemical", "food", "manufacturing"]):
                score += 0.2
            
            # Waste tracking
            if session.company_profile.get("tracks_waste"):
                score += 0.15
            
            # Sustainability goals
            goals = session.company_profile.get("sustainability_goals", [])
            if goals and "None" not in goals:
                score += 0.15
            
            # Response quality
            avg_quality = np.mean([
                {"excellent": 1.0, "good": 0.7, "basic": 0.5, "poor": 0.3}[r.answer_quality.value]
                for r in session.user_responses
            ])
            score += avg_quality * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating symbiosis score: {e}")
            return 0.5

    def _calculate_compliance_score(self, session: OnboardingSession) -> float:
        """Calculate compliance readiness score"""
        try:
            score = 0.5  # Base score
            
            compliance_status = session.compliance_status
            if compliance_status.get("current_status") == "compliant":
                score += 0.3
            
            # Industry-specific compliance
            industry = session.company_profile.get("industry", "")
            if industry in self.compliance_requirements:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating compliance score: {e}")
            return 0.5

    def _calculate_sustainability_maturity(self, session: OnboardingSession) -> str:
        """Calculate sustainability maturity level"""
        try:
            goals = session.company_profile.get("sustainability_goals", [])
            tracks_waste = session.company_profile.get("tracks_waste", False)
            
            if tracks_waste and goals and "None" not in goals:
                return "Advanced"
            elif goals and "None" not in goals:
                return "Intermediate"
            elif tracks_waste:
                return "Basic"
            else:
                return "Beginner"
                
        except Exception as e:
            logger.error(f"Error calculating sustainability maturity: {e}")
            return "Beginner"

    def _generate_opportunity_recommendations(self, session: OnboardingSession) -> List[Dict[str, Any]]:
        """Generate symbiosis opportunity recommendations"""
        try:
            opportunities = []
            industry = session.company_profile.get("industry", "").lower()
            
            if "steel" in industry:
                opportunities.append({
                    "type": "waste_exchange",
                    "title": "Steel Slag Utilization",
                    "description": "Connect with cement manufacturers to utilize steel slag",
                    "potential_savings": "$50K-200K annually",
                    "carbon_reduction": "100-500 tons CO2",
                    "implementation_time": "6-12 months"
                })
            
            if "chemical" in industry:
                opportunities.append({
                    "type": "byproduct_recovery",
                    "title": "Chemical Byproduct Recovery",
                    "description": "Recover and sell chemical byproducts to other industries",
                    "potential_savings": "$100K-500K annually",
                    "carbon_reduction": "200-1000 tons CO2",
                    "implementation_time": "3-9 months"
                })
            
            if "food" in industry:
                opportunities.append({
                    "type": "organic_waste",
                    "title": "Organic Waste Processing",
                    "description": "Convert organic waste to biogas or compost",
                    "potential_savings": "$25K-100K annually",
                    "carbon_reduction": "50-200 tons CO2",
                    "implementation_time": "4-8 months"
                })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error generating opportunity recommendations: {e}")
            return []

    def _assess_risks(self, session: OnboardingSession) -> Dict[str, Any]:
        """Assess risks for symbiosis implementation"""
        try:
            risks = {
                "overall_risk": "low",
                "risk_factors": [],
                "mitigation_strategies": []
            }
            
            # Check compliance risks
            if session.compliance_status.get("current_status") != "compliant":
                risks["risk_factors"].append("Compliance requirements not met")
                risks["mitigation_strategies"].append("Obtain necessary permits and certifications")
            
            # Check data quality risks
            poor_responses = [r for r in session.user_responses if r.answer_quality == AnswerQuality.POOR]
            if len(poor_responses) > len(session.user_responses) * 0.3:
                risks["risk_factors"].append("Low data quality")
                risks["mitigation_strategies"].append("Conduct detailed waste audit")
                risks["overall_risk"] = "medium"
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            return {"overall_risk": "unknown", "error": str(e)}

    def _generate_implementation_roadmap(self, session: OnboardingSession) -> List[Dict[str, Any]]:
        """Generate implementation roadmap"""
        try:
            roadmap = [
                {
                    "phase": "Phase 1: Assessment",
                    "duration": "1-2 months",
                    "activities": [
                        "Conduct detailed waste audit",
                        "Identify potential partners",
                        "Assess technical feasibility"
                    ],
                    "milestones": ["Waste audit completed", "Partner shortlist created"]
                },
                {
                    "phase": "Phase 2: Planning",
                    "duration": "2-3 months",
                    "activities": [
                        "Develop partnership agreements",
                        "Design logistics solutions",
                        "Obtain necessary permits"
                    ],
                    "milestones": ["Partnership agreements signed", "Permits obtained"]
                },
                {
                    "phase": "Phase 3: Implementation",
                    "duration": "3-6 months",
                    "activities": [
                        "Set up logistics infrastructure",
                        "Begin material exchanges",
                        "Monitor and optimize"
                    ],
                    "milestones": ["First material exchange", "System operational"]
                }
            ]
            
            return roadmap
            
        except Exception as e:
            logger.error(f"Error generating implementation roadmap: {e}")
            return []

    def _generate_ai_insights(self, session: OnboardingSession) -> Dict[str, Any]:
        """Generate AI insights based on onboarding data"""
        try:
            insights = {
                "key_findings": [],
                "recommendations": [],
                "market_opportunities": [],
                "competitive_advantages": []
            }
            
            # Generate insights based on responses
            for response in session.user_responses:
                if response.answer_quality == AnswerQuality.EXCELLENT:
                    insights["key_findings"].append(f"Strong engagement in {response.metadata.get('category', 'unknown')} area")
            
            # Industry-specific insights
            industry = session.company_profile.get("industry", "").lower()
            if "steel" in industry:
                insights["market_opportunities"].append("High demand for steel slag in construction industry")
            elif "chemical" in industry:
                insights["market_opportunities"].append("Growing market for chemical byproduct recovery")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return {"error": str(e)}

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            return {
                "session_id": session_id,
                "user_id": session.user_id,
                "current_phase": session.current_phase,
                "completion_percentage": session.completion_percentage,
                "questions_asked": len(session.questions_asked),
                "responses_received": len(session.user_responses),
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return None

# Global instance
adaptive_onboarding = AdaptiveAIOnboarding() 

class AdaptiveOnboardingAgent:
    def __init__(self, state_dim=20, action_dim=10, model_dir="onboarding_models"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dir = model_dir
        self.agent = BaseRLAgent(state_dim, action_dim)
    def train(self, env, episodes=100):
        train_rl(self.agent, env, episodes=episodes)
        save_checkpoint(self.agent, None, episodes, os.path.join(self.model_dir, "onboarding_agent.pt"))
    def select_question(self, state):
        self.agent.eval()
        with torch.no_grad():
            q_values = self.agent(torch.tensor(state, dtype=torch.float))
            return torch.argmax(q_values).item() 