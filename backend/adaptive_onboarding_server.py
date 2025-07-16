"""
Flask Server for Adaptive AI Onboarding
Provides REST API endpoints for the adaptive onboarding system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import asyncio
import threading
from datetime import datetime
import sys
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the adaptive onboarding service
from adaptive_ai_onboarding import AdaptiveAIOnboarding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the adaptive onboarding service
adaptive_onboarding = AdaptiveAIOnboarding()

# Global session storage (in production, use Redis or database)
sessions = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'adaptive_ai_onboarding',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(adaptive_onboarding.active_sessions)
    })

@app.route('/api/adaptive-onboarding/start', methods=['POST'])
def start_onboarding():
    """Start a new adaptive onboarding session"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        initial_profile = data.get('initial_profile', {})
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Start onboarding session
        session = asyncio.run(adaptive_onboarding.start_onboarding_session(user_id, initial_profile))
        
        # Convert session to JSON-serializable format
        session_data = {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'initial_questions': [
                {
                    'id': q.id,
                    'question': q.question,
                    'type': q.expected_answer_type,
                    'category': q.category,
                    'importance': q.importance,
                    'options': q.options,
                    'reasoning': q.reasoning,
                    'compliance_related': q.compliance_related
                }
                for q in session.questions_asked
            ],
            'completion_percentage': session.completion_percentage,
            'created_at': session.created_at.isoformat()
        }
        
        logger.info(f"Started onboarding session {session.session_id} for user {user_id}")
        
        return jsonify({
            'success': True,
            'session': session_data
        })
        
    except Exception as e:
        logger.error(f"Error starting onboarding session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive-onboarding/respond', methods=['POST'])
def process_response():
    """Process user response and determine next actions"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        question_id = data.get('question_id')
        answer = data.get('answer')
        
        if not all([session_id, question_id, answer is not None]):
            return jsonify({'error': 'session_id, question_id, and answer are required'}), 400
        
        # Process user response
        result = asyncio.run(adaptive_onboarding.process_user_response(
            session_id, question_id, answer
        ))
        
        # Convert result to JSON-serializable format
        result_data = {
            'session_id': result['session_id'],
            'answer_quality': result['answer_quality'],
            'confidence': result['confidence'],
            'completion_percentage': result['completion_percentage'],
            'next_actions': result.get('next_actions', {}),
            'compliance_status': result.get('compliance_status', {}),
            'ai_insights': result.get('ai_insights', {})
        }
        
        logger.info(f"Processed response for session {session_id}, question {question_id}")
        
        return jsonify({
            'success': True,
            **result_data
        })
        
    except Exception as e:
        logger.error(f"Error processing user response: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive-onboarding/complete', methods=['POST'])
def complete_onboarding():
    """Complete the onboarding process and generate final analysis"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        # Complete onboarding
        result = asyncio.run(adaptive_onboarding.complete_onboarding(session_id))
        
        # Convert result to JSON-serializable format
        result_data = {
            'session_id': result['session_id'],
            'success': result['success'],
            'company_profile': result.get('company_profile', {}),
            'compliance_status': result.get('compliance_status', {}),
            'federated_contribution': result.get('federated_contribution', 0),
            'completion_timestamp': result.get('completion_timestamp', ''),
            'analysis': result.get('analysis', {})
        }
        
        logger.info(f"Completed onboarding session {session_id}")
        
        return jsonify(result_data)
        
    except Exception as e:
        logger.error(f"Error completing onboarding: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive-onboarding/status/<session_id>', methods=['GET'])
def get_session_status(session_id):
    """Get current session status"""
    try:
        # Get session status
        status = adaptive_onboarding.get_session_status(session_id)
        
        if not status:
            return jsonify({'error': 'Session not found'}), 404
        
        # Convert status to JSON-serializable format
        status_data = {
            'session_id': status['session_id'],
            'user_id': status['user_id'],
            'current_phase': status['current_phase'],
            'completion_percentage': status['completion_percentage'],
            'questions_asked': status['questions_asked'],
            'responses_received': status['responses_received'],
            'created_at': status['created_at'],
            'updated_at': status['updated_at']
        }
        
        return jsonify({
            'success': True,
            **status_data
        })
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive-onboarding/questions/<session_id>', methods=['GET'])
def get_next_questions(session_id):
    """Get next questions for a session (for polling)"""
    try:
        # Get session
        session = adaptive_onboarding.active_sessions.get(session_id)
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Check if there are new questions
        current_question_count = len(session.questions_asked)
        responses_count = len(session.user_responses)
        
        # If we have more questions than responses, return the next questions
        if current_question_count > responses_count:
            next_questions = session.questions_asked[responses_count:]
            return jsonify({
                'success': True,
                'next_questions': [
                    {
                        'id': q.id,
                        'question': q.question,
                        'type': q.expected_answer_type,
                        'category': q.category,
                        'importance': q.importance,
                        'options': q.options,
                        'reasoning': q.reasoning,
                        'compliance_related': q.compliance_related
                    }
                    for q in next_questions
                ]
            })
        else:
            return jsonify({
                'success': True,
                'next_questions': [],
                'completion_ready': session.completion_percentage >= 0.8
            })
        
    except Exception as e:
        logger.error(f"Error getting next questions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive-onboarding/federated/status', methods=['GET'])
def get_federated_status():
    """Get federated learning status"""
    try:
        if not adaptive_onboarding.federated_learner:
            return jsonify({
                'success': True,
                'federated_learning': 'not_available'
            })
        
        # Get federated learning summary
        summary = adaptive_onboarding.federated_learner.get_training_summary()
        
        return jsonify({
            'success': True,
            'federated_learning': 'active',
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting federated status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/adaptive-onboarding/compliance/check', methods=['POST'])
def check_compliance():
    """Check compliance for a specific scenario"""
    try:
        data = request.get_json()
        company_profile = data.get('company_profile', {})
        question = data.get('question', {})
        answer = data.get('answer')
        
        if not adaptive_onboarding.compliance_engine:
            return jsonify({
                'success': True,
                'compliance_status': 'compliance_engine_unavailable'
            })
        
        # Check compliance
        compliance_result = asyncio.run(adaptive_onboarding._check_compliance(
            company_profile, question, answer
        ))
        
        return jsonify({
            'success': True,
            'compliance_status': compliance_result
        })
        
    except Exception as e:
        logger.error(f"Error checking compliance: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Run Flask server on port 5003
    app.run(host='0.0.0.0', port=5003, debug=False) 