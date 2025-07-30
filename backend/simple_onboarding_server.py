"""
Simple Onboarding Server
A simplified version of the adaptive onboarding server with minimal dependencies
"""

import json
import os
import sys
import time
from datetime import datetime
import http.server
import socketserver
import urllib.parse
from uuid import uuid4

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample initial questions
INITIAL_QUESTIONS = [
    {
        "id": "q1",
        "question": "What type of materials are you primarily interested in?",
        "type": "multiple_choice",
        "category": "material_type",
        "importance": "high",
        "options": ["Metals", "Plastics", "Textiles", "Electronics", "Chemicals", "Other"],
        "reasoning": "Understanding material type helps with initial categorization",
        "compliance_related": False
    },
    {
        "id": "q2",
        "question": "What is your primary industry or sector?",
        "type": "multiple_choice",
        "category": "industry",
        "importance": "high",
        "options": ["Manufacturing", "Construction", "Automotive", "Aerospace", "Consumer Goods", "Healthcare", "Other"],
        "reasoning": "Industry context helps tailor material recommendations",
        "compliance_related": False
    },
    {
        "id": "q3", 
        "question": "What volume of materials do you typically handle monthly?",
        "type": "multiple_choice",
        "category": "volume",
        "importance": "medium",
        "options": ["Less than 100kg", "100kg-1000kg", "1-10 tons", "10-100 tons", "More than 100 tons"],
        "reasoning": "Volume information helps with logistics planning",
        "compliance_related": False
    },
    {
        "id": "q4",
        "question": "What sustainability certifications are important to you?",
        "type": "multiple_select",
        "category": "compliance",
        "importance": "medium",
        "options": ["ISO 14001", "Cradle to Cradle", "FSC", "GreenGuard", "Energy Star", "None"],
        "reasoning": "Compliance requirements affect material selection",
        "compliance_related": True
    }
]

# Sample material listings
SAMPLE_MATERIALS = [
    {
        "id": "m1",
        "name": "Recycled PET Plastic Granules",
        "type": "Plastics",
        "description": "High-quality recycled PET plastic granules suitable for manufacturing.",
        "quantity_available": "5000 kg",
        "price_range": "$0.80-$1.20 per kg",
        "sustainability_rating": 4.5,
        "certifications": ["ISO 14001", "GreenGuard"],
        "location": "Chicago, IL"
    },
    {
        "id": "m2",
        "name": "Reclaimed Steel Sheets",
        "type": "Metals",
        "description": "Industrial-grade reclaimed steel sheets, perfect for construction and manufacturing.",
        "quantity_available": "20 tons",
        "price_range": "$450-$550 per ton",
        "sustainability_rating": 4.2,
        "certifications": ["ISO 14001"],
        "location": "Pittsburgh, PA"
    },
    {
        "id": "m3",
        "name": "Organic Cotton Fabric",
        "type": "Textiles",
        "description": "Premium organic cotton fabric, ethically sourced and environmentally friendly.",
        "quantity_available": "10000 yards",
        "price_range": "$3.50-$4.50 per yard",
        "sustainability_rating": 5.0,
        "certifications": ["GOTS", "Fair Trade"],
        "location": "Portland, OR"
    },
    {
        "id": "m4",
        "name": "Reclaimed Circuit Boards",
        "type": "Electronics",
        "description": "Cleaned and sorted circuit boards from decommissioned servers, suitable for precious metal recovery.",
        "quantity_available": "500 kg",
        "price_range": "$15-$20 per kg",
        "sustainability_rating": 4.8,
        "certifications": ["R2"],
        "location": "Austin, TX"
    },
    {
        "id": "m5",
        "name": "Bio-based Solvents",
        "type": "Chemicals",
        "description": "Environmentally friendly bio-based solvents derived from agricultural waste.",
        "quantity_available": "2000 liters",
        "price_range": "$8-$12 per liter",
        "sustainability_rating": 4.7,
        "certifications": ["USDA BioPreferred", "EPA Safer Choice"],
        "location": "Minneapolis, MN"
    }
]

# Store active sessions
active_sessions = {}

class SimpleOnboardingHandler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200, content_type="application/json"):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        self._set_headers()
    
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Health check endpoint
        if parsed_path.path == '/health':
            self._set_headers()
            response = {
                'status': 'healthy',
                'service': 'simple_onboarding',
                'timestamp': datetime.now().isoformat(),
                'active_sessions': len(active_sessions)
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Get session status
        if parsed_path.path.startswith('/api/adaptive-onboarding/status/'):
            session_id = parsed_path.path.split('/')[-1]
            if session_id in active_sessions:
                self._set_headers()
                session = active_sessions[session_id]
                response = {
                    'success': True,
                    'session_id': session_id,
                    'user_id': session.get('user_id', 'unknown'),
                    'current_phase': 'initial_questions',
                    'completion_percentage': 0.5,
                    'questions_asked': len(INITIAL_QUESTIONS),
                    'responses_received': len(session.get('responses', [])),
                    'created_at': session.get('created_at', ''),
                    'updated_at': datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Session not found'}).encode())
            return
        
        # Get next questions
        if parsed_path.path.startswith('/api/adaptive-onboarding/questions/'):
            session_id = parsed_path.path.split('/')[-1]
            if session_id in active_sessions:
                self._set_headers()
                session = active_sessions[session_id]
                responses_count = len(session.get('responses', []))
                
                # If we have more questions than responses, return the next questions
                if len(INITIAL_QUESTIONS) > responses_count:
                    next_questions = INITIAL_QUESTIONS[responses_count:]
                    response = {
                        'success': True,
                        'next_questions': next_questions
                    }
                else:
                    response = {
                        'success': True,
                        'next_questions': [],
                        'completion_ready': True
                    }
                self.wfile.write(json.dumps(response).encode())
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Session not found'}).encode())
            return
        
        # Default response for unknown endpoints
        self._set_headers(404)
        self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode())
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode())
            return
        
        # Start onboarding session
        if self.path == '/api/adaptive-onboarding/start':
            user_id = data.get('user_id')
            if not user_id:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'user_id is required'}).encode())
                return
            
            # Create new session
            session_id = str(uuid4())
            active_sessions[session_id] = {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'responses': []
            }
            
            # Return session data
            self._set_headers()
            response = {
                'success': True,
                'session': {
                    'session_id': session_id,
                    'user_id': user_id,
                    'initial_questions': INITIAL_QUESTIONS,
                    'completion_percentage': 0,
                    'created_at': active_sessions[session_id]['created_at']
                }
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Process user response
        if self.path == '/api/adaptive-onboarding/respond':
            session_id = data.get('session_id')
            question_id = data.get('question_id')
            answer = data.get('answer')
            
            if not all([session_id, question_id, answer is not None]):
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'session_id, question_id, and answer are required'}).encode())
                return
            
            if session_id not in active_sessions:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Session not found'}).encode())
                return
            
            # Store response
            session = active_sessions[session_id]
            if 'responses' not in session:
                session['responses'] = []
            
            session['responses'].append({
                'question_id': question_id,
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            # Calculate completion percentage
            completion_percentage = min(1.0, len(session['responses']) / len(INITIAL_QUESTIONS))
            
            # Return result
            self._set_headers()
            response = {
                'success': True,
                'session_id': session_id,
                'answer_quality': 'good',
                'confidence': 0.9,
                'completion_percentage': completion_percentage,
                'next_actions': {
                    'ask_more_questions': completion_percentage < 1.0
                }
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Complete onboarding
        if self.path == '/api/adaptive-onboarding/complete':
            session_id = data.get('session_id')
            
            if not session_id:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'session_id is required'}).encode())
                return
            
            if session_id not in active_sessions:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Session not found'}).encode())
                return
            
            # Generate material listings based on responses
            session = active_sessions[session_id]
            responses = session.get('responses', [])
            
            # Filter materials based on responses
            filtered_materials = SAMPLE_MATERIALS
            
            # Look for material type preference
            for response in responses:
                if response['question_id'] == 'q1':  # Material type question
                    material_type = response['answer']
                    if isinstance(material_type, str):
                        filtered_materials = [m for m in filtered_materials if m['type'] == material_type]
            
            # Return result with materials
            self._set_headers()
            response = {
                'success': True,
                'session_id': session_id,
                'company_profile': {
                    'user_id': session['user_id'],
                    'preferences': {r['question_id']: r['answer'] for r in responses}
                },
                'material_listings': filtered_materials,
                'completion_timestamp': datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Default response for unknown endpoints
        self._set_headers(404)
        self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())

class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Handle requests in a separate thread."""
    allow_reuse_address = True

def run_server(port=5003):
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, SimpleOnboardingHandler)
    print(f"Starting server on port {port}")
    print(f"Access URL: http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    print("Starting Simple Onboarding Server...")
    run_server()