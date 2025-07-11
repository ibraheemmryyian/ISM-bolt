#!/usr/bin/env python3
"""
Adaptive Onboarding Initialization Script
CLI script for initializing the adaptive onboarding service
"""

import json
import sys
import os
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_adaptive_onboarding():
    """Initialize the adaptive onboarding service"""
    try:
        # Import the adaptive onboarding service
        import sys
        import os
        # Add current directory to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from adaptive_ai_onboarding import AdaptiveAIOnboarding
        
        # Initialize the service
        adaptive_onboarding = AdaptiveAIOnboarding()
        
        # Test basic functionality
        test_result = {
            'status': 'initialized',
            'timestamp': datetime.now().isoformat(),
            'service': 'adaptive_ai_onboarding',
            'features': {
                'federated_learning': adaptive_onboarding.federated_learner is not None,
                'compliance_engine': adaptive_onboarding.compliance_engine is not None,
                'questions_generator': adaptive_onboarding.questions_generator is not None
            },
            'question_templates': len(adaptive_onboarding.question_templates),
            'industry_questions': len(adaptive_onboarding.industry_questions),
            'compliance_requirements': len(adaptive_onboarding.compliance_requirements)
        }
        
        logger.info("✅ Adaptive onboarding service initialized successfully")
        return test_result
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize adaptive onboarding: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'service': 'adaptive_ai_onboarding'
        }

def health_check():
    """Perform a health check on the adaptive onboarding service"""
    try:
        import sys
        import os
        # Add current directory to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from adaptive_ai_onboarding import AdaptiveAIOnboarding
        
        adaptive_onboarding = AdaptiveAIOnboarding()
        
        health_result = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'adaptive_ai_onboarding',
            'active_sessions': len(adaptive_onboarding.active_sessions),
            'components': {
                'federated_learner': adaptive_onboarding.federated_learner is not None,
                'compliance_engine': adaptive_onboarding.compliance_engine is not None,
                'questions_generator': adaptive_onboarding.questions_generator is not None
            }
        }
        
        return health_result
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'service': 'adaptive_ai_onboarding'
        }

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print(json.dumps({
            'error': 'Action required: initialize, health_check, or restart_service'
        }))
        sys.exit(1)
    
    action = sys.argv[1]
    
    try:
        if action == 'initialize':
            result = initialize_adaptive_onboarding()
        elif action == 'health_check':
            result = health_check()
        elif action == 'restart_service':
            result = {
                'status': 'restarted',
                'timestamp': datetime.now().isoformat(),
                'service': 'adaptive_ai_onboarding',
                'message': 'Service restart requested'
            }
        else:
            result = {
                'error': f'Unknown action: {action}'
            }
        
        # Output JSON to stdout (this is what the backend expects)
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'service': 'adaptive_ai_onboarding'
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main() 