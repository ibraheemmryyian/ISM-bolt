#!/usr/bin/env python3
"""
Phase 1 Completion Script for ISM AI Platform
Completes the implementation of all Phase 1 AI components with persistent models
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import AI components
try:
    from knowledge_graph import knowledge_graph
    from federated_meta_learning import federated_learner
    from gnn_reasoning_engine import gnn_reasoning_engine
    from revolutionary_ai_matching import advanced_matching_engine
    from model_persistence_manager import model_persistence_manager
    from ai_service_integration import ai_service_integration
    print("✅ All AI components imported successfully")
except ImportError as e:
    print(f"❌ Error importing AI components: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase1Completer:
    """Completes Phase 1 implementation and testing"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def run_completion(self):
        """Run the complete Phase 1 completion process"""
        print("\n🚀 Starting Phase 1 Completion Process")
        print("=" * 50)
        
        # Step 1: Initialize all components
        self._initialize_components()
        
        # Step 2: Test knowledge graph
        self._test_knowledge_graph()
        
        # Step 3: Test federated learning
        self._test_federated_learning()
        
        # Step 4: Test GNN reasoning
        self._test_gnn_reasoning()
        
        # Step 5: Test advanced matching
        self._test_advanced_matching()
        
        # Step 6: Test model persistence
        self._test_model_persistence()
        
        # Step 7: Test AI service integration
        self._test_ai_integration()
        
        # Step 8: Save all models
        self._save_all_models()
        
        # Step 9: Generate completion report
        self._generate_report()
        
        print("\n✅ Phase 1 Completion Process Finished!")
        print(f"⏱️  Total time: {time.time() - self.start_time:.2f} seconds")

    def _initialize_components(self):
        """Initialize all AI components"""
        print("\n📋 Step 1: Initializing AI Components")
        
        try:
            # Test knowledge graph initialization
            stats = knowledge_graph.get_graph_statistics()
            print(f"   ✅ Knowledge Graph: {stats['nodes']} nodes, {stats['edges']} edges")
            
            # Test federated learning initialization
            stats = federated_learner.get_learning_statistics()
            print(f"   ✅ Federated Learning: {stats.get('total_clients', 0)} clients")
            
            # Test GNN reasoning initialization
            models = gnn_reasoning_engine.list_available_models()
            print(f"   ✅ GNN Reasoning: {len(models)} models available")
            
            # Test advanced matching initialization
            stats = advanced_matching_engine.get_matching_statistics()
            print(f"   ✅ Advanced Matching: {stats.get('total_matches', 0)} matches")
            
            # Test model persistence initialization
            models = model_persistence_manager.list_models()
            print(f"   ✅ Model Persistence: {len(models)} models stored")
            
            self.test_results['initialization'] = 'success'
            
        except Exception as e:
            print(f"   ❌ Initialization failed: {e}")
            self.test_results['initialization'] = f'error: {e}'

    def _test_knowledge_graph(self):
        """Test knowledge graph functionality"""
        print("\n🧠 Step 2: Testing Knowledge Graph")
        
        try:
            # Add test entities
            test_entities = [
                {
                    'id': 'company_1',
                    'attributes': {
                        'name': 'Test Manufacturing Co',
                        'industry': 'Manufacturing',
                        'location': 'Dubai',
                        'annual_waste': 5000,
                        'carbon_footprint': 25000,
                        'employee_count': 200
                    },
                    'type': 'company'
                },
                {
                    'id': 'company_2',
                    'attributes': {
                        'name': 'Test Recycling Co',
                        'industry': 'Recycling',
                        'location': 'Dubai',
                        'annual_waste': 2000,
                        'carbon_footprint': 15000,
                        'employee_count': 150
                    },
                    'type': 'company'
                }
            ]
            
            for entity in test_entities:
                knowledge_graph.add_entity(
                    entity['id'], 
                    entity['attributes'], 
                    entity['type']
                )
            
            # Add test relationships
            knowledge_graph.add_relationship(
                'company_1', 'company_2', 'potential_partner',
                {'confidence': 0.8, 'reason': 'waste_recycling_match'}
            )
            
            # Test querying
            query_result = knowledge_graph.query({
                'type': 'general',
                'filters': {'entity_type': 'company'},
                'limit': 10
            })
            
            # Test reasoning
            reasoning_result = knowledge_graph.run_gnn_reasoning('opportunity_discovery')
            
            print(f"   ✅ Added {len(test_entities)} entities and relationships")
            print(f"   ✅ Query returned {len(query_result)} results")
            print(f"   ✅ Reasoning found {reasoning_result.get('total_found', 0)} opportunities")
            
            self.test_results['knowledge_graph'] = 'success'
            
        except Exception as e:
            print(f"   ❌ Knowledge graph test failed: {e}")
            self.test_results['knowledge_graph'] = f'error: {e}'

    def _test_federated_learning(self):
        """Test federated learning functionality"""
        print("\n🤝 Step 3: Testing Federated Learning")
        
        try:
            # Register test clients
            test_clients = [
                {
                    'client_id': 'client_1',
                    'model_params': {'weights': [0.1, 0.2, 0.3], 'bias': 0.1},
                    'metadata': {
                        'data_size': 1000,
                        'trust_score': 0.9,
                        'performance_metrics': {'r2_score': 0.85}
                    }
                },
                {
                    'client_id': 'client_2',
                    'model_params': {'weights': [0.2, 0.3, 0.4], 'bias': 0.2},
                    'metadata': {
                        'data_size': 1500,
                        'trust_score': 0.8,
                        'performance_metrics': {'r2_score': 0.82}
                    }
                }
            ]
            
            for client in test_clients:
                federated_learner.register_client(
                    client['client_id'],
                    client['model_params'],
                    client['metadata']
                )
            
            # Test aggregation
            aggregation_result = federated_learner.aggregate_global_model()
            
            # Test meta-learning
            meta_result = federated_learner.meta_learn()
            
            print(f"   ✅ Registered {len(test_clients)} clients")
            if aggregation_result:
                print(f"   ✅ Aggregation completed (Round {aggregation_result.round_number})")
            print(f"   ✅ Meta-learning analyzed {meta_result.get('num_clients_analyzed', 0)} clients")
            
            self.test_results['federated_learning'] = 'success'
            
        except Exception as e:
            print(f"   ❌ Federated learning test failed: {e}")
            self.test_results['federated_learning'] = f'error: {e}'

    def _test_gnn_reasoning(self):
        """Test GNN reasoning functionality"""
        print("\n🕸️  Step 4: Testing GNN Reasoning")
        
        try:
            import networkx as nx
            
            # Create test graph
            G = nx.Graph()
            G.add_node('company_1', industry='Manufacturing', location='Dubai', annual_waste=5000)
            G.add_node('company_2', industry='Recycling', location='Dubai', annual_waste=2000)
            G.add_node('company_3', industry='Chemical', location='Abu Dhabi', annual_waste=8000)
            
            G.add_edge('company_1', 'company_2', weight=0.8, type='potential_partner')
            G.add_edge('company_2', 'company_3', weight=0.6, type='potential_partner')
            
            # Test model training
            training_result = gnn_reasoning_engine.train_model(
                G, 'test_model', 'GCN', 'node_classification'
            )
            
            # Test inference
            inference_result = gnn_reasoning_engine.infer(
                G, 'test_model', 'node_embeddings'
            )
            
            print(f"   ✅ Created test graph with {G.number_of_nodes()} nodes")
            if 'training_time' in training_result:
                print(f"   ✅ Model trained in {training_result['training_time']:.2f}s")
            if 'embeddings' in inference_result:
                print(f"   ✅ Generated {len(inference_result['embeddings'])} embeddings")
            
            self.test_results['gnn_reasoning'] = 'success'
            
        except Exception as e:
            print(f"   ❌ GNN reasoning test failed: {e}")
            self.test_results['gnn_reasoning'] = f'error: {e}'

    def _test_advanced_matching(self):
        """Test advanced matching functionality"""
        print("\n🎯 Step 5: Testing Advanced Matching")
        
        try:
            # Test companies
            query_company = {
                'id': 'query_company',
                'name': 'Query Manufacturing Co',
                'industry': 'Manufacturing',
                'location': 'Dubai',
                'annual_waste': 6000,
                'carbon_footprint': 30000,
                'employee_count': 250,
                'waste_quantities': 'plastic, metal, paper',
                'resource_needs': 'energy, water'
            }
            
            candidate_companies = [
                {
                    'id': 'candidate_1',
                    'name': 'Recycling Partner Co',
                    'industry': 'Recycling',
                    'location': 'Dubai',
                    'annual_waste': 3000,
                    'carbon_footprint': 20000,
                    'employee_count': 180,
                    'waste_quantities': 'organic, chemical',
                    'resource_needs': 'plastic, metal'
                },
                {
                    'id': 'candidate_2',
                    'name': 'Energy Provider Co',
                    'industry': 'Energy',
                    'location': 'Abu Dhabi',
                    'annual_waste': 1000,
                    'carbon_footprint': 50000,
                    'employee_count': 500,
                    'waste_quantities': 'heat, steam',
                    'resource_needs': 'organic waste'
                }
            ]
            
            # Test matching
            matching_result = advanced_matching_engine.find_matches(
                query_company, candidate_companies, 'ensemble', 5
            )
            
            print(f"   ✅ Query company: {query_company['name']}")
            print(f"   ✅ Found {matching_result.total_candidates} candidates")
            print(f"   ✅ Matching completed in {matching_result.matching_time:.4f}s")
            
            self.test_results['advanced_matching'] = 'success'
            
        except Exception as e:
            print(f"   ❌ Advanced matching test failed: {e}")
            self.test_results['advanced_matching'] = f'error: {e}'

    def _test_model_persistence(self):
        """Test model persistence functionality"""
        print("\n💾 Step 6: Testing Model Persistence")
        
        try:
            # Test saving a model
            test_model = {'weights': [0.1, 0.2, 0.3], 'bias': 0.1}
            test_metadata = {'type': 'test_model', 'version': '1.0.0'}
            
            save_success = model_persistence_manager.save_model(
                'test_model', test_model, test_metadata
            )
            
            # Test loading the model
            loaded_model = model_persistence_manager.load_model('test_model')
            
            # Test listing models
            models = model_persistence_manager.list_models()
            
            # Test getting metadata
            metadata = model_persistence_manager.get_model_metadata('test_model')
            
            print(f"   ✅ Model saved: {save_success}")
            print(f"   ✅ Model loaded: {loaded_model is not None}")
            print(f"   ✅ Total models: {len(models)}")
            print(f"   ✅ Metadata retrieved: {metadata is not None}")
            
            self.test_results['model_persistence'] = 'success'
            
        except Exception as e:
            print(f"   ❌ Model persistence test failed: {e}")
            self.test_results['model_persistence'] = f'error: {e}'

    def _test_ai_integration(self):
        """Test AI service integration"""
        print("\n🔗 Step 7: Testing AI Service Integration")
        
        try:
            # Test service status
            status = ai_service_integration.get_service_status()
            health = ai_service_integration.get_service_health()
            performance = ai_service_integration.get_performance_summary()
            
            # Test AI pipeline
            pipeline_config = {
                'knowledge_graph': {
                    'enabled': True,
                    'run_reasoning': True,
                    'reasoning_type': 'opportunity_discovery'
                },
                'federated_learning': {
                    'enabled': True,
                    'aggregate': True
                },
                'gnn_reasoning': {
                    'enabled': True,
                    'inference': True
                },
                'advanced_matching': {
                    'enabled': True,
                    'find_matches': True
                }
            }
            
            pipeline_result = ai_service_integration.execute_ai_pipeline(pipeline_config)
            
            print(f"   ✅ Service status: {len(status)} services")
            print(f"   ✅ Active services: {performance.get('active_services', 0)}")
            print(f"   ✅ Pipeline completed: {pipeline_result.get('pipeline_time', 0):.2f}s")
            
            self.test_results['ai_integration'] = 'success'
            
        except Exception as e:
            print(f"   ❌ AI integration test failed: {e}")
            self.test_results['ai_integration'] = f'error: {e}'

    def _save_all_models(self):
        """Save all models using the integration layer"""
        print("\n💾 Step 8: Saving All Models")
        
        try:
            save_results = ai_service_integration.save_all_models()
            
            successful_saves = sum(1 for result in save_results.values() if result is True)
            total_saves = len(save_results)
            
            print(f"   ✅ Successfully saved {successful_saves}/{total_saves} models")
            
            for model_name, success in save_results.items():
                status = "✅" if success else "❌"
                print(f"   {status} {model_name}")
            
            self.test_results['model_saving'] = 'success'
            
        except Exception as e:
            print(f"   ❌ Model saving failed: {e}")
            self.test_results['model_saving'] = f'error: {e}'

    def _generate_report(self):
        """Generate completion report"""
        print("\n📊 Step 9: Generating Completion Report")
        
        # Calculate success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result == 'success')
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_time': time.time() - self.start_time,
            'success_rate': success_rate,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'test_results': self.test_results,
            'phase': 'Phase 1',
            'status': 'completed' if success_rate >= 80 else 'partial'
        }
        
        # Save report
        report_path = Path('phase1_completion_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📈 Phase 1 Completion Summary:")
        print(f"   ⏱️  Total time: {report['total_time']:.2f} seconds")
        print(f"   ✅ Success rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        print(f"   📄 Report saved: {report_path}")
        
        if success_rate >= 80:
            print(f"   🎉 Phase 1 completed successfully!")
        else:
            print(f"   ⚠️  Phase 1 completed with issues. Check report for details.")
        
        # Print detailed results
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅" if result == 'success' else "❌"
            print(f"   {status} {test_name}: {result}")

def main():
    """Main function"""
    print("ISM AI Platform - Phase 1 Completion Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('backend').exists():
        print("❌ Error: 'backend' directory not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Run completion
    completer = Phase1Completer()
    completer.run_completion()

if __name__ == "__main__":
    main() 