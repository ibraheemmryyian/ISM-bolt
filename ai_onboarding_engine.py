import json
from typing import Dict, List
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from ai_service import ai_service
import uuid
import argparse
import sys

class AIOnboardingEngine:
    def __init__(self, companies_data_path: str = 'companies.json'):
        with open(companies_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)["companies"]
        self.feedback_log = []
        self.learning_patterns = {}
        self.similarity_matrix = None
        self.vectorizer = None
        self._build_learning_models()

    def _build_learning_models(self):
        """Build contextual learning models from the company dataset"""
        print(f"Building contextual learning models from {len(self.data)} companies...")
        
        # Prepare text data for similarity analysis
        company_texts = []
        for company in self.data:
            text_parts = []
            if company.get('name'):
                text_parts.append(company['name'])
            if company.get('industry'):
                text_parts.append(company['industry'])
            if company.get('processes'):
                text_parts.append(company['processes'])
            if company.get('materials'):
                text_parts.extend(company['materials'])
            if company.get('location'):
                text_parts.append(company['location'])
            
            company_texts.append(' '.join(text_parts))
        
        # Build TF-IDF vectorizer for similarity analysis
        if company_texts:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = self.vectorizer.fit_transform(company_texts)
            self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Extract learning patterns
        self._extract_industry_patterns()
        self._extract_location_patterns()
        self._extract_process_patterns()
        self._extract_material_patterns()
        self._extract_success_patterns()

    def _extract_industry_patterns(self):
        """Extract patterns from industry data"""
        industry_data = {}
        for company in self.data:
            industry = company.get('industry', '').lower()
            if industry:
                if industry not in industry_data:
                    industry_data[industry] = {
                        'companies': [],
                        'common_materials': [],
                        'common_processes': [],
                        'avg_size': 0,
                        'locations': []
                    }
                industry_data[industry]['companies'].append(company)
                
                if company.get('materials'):
                    industry_data[industry]['common_materials'].extend(company['materials'])
                if company.get('processes'):
                    industry_data[industry]['common_processes'].append(company['processes'])
                if company.get('employee_count'):
                    industry_data[industry]['avg_size'] += company['employee_count']
                if company.get('location'):
                    industry_data[industry]['locations'].append(company['location'])
        
        # Calculate averages and most common items
        for industry, data in industry_data.items():
            if data['companies']:
                data['avg_size'] = data['avg_size'] / len(data['companies'])
                data['common_materials'] = Counter(data['common_materials']).most_common(5)
                data['locations'] = Counter(data['locations']).most_common(3)
        
        self.learning_patterns['industry'] = industry_data

    def _extract_location_patterns(self):
        """Extract patterns from location data"""
        location_data = {}
        for company in self.data:
            location = company.get('location', '').lower()
            if location:
                if location not in location_data:
                    location_data[location] = {
                        'companies': [],
                        'industries': [],
                        'regulatory_requirements': [],
                        'logistics_opportunities': []
                    }
                location_data[location]['companies'].append(company)
                location_data[location]['industries'].append(company.get('industry', ''))
        
        # Add location-specific insights
        for location, data in location_data.items():
            data['industries'] = Counter(data['industries']).most_common(3)
            
            # Location-specific patterns
            if 'cairo' in location or 'egypt' in location:
                data['regulatory_requirements'] = ['Egyptian Environmental Law', 'Waste Management Regulations']
                data['logistics_opportunities'] = ['Alexandria Port', 'Suez Canal', 'Regional Trade']
            elif 'new york' in location or 'ny' in location:
                data['regulatory_requirements'] = ['NY State Environmental Regulations', 'NYC Waste Management']
                data['logistics_opportunities'] = ['NYC Ports', 'Regional Distribution Centers']
        
        self.learning_patterns['location'] = location_data

    def _extract_process_patterns(self):
        """Extract patterns from process data"""
        process_patterns = {}
        for company in self.data:
            processes = company.get('processes', '')
            if processes:
                # Extract process steps
                process_steps = re.findall(r'([^→]+)→', processes)
                for step in process_steps:
                    step = step.strip().lower()
                    if step not in process_patterns:
                        process_patterns[step] = {
                            'frequency': 0,
                            'industries': [],
                            'waste_streams': [],
                            'efficiency_opportunities': []
                        }
                    process_patterns[step]['frequency'] += 1
                    process_patterns[step]['industries'].append(company.get('industry', ''))
        
        # Analyze process patterns
        for process, data in process_patterns.items():
            data['industries'] = Counter(data['industries']).most_common(3)
            
            # Add process-specific insights
            if 'cutting' in process:
                data['waste_streams'] = ['Cutting waste', 'Metal scraps', 'Cutting fluids']
                data['efficiency_opportunities'] = ['Optimize cutting patterns', 'Recycle cutting fluids']
            elif 'heating' in process:
                data['waste_streams'] = ['Waste heat', 'Combustion byproducts']
                data['efficiency_opportunities'] = ['Heat recovery systems', 'Energy optimization']
        
        self.learning_patterns['process'] = process_patterns

    def _extract_material_patterns(self):
        """Extract patterns from material data"""
        material_patterns = {}
        for company in self.data:
            materials = company.get('materials', [])
            for material in materials:
                material_lower = material.lower()
                if material_lower not in material_patterns:
                    material_patterns[material_lower] = {
                        'frequency': 0,
                        'industries': [],
                        'recycling_opportunities': [],
                        'substitution_options': []
                    }
                material_patterns[material_lower]['frequency'] += 1
                material_patterns[material_lower]['industries'].append(company.get('industry', ''))
        
        # Analyze material patterns
        for material, data in material_patterns.items():
            data['industries'] = Counter(data['industries']).most_common(3)
            
            # Add material-specific insights
            if 'plastic' in material:
                data['recycling_opportunities'] = ['Mechanical recycling', 'Chemical recycling', 'Energy recovery']
                data['substitution_options'] = ['Bioplastics', 'Recycled plastics', 'Alternative materials']
            elif 'metal' in material or 'steel' in material:
                data['recycling_opportunities'] = ['Metal recycling', 'Scrap metal recovery']
                data['substitution_options'] = ['Recycled metals', 'Lightweight alternatives']
        
        self.learning_patterns['material'] = material_patterns

    def _extract_success_patterns(self):
        """Extract patterns from successful companies"""
        success_patterns = {
            'high_completeness': [],
            'diverse_materials': [],
            'efficient_processes': [],
            'sustainability_focus': []
        }
        
        for company in self.data:
            # High completeness companies
            completeness_score = self._calculate_completeness_score(company)
            if completeness_score > 80:
                success_patterns['high_completeness'].append(company)
            
            # Companies with diverse materials
            if company.get('materials') and len(company['materials']) > 3:
                success_patterns['diverse_materials'].append(company)
            
            # Companies with detailed processes
            if company.get('processes') and len(company['processes']) > 100:
                success_patterns['efficient_processes'].append(company)
        
        self.learning_patterns['success'] = success_patterns

    def _calculate_completeness_score(self, company: Dict) -> float:
        """Calculate data completeness score for a company"""
        fields = ['name', 'industry', 'location', 'processes', 'materials', 'employee_count']
        filled_fields = sum(1 for field in fields if company.get(field))
        return (filled_fields / len(fields)) * 100

    def find_similar_companies(self, company_data: Dict, top_k: int = 5) -> List[Dict]:
        """Find similar companies using contextual learning"""
        if not self.vectorizer or not self.similarity_matrix:
            return []
        
        # Create text representation of the company
        company_text = ' '.join([
            company_data.get('name', ''),
            company_data.get('industry', ''),
            company_data.get('processes', ''),
            ' '.join(company_data.get('materials', [])),
            company_data.get('location', '')
        ])
        
        # Transform to TF-IDF
        company_vector = self.vectorizer.transform([company_text])
        
        # Calculate similarities
        similarities = cosine_similarity(company_vector, self.vectorizer.transform([
            ' '.join([
                c.get('name', ''),
                c.get('industry', ''),
                c.get('processes', ''),
                ' '.join(c.get('materials', [])),
                c.get('location', '')
            ]) for c in self.data
        ]))[0]
        
        # Get top similar companies
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        similar_companies = []
        
        for idx in similar_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                similar_companies.append({
                    'company': self.data[idx],
                    'similarity_score': similarities[idx],
                    'insights': self._generate_insights_from_similar_company(self.data[idx], company_data)
                })
        
        return similar_companies

    def _generate_insights_from_similar_company(self, similar_company: Dict, target_company: Dict) -> List[str]:
        """Generate insights based on similar company patterns"""
        insights = []
        
        # Industry insights
        if similar_company.get('industry') == target_company.get('industry'):
            if similar_company.get('materials') and not target_company.get('materials'):
                insights.append(f"Similar companies in {similar_company['industry']} commonly use: {', '.join(similar_company['materials'][:3])}")
        
        # Process insights
        if similar_company.get('processes') and not target_company.get('processes'):
            insights.append(f"Consider adding process details like: {similar_company['processes'][:100]}...")
        
        # Location insights
        if similar_company.get('location') == target_company.get('location'):
            insights.append(f"Companies in {similar_company['location']} often have similar regulatory requirements")
        
        return insights

    def get_contextual_recommendations(self, company_data: Dict) -> Dict:
        """Get contextual recommendations based on learned patterns"""
        recommendations = {
            'industry_insights': [],
            'location_insights': [],
            'process_insights': [],
            'material_insights': [],
            'similar_companies': [],
            'success_patterns': []
        }
        
        industry = company_data.get('industry', '').lower()
        location = company_data.get('location', '').lower()
        
        # Industry-based recommendations
        if industry and industry in self.learning_patterns.get('industry', {}):
            industry_data = self.learning_patterns['industry'][industry]
            recommendations['industry_insights'].extend([
                f"Average company size in {industry}: {industry_data['avg_size']:.0f} employees",
                f"Common materials: {', '.join([m[0] for m in industry_data['common_materials'][:3]])}",
                f"Popular locations: {', '.join([l[0] for l in industry_data['locations'][:3]])}"
            ])
        
        # Location-based recommendations
        if location and location in self.learning_patterns.get('location', {}):
            location_data = self.learning_patterns['location'][location]
            recommendations['location_insights'].extend([
                f"Regulatory requirements: {', '.join(location_data['regulatory_requirements'])}",
                f"Logistics opportunities: {', '.join(location_data['logistics_opportunities'])}"
            ])
        
        # Process-based recommendations
        processes = company_data.get('processes', '')
        if processes:
            for process, data in self.learning_patterns.get('process', {}).items():
                if process in processes.lower():
                    recommendations['process_insights'].extend([
                        f"Process '{process}' is common in: {', '.join([i[0] for i in data['industries'][:2]])}",
                        f"Waste streams: {', '.join(data['waste_streams'])}",
                        f"Efficiency opportunities: {', '.join(data['efficiency_opportunities'])}"
                    ])
        
        # Material-based recommendations
        materials = company_data.get('materials', [])
        for material in materials:
            material_lower = material.lower()
            if material_lower in self.learning_patterns.get('material', {}):
                material_data = self.learning_patterns['material'][material_lower]
                recommendations['material_insights'].extend([
                    f"Material '{material}' recycling opportunities: {', '.join(material_data['recycling_opportunities'])}",
                    f"Substitution options: {', '.join(material_data['substitution_options'])}"
                ])
        
        # Similar companies
        similar_companies = self.find_similar_companies(company_data, 3)
        recommendations['similar_companies'] = similar_companies
        
        # Success patterns
        completeness_score = self._calculate_completeness_score(company_data)
        if completeness_score < 70:
            recommendations['success_patterns'].append("High-performing companies typically have completeness scores above 80%")
        
        return recommendations

    def learn_from_feedback(self, company_id: str, feedback: Dict):
        """Learn from user feedback to improve future recommendations"""
        self.feedback_log.append({
            "company_id": company_id,
            "feedback": feedback,
            "timestamp": "2024-01-01T00:00:00Z"  # In production, use actual timestamp
        })
        
        # Update learning patterns based on feedback
        if feedback.get('useful_suggestions'):
            # Mark suggestions as useful for future recommendations
            pass
        
        if feedback.get('missing_information'):
            # Update patterns to include commonly missing information
            pass

    def generate_onboarding_flow(self, company_id: str, company_data: Dict = None) -> List[Dict]:
        # Accept either company_id or direct company_data
        if company_data:
            company = company_data
        else:
            company = next((c for c in self.data if c['id'] == company_id), None)
        if not company:
            return []
        industry = company.get('industry', '').lower()
        processes = company.get('processes', '').lower()
        materials = company.get('materials', [])
        location = company.get('location', '').lower()
        size = company.get('employee_count', None) or company.get('size', None)

        # Dynamic step generation
        steps = []
        # Step 1: Basic Info
        steps.append({
            "step": 1,
            "title": "Company Overview",
            "fields": [
                {"label": "Company Name", "type": "text", "key": "name", "required": True},
                {"label": "Industry", "type": "text", "key": "industry", "required": True},
                {"label": "Location", "type": "text", "key": "location", "required": True},
                {"label": "Company Size", "type": "number", "key": "employee_count", "required": False}
            ]
        })
        # Step 2: Industry-specific
        if industry:
            steps.append({
                "step": 2,
                "title": f"{industry} Details",
                "fields": self.get_industry_fields(industry)
            })
        # Step 3: Process Details
        if processes:
            steps.append({
                "step": 3,
                "title": "Production Processes",
                "fields": [
                    {"label": "Describe your main production process", "type": "textarea", "key": "processes", "required": True},
                    {"label": "Which process steps generate the most waste?", "type": "text", "key": "waste_steps", "required": False}
                ]
            })
        # Step 4: Materials
        if materials:
            steps.append({
                "step": 4,
                "title": "Materials",
                "fields": [
                    {"label": f"What are your main input materials? (e.g., {', '.join(materials)})", "type": "text", "key": "materials", "required": True}
                ]
            })
        # Step 5: Location-specific
        if location:
            steps.append({
                "step": 5,
                "title": "Location Considerations",
                "fields": self.get_location_fields(location)
            })
        # Step 6: Size-specific
        if size:
            steps.append({
                "step": 6,
                "title": "Scale & Operations",
                "fields": [
                    {"label": "How does your company size affect your operations?", "type": "textarea", "key": "size_impact", "required": False}
                ]
            })
        # Step 7: Final Review
        steps.append({
            "step": len(steps) + 1,
            "title": "Review & Submit",
            "fields": [
                {"label": "Review all information and submit onboarding.", "type": "info", "key": "review"}
            ]
        })
        return steps

    def generate_ai_questions(self, company_data: Dict) -> List[Dict]:
        """Generate AI-driven questions customized for each company/factory using advanced AI"""
        # First, get contextual recommendations from our dataset
        contextual_recs = self.get_contextual_recommendations(company_data)
        
        # Use advanced AI to generate intelligent questions
        ai_questions = ai_service.generate_intelligent_questions(company_data, 
            context=f"Industry insights: {contextual_recs.get('industry_insights', [])}")
        
        # Combine AI questions with contextual insights
        enhanced_questions = []
        
        # Add AI-generated questions
        for question in ai_questions:
            enhanced_questions.append({
                "question": question.get("question", ""),
                "type": question.get("type", "text"),
                "key": question.get("key", ""),
                "required": question.get("required", False),
                "reasoning": question.get("reasoning", ""),
                "category": question.get("category", "general"),
                "source": "advanced_ai"
            })
        
        # Add contextual insights as informational questions
        if contextual_recs.get('industry_insights'):
            enhanced_questions.append({
                "question": f"Based on our analysis of {len(self.data)} companies, here are insights for your industry:",
                "type": "info",
                "key": "industry_insights",
                "required": False,
                "reasoning": "Industry-specific insights help optimize your profile",
                "category": "insights",
                "source": "contextual_learning",
                "insights": contextual_recs['industry_insights']
            })
        
        if contextual_recs.get('similar_companies'):
            enhanced_questions.append({
                "question": f"We found {len(contextual_recs['similar_companies'])} similar companies in our database:",
                "type": "info",
                "key": "similar_companies",
                "required": False,
                "reasoning": "Similar companies can provide valuable insights",
                "category": "insights",
                "source": "contextual_learning",
                "similar_companies": contextual_recs['similar_companies']
            })
        
        return enhanced_questions

    def generate_enhanced_material_listings(self, company_data: Dict) -> List[Dict]:
        """Generate enhanced material listings using advanced AI and contextual learning"""
        # Get AI-generated listings
        ai_listings = ai_service.generate_material_listings(company_data)
        
        # Get contextual recommendations
        contextual_recs = self.get_contextual_recommendations(company_data)
        
        # Enhance listings with contextual insights
        enhanced_listings = []
        
        for listing in ai_listings:
            enhanced_listing = listing.copy()
            
            # Add contextual insights
            if contextual_recs.get('material_insights'):
                enhanced_listing['contextual_insights'] = contextual_recs['material_insights']
            
            # Add sustainability insights
            sustainability_insights = ai_service.generate_sustainability_insights(company_data)
            enhanced_listing['sustainability_analysis'] = sustainability_insights
            
            enhanced_listings.append(enhanced_listing)
        
        return enhanced_listings

    def get_comprehensive_analysis(self, company_data: Dict) -> Dict:
        """Get comprehensive analysis combining contextual learning and advanced AI"""
        # Get contextual recommendations
        contextual_recs = self.get_contextual_recommendations(company_data)
        
        # Get AI analysis
        ai_analysis = ai_service.analyze_company_data(company_data)
        
        # Get sustainability insights
        sustainability_insights = ai_service.generate_sustainability_insights(company_data)
        
        # Combine all analyses
        comprehensive_analysis = {
            "contextual_insights": contextual_recs,
            "ai_analysis": ai_analysis,
            "sustainability_insights": sustainability_insights,
            "similar_companies": contextual_recs.get('similar_companies', []),
            "data_quality_score": self.get_data_quality_score(company_data),
            "recommendations": {
                "immediate_actions": self._generate_immediate_actions(company_data, ai_analysis),
                "long_term_strategy": self._generate_long_term_strategy(company_data, sustainability_insights),
                "partnership_opportunities": self._generate_partnership_opportunities(company_data, contextual_recs)
            }
        }
        
        return comprehensive_analysis

    def _generate_immediate_actions(self, company_data: Dict, ai_analysis: Dict) -> List[str]:
        """Generate immediate actionable recommendations"""
        actions = []
        
        # Based on AI analysis
        if ai_analysis.get('waste_opportunities'):
            actions.append(f"Audit your waste streams: {', '.join(ai_analysis['waste_opportunities'][:3])}")
        
        if ai_analysis.get('improvement_areas'):
            actions.append(f"Focus on: {', '.join(ai_analysis['improvement_areas'][:2])}")
        
        # Based on data quality
        quality_score = self.get_data_quality_score(company_data)
        if quality_score['percentage'] < 70:
            actions.append("Complete your company profile to improve matching accuracy")
        
        return actions

    def _generate_long_term_strategy(self, company_data: Dict, sustainability_insights: Dict) -> List[str]:
        """Generate long-term strategic recommendations"""
        strategy = []
        
        if sustainability_insights.get('circular_economy_opportunities'):
            ce_opps = sustainability_insights['circular_economy_opportunities']
            if ce_opps.get('resource_recovery'):
                strategy.append(f"Develop resource recovery systems for: {', '.join(ce_opps['resource_recovery'][:2])}")
        
        if sustainability_insights.get('financial_benefits'):
            financial = sustainability_insights['financial_benefits']
            if financial.get('revenue_opportunities'):
                strategy.append(f"Explore new revenue streams: {', '.join(financial['revenue_opportunities'][:2])}")
        
        return strategy

    def _generate_partnership_opportunities(self, company_data: Dict, contextual_recs: Dict) -> List[Dict]:
        """Generate partnership opportunities based on contextual learning"""
        opportunities = []
        
        # Based on similar companies
        for similar in contextual_recs.get('similar_companies', []):
            if similar.get('similarity_score', 0) > 0.5:
                opportunities.append({
                    "type": "similar_company",
                    "description": f"Connect with {similar['company'].get('name', 'similar company')}",
                    "reason": f"High similarity score ({similar['similarity_score']:.2f})",
                    "potential_benefits": similar.get('insights', [])
                })
        
        # Based on industry patterns
        if contextual_recs.get('industry_insights'):
            opportunities.append({
                "type": "industry_network",
                "description": "Join industry-specific circular economy network",
                "reason": "Based on industry patterns from our dataset",
                "potential_benefits": ["Shared resources", "Collective bargaining", "Knowledge exchange"]
            })
        
        return opportunities

    def get_data_quality_score(self, company_data: Dict) -> Dict:
        """Calculate data quality score and improvement suggestions"""
        score = 0
        max_score = 100
        improvements = []
        
        # Basic information (30 points)
        if company_data.get('name'): score += 10
        if company_data.get('industry'): score += 10
        if company_data.get('location'): score += 10
        
        # Detailed information (40 points)
        if company_data.get('processes'): score += 15
        if company_data.get('materials'): score += 15
        if company_data.get('productionVolume'): score += 10
        
        # Advanced information (30 points)
        if company_data.get('employee_count'): score += 10
        if company_data.get('waste_streams'): score += 10
        if company_data.get('sustainability_goals'): score += 10
        
        # Calculate percentage
        quality_percentage = (score / max_score) * 100
        
        # Generate improvement suggestions
        if quality_percentage < 50:
            improvements.append("Add basic company information (name, industry, location)")
        if quality_percentage < 70:
            improvements.append("Add detailed process and materials information")
        if quality_percentage < 90:
            improvements.append("Add waste streams and sustainability goals")
        
        return {
            'score': score,
            'max_score': max_score,
            'percentage': quality_percentage,
            'grade': self._get_quality_grade(quality_percentage),
            'improvements': improvements
        }

    def _get_quality_grade(self, percentage: float) -> str:
        """Get letter grade for data quality"""
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"

    def get_industry_fields(self, industry: str) -> List[Dict]:
        templates = {
            "Textile Manufacturing": [
                {"label": "What types of fibers do you process?", "type": "text", "key": "fiber_types", "required": True},
                {"label": "Describe your dyeing and finishing processes.", "type": "textarea", "key": "dyeing_process", "required": False}
            ],
            "Electronics Manufacturing": [
                {"label": "What are your main electronic components?", "type": "text", "key": "components", "required": True},
                {"label": "Describe your PCB assembly process.", "type": "textarea", "key": "pcb_process", "required": False}
            ]
        }
        return templates.get(industry, [
            {"label": f"Describe your main activities in {industry}.", "type": "textarea", "key": "industry_activities", "required": True}
        ])

    def get_location_fields(self, location: str) -> List[Dict]:
        # Example: Add more location-specific logic as needed
        if "cairo" in location.lower():
            return [{"label": "Do you export/import via Alexandria port?", "type": "checkbox", "key": "use_alexandria_port", "required": False}]
        if "new york" in location.lower():
            return [{"label": "Are you subject to NY state environmental regulations?", "type": "checkbox", "key": "ny_regulations", "required": False}]
        return []

    def collect_feedback(self, company_id: str, feedback: Dict):
        self.feedback_log.append({"company_id": company_id, "feedback": feedback})

    def validate_and_enrich_company_data(self, company_data: Dict) -> Dict:
        """Real-time data validation and enrichment with auto-suggestions"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'enriched_data': company_data.copy(),
            'missing_fields': [],
            'auto_completed': []
        }
        
        # Validate required fields
        required_fields = ['name', 'industry', 'location']
        for field in required_fields:
            if not company_data.get(field) or str(company_data.get(field)).strip() == '':
                validation_results['errors'].append(f"{field} is required")
                validation_results['missing_fields'].append(field)
                validation_results['is_valid'] = False
        
        # Industry validation and enrichment
        industry = company_data.get('industry', '').lower()
        if industry:
            validation_results.update(self._validate_industry(industry, company_data))
        
        # Location validation and enrichment
        location = company_data.get('location', '').lower()
        if location:
            validation_results.update(self._validate_location(location, company_data))
        
        # Process validation and enrichment
        processes = company_data.get('processes', '')
        if processes:
            validation_results.update(self._validate_processes(processes, company_data))
        
        # Materials validation and enrichment
        materials = company_data.get('materials', [])
        if materials:
            validation_results.update(self._validate_materials(materials, company_data))
        
        # Size validation and enrichment
        size = company_data.get('employee_count')
        if size:
            validation_results.update(self._validate_size(size, company_data))
        
        # Auto-suggest missing information
        suggestions = self._generate_missing_info_suggestions(company_data)
        validation_results['suggestions'].extend(suggestions)
        
        # Auto-complete based on patterns
        auto_completed = self._auto_complete_fields(company_data)
        validation_results['auto_completed'].extend(auto_completed)
        
        return validation_results

    def _validate_industry(self, industry: str, company_data: Dict) -> Dict:
        """Validate and enrich industry data"""
        results = {'errors': [], 'warnings': [], 'suggestions': [], 'enriched_data': {}}
        
        # Check if industry exists in our dataset
        known_industries = [c.get('industry', '').lower() for c in self.data if c.get('industry')]
        industry_variations = {
            'textile': ['textile', 'fabric', 'clothing', 'apparel', 'garment'],
            'electronics': ['electronics', 'electronic', 'semiconductor', 'pcb', 'component'],
            'food': ['food', 'beverage', 'agriculture', 'farming', 'processing'],
            'chemical': ['chemical', 'pharmaceutical', 'pharma', 'petrochemical'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'industrial'],
            'construction': ['construction', 'building', 'cement', 'concrete']
        }
        
        # Find best match
        best_match = None
        for category, variations in industry_variations.items():
            if any(var in industry for var in variations):
                best_match = category
                break
        
        if not best_match:
            results['warnings'].append(f"Industry '{industry}' not recognized. Consider using standard industry categories.")
            results['suggestions'].append("Suggested industries: Textile Manufacturing, Electronics Manufacturing, Food Processing, Chemical Manufacturing, General Manufacturing, Construction")
        
        # Enrich with industry-specific data
        if best_match:
            results['enriched_data']['industry_category'] = best_match
            results['enriched_data']['standardized_industry'] = f"{best_match.title()} Manufacturing"
        
        return results

    def _validate_location(self, location: str, company_data: Dict) -> Dict:
        """Validate and enrich location data"""
        results = {'errors': [], 'warnings': [], 'suggestions': [], 'enriched_data': {}}
        
        # Check if location is in our dataset
        known_locations = [c.get('location', '').lower() for c in self.data if c.get('location')]
        
        # Standardize location format
        if ',' in location:
            city, country = location.split(',', 1)
            results['enriched_data']['city'] = city.strip().title()
            results['enriched_data']['country'] = country.strip().title()
        else:
            results['warnings'].append("Location format should be 'City, Country'")
            results['suggestions'].append("Example: 'New York, USA' or 'Cairo, Egypt'")
        
        # Add location-specific suggestions
        if 'cairo' in location.lower():
            results['suggestions'].append("Consider Alexandria port for export/import operations")
        elif 'new york' in location.lower():
            results['suggestions'].append("NY state has strict environmental regulations - ensure compliance")
        
        return results

    def _validate_processes(self, processes: str, company_data: Dict) -> Dict:
        """Validate and enrich process data"""
        results = {'errors': [], 'warnings': [], 'suggestions': [], 'enriched_data': {}}
        
        # Check for common process keywords
        process_keywords = ['cutting', 'machining', 'heating', 'cooling', 'cleaning', 'drying', 'mixing', 'assembly', 'packaging']
        found_processes = [kw for kw in process_keywords if kw in processes.lower()]
        
        if found_processes:
            results['enriched_data']['identified_processes'] = found_processes
        
        # Suggest missing process information
        industry = company_data.get('industry', '').lower()
        if 'textile' in industry and 'dyeing' not in processes.lower():
            results['suggestions'].append("Consider adding dyeing and finishing processes if applicable")
        elif 'electronics' in industry and 'assembly' not in processes.lower():
            results['suggestions'].append("Consider adding PCB assembly or component assembly processes")
        
        return results

    def _validate_materials(self, materials: List[str], company_data: Dict) -> Dict:
        """Validate and enrich materials data"""
        results = {'errors': [], 'warnings': [], 'suggestions': [], 'enriched_data': {}}
        
        # Check for material categorization
        material_categories = {
            'metals': ['steel', 'aluminum', 'copper', 'iron', 'metal'],
            'plastics': ['plastic', 'polymer', 'pet', 'hdpe', 'pvc'],
            'chemicals': ['chemical', 'solvent', 'acid', 'base', 'catalyst'],
            'organics': ['organic', 'biomass', 'food', 'agricultural', 'waste']
        }
        
        categorized_materials = {}
        for material in materials:
            material_lower = material.lower()
            for category, keywords in material_categories.items():
                if any(kw in material_lower for kw in keywords):
                    if category not in categorized_materials:
                        categorized_materials[category] = []
                    categorized_materials[category].append(material)
                    break
        
        if categorized_materials:
            results['enriched_data']['material_categories'] = categorized_materials
        
        # Suggest missing materials based on industry
        industry = company_data.get('industry', '').lower()
        if 'textile' in industry and not any('cotton' in m.lower() or 'fiber' in m.lower() for m in materials):
            results['suggestions'].append("Consider adding fiber types (cotton, polyester, wool) to materials list")
        
        return results

    def _validate_size(self, size: int, company_data: Dict) -> Dict:
        """Validate and enrich company size data"""
        results = {'errors': [], 'warnings': [], 'suggestions': [], 'enriched_data': {}}
        
        if size <= 0:
            results['errors'].append("Company size must be positive")
        elif size > 100000:
            results['warnings'].append("Company size seems unusually large - please verify")
        elif size < 10:
            results['warnings'].append("Company size seems unusually small - please verify")
        
        # Categorize company size
        if size < 50:
            size_category = "Small"
        elif size < 500:
            size_category = "Medium"
        elif size < 5000:
            size_category = "Large"
        else:
            size_category = "Enterprise"
        
        results['enriched_data']['size_category'] = size_category
        
        # Suggest based on size
        if size < 50:
            results['suggestions'].append("Small companies often benefit from shared waste management facilities")
        elif size > 1000:
            results['suggestions'].append("Large companies may qualify for dedicated sustainability programs")
        
        return results

    def _generate_missing_info_suggestions(self, company_data: Dict) -> List[str]:
        """Generate suggestions for missing information"""
        suggestions = []
        
        # Check for missing volume information
        if not company_data.get('productionVolume'):
            suggestions.append("Add production volume to help identify waste generation potential")
        
        # Check for missing process details
        if not company_data.get('processes'):
            suggestions.append("Describe your production processes to identify waste streams")
        
        # Check for missing materials
        if not company_data.get('materials'):
            suggestions.append("List your main input materials to identify recycling opportunities")
        
        # Check for missing waste information
        if not any('waste' in str(v).lower() for v in company_data.values()):
            suggestions.append("Consider adding information about your current waste streams")
        
        # Industry-specific suggestions
        industry = company_data.get('industry', '').lower()
        if 'textile' in industry:
            suggestions.append("Add information about dyeing processes and wastewater generation")
        elif 'electronics' in industry:
            suggestions.append("Add information about PCB waste and electronic component disposal")
        elif 'food' in industry:
            suggestions.append("Add information about organic waste and packaging materials")
        
        return suggestions

    def _auto_complete_fields(self, company_data: Dict) -> List[Dict]:
        """Auto-complete fields based on patterns and similar companies"""
        auto_completed = []
        
        industry = company_data.get('industry', '').lower()
        location = company_data.get('location', '').lower()
        
        # Find similar companies
        similar_companies = []
        for company in self.data:
            if (company.get('industry', '').lower() == industry and 
                company.get('location', '').lower() == location):
                similar_companies.append(company)
        
        if similar_companies:
            # Auto-suggest common materials
            common_materials = []
            for company in similar_companies:
                if company.get('materials'):
                    common_materials.extend(company.get('materials', []))
            
            if common_materials and not company_data.get('materials'):
                most_common = Counter(common_materials).most_common(3)
                auto_completed.append({
                    'field': 'materials',
                    'suggestion': [item[0] for item in most_common],
                    'reason': f"Based on {len(similar_companies)} similar companies in {industry} industry"
                })
            
            # Auto-suggest common processes
            common_processes = []
            for company in similar_companies:
                if company.get('processes'):
                    common_processes.append(company.get('processes'))
            
            if common_processes and not company_data.get('processes'):
                # Find most common process patterns
                process_patterns = []
                for process in common_processes:
                    if '→' in process:
                        process_patterns.append(process)
                
                if process_patterns:
                    auto_completed.append({
                        'field': 'processes',
                        'suggestion': process_patterns[0],
                        'reason': f"Common process pattern in {industry} industry"
                    })
        
        return auto_completed

    def get_industry_specific_templates(self) -> Dict[str, Dict]:
        """Get industry-specific onboarding templates"""
        return {
            "textile_manufacturing": {
                "name": "Textile Manufacturing",
                "description": "Specialized onboarding for textile and apparel manufacturers",
                "icon": "🧵",
                "steps": [
                    {
                        "step": 1,
                        "title": "Fiber & Material Information",
                        "fields": [
                            {"label": "What types of fibers do you primarily process?", "type": "text", "key": "fiber_types", "required": True, "options": ["Cotton", "Polyester", "Wool", "Silk", "Hemp", "Bamboo", "Recycled fibers", "Other"]},
                            {"label": "Do you use synthetic dyes?", "type": "checkbox", "key": "synthetic_dyes", "required": False},
                            {"label": "What is your monthly fiber consumption?", "type": "number", "key": "fiber_consumption", "required": False, "unit": "tons"},
                            {"label": "Do you have organic/natural fiber options?", "type": "checkbox", "key": "organic_fibers", "required": False}
                        ]
                    },
                    {
                        "step": 2,
                        "title": "Production Processes",
                        "fields": [
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "spinning", "required": False, "label": "Spinning"},
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "weaving", "required": False, "label": "Weaving"},
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "knitting", "required": False, "label": "Knitting"},
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "dyeing", "required": False, "label": "Dyeing & Finishing"},
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "cutting", "required": False, "label": "Cutting & Sewing"},
                            {"label": "Describe your dyeing process in detail", "type": "textarea", "key": "dyeing_process", "required": False}
                        ]
                    },
                    {
                        "step": 3,
                        "title": "Waste & Byproducts",
                        "fields": [
                            {"label": "What types of fabric waste do you generate?", "type": "text", "key": "fabric_waste", "required": False},
                            {"label": "Do you have wastewater treatment?", "type": "checkbox", "key": "wastewater_treatment", "required": False},
                            {"label": "What happens to your dye wastewater?", "type": "text", "key": "dye_wastewater", "required": False},
                            {"label": "Do you have thread/yarn waste?", "type": "checkbox", "key": "thread_waste", "required": False},
                            {"label": "Monthly waste volume", "type": "number", "key": "waste_volume", "required": False, "unit": "tons"}
                        ]
                    },
                    {
                        "step": 4,
                        "title": "Sustainability Goals",
                        "fields": [
                            {"label": "Are you certified organic?", "type": "checkbox", "key": "organic_certified", "required": False},
                            {"label": "Do you use recycled materials?", "type": "checkbox", "key": "recycled_materials", "required": False},
                            {"label": "What percentage of your materials are sustainable?", "type": "number", "key": "sustainable_percentage", "required": False, "unit": "%"},
                            {"label": "Describe your sustainability goals", "type": "textarea", "key": "sustainability_goals", "required": False}
                        ]
                    }
                ],
                "validation_rules": {
                    "fiber_types": "required",
                    "waste_volume": "positive_number",
                    "sustainable_percentage": "percentage_range"
                },
                "ai_insights": [
                    "Textile waste can be recycled into new fabrics",
                    "Dye wastewater can be treated and reused",
                    "Organic fibers have higher market value",
                    "Circular fashion is growing rapidly"
                ]
            },
            "electronics_manufacturing": {
                "name": "Electronics Manufacturing",
                "description": "Specialized onboarding for electronics and semiconductor manufacturers",
                "icon": "🔌",
                "steps": [
                    {
                        "step": 1,
                        "title": "Component Information",
                        "fields": [
                            {"label": "What types of components do you manufacture?", "type": "text", "key": "component_types", "required": True, "options": ["PCBs", "Semiconductors", "ICs", "Displays", "Batteries", "Connectors", "Other"]},
                            {"label": "Do you use precious metals?", "type": "checkbox", "key": "precious_metals", "required": False},
                            {"label": "Which precious metals?", "type": "text", "key": "precious_metal_types", "required": False, "options": ["Gold", "Silver", "Palladium", "Platinum", "Other"]},
                            {"label": "Monthly component production", "type": "number", "key": "component_production", "required": False, "unit": "units"}
                        ]
                    },
                    {
                        "step": 2,
                        "title": "Manufacturing Processes",
                        "fields": [
                            {"label": "Do you have PCB assembly?", "type": "checkbox", "key": "pcb_assembly", "required": False},
                            {"label": "Do you have surface mount technology (SMT)?", "type": "checkbox", "key": "smt", "required": False},
                            {"label": "Do you have through-hole assembly?", "type": "checkbox", "key": "through_hole", "required": False},
                            {"label": "Do you have testing facilities?", "type": "checkbox", "key": "testing", "required": False},
                            {"label": "Describe your manufacturing process", "type": "textarea", "key": "manufacturing_process", "required": False}
                        ]
                    },
                    {
                        "step": 3,
                        "title": "E-Waste & Materials",
                        "fields": [
                            {"label": "What types of e-waste do you generate?", "type": "text", "key": "ewaste_types", "required": False},
                            {"label": "Do you have component recovery processes?", "type": "checkbox", "key": "component_recovery", "required": False},
                            {"label": "What happens to defective components?", "type": "text", "key": "defective_components", "required": False},
                            {"label": "Monthly e-waste volume", "type": "number", "key": "ewaste_volume", "required": False, "unit": "tons"},
                            {"label": "Do you use lead-free solder?", "type": "checkbox", "key": "lead_free_solder", "required": False}
                        ]
                    },
                    {
                        "step": 4,
                        "title": "Compliance & Standards",
                        "fields": [
                            {"label": "Are you RoHS compliant?", "type": "checkbox", "key": "rohs_compliant", "required": False},
                            {"label": "Are you REACH compliant?", "type": "checkbox", "key": "reach_compliant", "required": False},
                            {"label": "Do you have ISO 14001 certification?", "type": "checkbox", "key": "iso_14001", "required": False},
                            {"label": "Describe your compliance strategy", "type": "textarea", "key": "compliance_strategy", "required": False}
                        ]
                    }
                ],
                "validation_rules": {
                    "component_types": "required",
                    "ewaste_volume": "positive_number",
                    "component_production": "positive_number"
                },
                "ai_insights": [
                    "Precious metals in e-waste have high recovery value",
                    "Component recovery can reduce costs significantly",
                    "RoHS compliance is mandatory in many markets",
                    "Circular electronics is a growing trend"
                ]
            },
            "food_processing": {
                "name": "Food Processing",
                "description": "Specialized onboarding for food and beverage manufacturers",
                "icon": "🍎",
                "steps": [
                    {
                        "step": 1,
                        "title": "Product Information",
                        "fields": [
                            {"label": "What types of food products do you process?", "type": "text", "key": "food_products", "required": True, "options": ["Fruits & Vegetables", "Grains", "Dairy", "Meat", "Beverages", "Snacks", "Frozen Foods", "Other"]},
                            {"label": "Do you have organic products?", "type": "checkbox", "key": "organic_products", "required": False},
                            {"label": "What is your daily production capacity?", "type": "number", "key": "daily_capacity", "required": False, "unit": "tons"},
                            {"label": "Do you have seasonal production?", "type": "checkbox", "key": "seasonal_production", "required": False}
                        ]
                    },
                    {
                        "step": 2,
                        "title": "Processing Methods",
                        "fields": [
                            {"label": "Which processing methods do you use?", "type": "checkbox", "key": "canning", "required": False, "label": "Canning"},
                            {"label": "Which processing methods do you use?", "type": "checkbox", "key": "freezing", "required": False, "label": "Freezing"},
                            {"label": "Which processing methods do you use?", "type": "checkbox", "key": "drying", "required": False, "label": "Drying"},
                            {"label": "Which processing methods do you use?", "type": "checkbox", "key": "fermentation", "required": False, "label": "Fermentation"},
                            {"label": "Which processing methods do you use?", "type": "checkbox", "key": "pasteurization", "required": False, "label": "Pasteurization"},
                            {"label": "Describe your main processing method", "type": "textarea", "key": "processing_method", "required": False}
                        ]
                    },
                    {
                        "step": 3,
                        "title": "Organic Waste & Byproducts",
                        "fields": [
                            {"label": "What types of organic waste do you generate?", "type": "text", "key": "organic_waste", "required": False},
                            {"label": "Do you have composting facilities?", "type": "checkbox", "key": "composting", "required": False},
                            {"label": "What happens to food scraps?", "type": "text", "key": "food_scraps", "required": False},
                            {"label": "Do you have animal feed production?", "type": "checkbox", "key": "animal_feed", "required": False},
                            {"label": "Monthly organic waste volume", "type": "number", "key": "organic_waste_volume", "required": False, "unit": "tons"}
                        ]
                    },
                    {
                        "step": 4,
                        "title": "Packaging & Sustainability",
                        "fields": [
                            {"label": "What packaging materials do you use?", "type": "text", "key": "packaging_materials", "required": False},
                            {"label": "Do you use recyclable packaging?", "type": "checkbox", "key": "recyclable_packaging", "required": False},
                            {"label": "What percentage is recyclable?", "type": "number", "key": "recyclable_percentage", "required": False, "unit": "%"},
                            {"label": "Describe your waste reduction goals", "type": "textarea", "key": "waste_reduction_goals", "required": False}
                        ]
                    }
                ],
                "validation_rules": {
                    "food_products": "required",
                    "organic_waste_volume": "positive_number",
                    "recyclable_percentage": "percentage_range"
                },
                "ai_insights": [
                    "Organic waste can be converted to biogas",
                    "Food scraps can be used for animal feed",
                    "Composting reduces disposal costs",
                    "Sustainable packaging increases market appeal"
                ]
            },
            "chemical_manufacturing": {
                "name": "Chemical Manufacturing",
                "description": "Specialized onboarding for chemical and pharmaceutical manufacturers",
                "icon": "🧪",
                "steps": [
                    {
                        "step": 1,
                        "title": "Chemical Products",
                        "fields": [
                            {"label": "What types of chemicals do you manufacture?", "type": "text", "key": "chemical_types", "required": True, "options": ["Pharmaceuticals", "Industrial Chemicals", "Agrochemicals", "Polymers", "Solvents", "Catalysts", "Other"]},
                            {"label": "Do you handle hazardous materials?", "type": "checkbox", "key": "hazardous_materials", "required": False},
                            {"label": "What safety certifications do you have?", "type": "text", "key": "safety_certifications", "required": False},
                            {"label": "Monthly chemical production", "type": "number", "key": "chemical_production", "required": False, "unit": "tons"}
                        ]
                    },
                    {
                        "step": 2,
                        "title": "Chemical Processes",
                        "fields": [
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "synthesis", "required": False, "label": "Chemical Synthesis"},
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "distillation", "required": False, "label": "Distillation"},
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "crystallization", "required": False, "label": "Crystallization"},
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "filtration", "required": False, "label": "Filtration"},
                            {"label": "Which processes do you use?", "type": "checkbox", "key": "drying", "required": False, "label": "Drying"},
                            {"label": "Describe your main chemical process", "type": "textarea", "key": "chemical_process", "required": False}
                        ]
                    },
                    {
                        "step": 3,
                        "title": "Waste & Recovery",
                        "fields": [
                            {"label": "What types of chemical waste do you generate?", "type": "text", "key": "chemical_waste", "required": False},
                            {"label": "Do you have solvent recovery?", "type": "checkbox", "key": "solvent_recovery", "required": False},
                            {"label": "Do you have catalyst recovery?", "type": "checkbox", "key": "catalyst_recovery", "required": False},
                            {"label": "What happens to spent catalysts?", "type": "text", "key": "spent_catalysts", "required": False},
                            {"label": "Monthly chemical waste volume", "type": "number", "key": "chemical_waste_volume", "required": False, "unit": "tons"}
                        ]
                    },
                    {
                        "step": 4,
                        "title": "Environmental Compliance",
                        "fields": [
                            {"label": "Do you have environmental permits?", "type": "checkbox", "key": "environmental_permits", "required": False},
                            {"label": "Are you ISO 14001 certified?", "type": "checkbox", "key": "iso_14001", "required": False},
                            {"label": "Do you have waste treatment facilities?", "type": "checkbox", "key": "waste_treatment", "required": False},
                            {"label": "Describe your environmental management", "type": "textarea", "key": "environmental_management", "required": False}
                        ]
                    }
                ],
                "validation_rules": {
                    "chemical_types": "required",
                    "chemical_waste_volume": "positive_number",
                    "chemical_production": "positive_number"
                },
                "ai_insights": [
                    "Solvent recovery can reduce costs by 60-80%",
                    "Catalyst recovery preserves precious metals",
                    "Chemical waste treatment is often mandatory",
                    "Green chemistry is a growing trend"
                ]
            }
        }

    def get_process_specific_templates(self) -> Dict[str, Dict]:
        """Get process-specific onboarding templates"""
        return {
            "cutting_machining": {
                "name": "Cutting & Machining",
                "description": "Specialized onboarding for cutting and machining operations",
                "icon": "⚙️",
                "steps": [
                    {
                        "step": 1,
                        "title": "Machining Operations",
                        "fields": [
                            {"label": "What materials do you machine?", "type": "text", "key": "machined_materials", "required": True, "options": ["Steel", "Aluminum", "Titanium", "Plastics", "Composites", "Other"]},
                            {"label": "What cutting tools do you use?", "type": "text", "key": "cutting_tools", "required": False},
                            {"label": "Do you use CNC machines?", "type": "checkbox", "key": "cnc_machines", "required": False},
                            {"label": "Monthly machining volume", "type": "number", "key": "machining_volume", "required": False, "unit": "tons"}
                        ]
                    },
                    {
                        "step": 2,
                        "title": "Cutting Fluids & Coolants",
                        "fields": [
                            {"label": "What types of cutting fluids do you use?", "type": "text", "key": "cutting_fluids", "required": False},
                            {"label": "Do you recycle cutting fluids?", "type": "checkbox", "key": "fluid_recycling", "required": False},
                            {"label": "What happens to used cutting fluids?", "type": "text", "key": "used_fluids", "required": False},
                            {"label": "Monthly fluid consumption", "type": "number", "key": "fluid_consumption", "required": False, "unit": "liters"}
                        ]
                    },
                    {
                        "step": 3,
                        "title": "Metal Waste & Scrap",
                        "fields": [
                            {"label": "What types of metal scrap do you generate?", "type": "text", "key": "metal_scrap", "required": False},
                            {"label": "Do you separate scrap by material?", "type": "checkbox", "key": "scrap_separation", "required": False},
                            {"label": "What is your scrap recycling rate?", "type": "number", "key": "scrap_recycling_rate", "required": False, "unit": "%"},
                            {"label": "Monthly scrap volume", "type": "number", "key": "scrap_volume", "required": False, "unit": "tons"}
                        ]
                    }
                ],
                "validation_rules": {
                    "machined_materials": "required",
                    "scrap_volume": "positive_number",
                    "scrap_recycling_rate": "percentage_range"
                },
                "ai_insights": [
                    "Cutting fluids can be recycled and reused",
                    "Metal scrap has high recycling value",
                    "CNC optimization reduces waste",
                    "Tool recycling programs are available"
                ]
            },
            "heating_furnace": {
                "name": "Heating & Furnace Operations",
                "description": "Specialized onboarding for heating and furnace operations",
                "icon": "🔥",
                "steps": [
                    {
                        "step": 1,
                        "title": "Furnace Types",
                        "fields": [
                            {"label": "What types of furnaces do you operate?", "type": "text", "key": "furnace_types", "required": True, "options": ["Heat Treatment", "Melting", "Drying", "Curing", "Annealing", "Other"]},
                            {"label": "What fuels do you use?", "type": "text", "key": "furnace_fuels", "required": False, "options": ["Natural Gas", "Electric", "Oil", "Coal", "Biomass", "Other"]},
                            {"label": "What is your furnace capacity?", "type": "number", "key": "furnace_capacity", "required": False, "unit": "tons/hour"},
                            {"label": "Operating temperature range", "type": "text", "key": "temperature_range", "required": False}
                        ]
                    },
                    {
                        "step": 2,
                        "title": "Energy & Heat Recovery",
                        "fields": [
                            {"label": "Do you have waste heat recovery?", "type": "checkbox", "key": "heat_recovery", "required": False},
                            {"label": "How do you use recovered heat?", "type": "text", "key": "heat_usage", "required": False},
                            {"label": "What is your energy efficiency?", "type": "number", "key": "energy_efficiency", "required": False, "unit": "%"},
                            {"label": "Monthly energy consumption", "type": "number", "key": "energy_consumption", "required": False, "unit": "MWh"}
                        ]
                    },
                    {
                        "step": 3,
                        "title": "Emissions & Byproducts",
                        "fields": [
                            {"label": "What emissions do you generate?", "type": "text", "key": "emissions", "required": False},
                            {"label": "Do you have emission controls?", "type": "checkbox", "key": "emission_controls", "required": False},
                            {"label": "What happens to combustion byproducts?", "type": "text", "key": "combustion_byproducts", "required": False},
                            {"label": "Monthly CO2 emissions", "type": "number", "key": "co2_emissions", "required": False, "unit": "tons"}
                        ]
                    }
                ],
                "validation_rules": {
                    "furnace_types": "required",
                    "energy_consumption": "positive_number",
                    "energy_efficiency": "percentage_range"
                },
                "ai_insights": [
                    "Waste heat can be used for other processes",
                    "Energy efficiency reduces costs significantly",
                    "Emission controls are often mandatory",
                    "Carbon capture opportunities exist"
                ]
            }
        }

    def get_industry_specific_onboarding_flow(self, industry: str, company_data: Dict = None) -> List[Dict]:
        """Generate industry-specific onboarding flow"""
        templates = self.get_industry_specific_templates()
        
        if industry.lower() in templates:
            template = templates[industry.lower()]
            flow = template["steps"].copy()
            
            # Add AI-generated questions specific to this industry
            if company_data:
                ai_questions = ai_service.generate_intelligent_questions(company_data, 
                    context=f"Industry: {template['name']}")
                
                # Add AI questions as additional step
                if ai_questions:
                    flow.append({
                        "step": len(flow) + 1,
                        "title": "AI-Generated Industry Questions",
                        "fields": ai_questions
                    })
            
            # Add final review step
            flow.append({
                "step": len(flow) + 1,
                "title": "Review & Submit",
                "fields": [
                    {"label": "Review all information and submit onboarding.", "type": "info", "key": "review", "required": False}
                ]
            })
            
            return flow
        
        # Fallback to generic flow
        return self.generate_onboarding_flow(None, company_data)

    def validate_industry_specific_data(self, industry: str, company_data: Dict) -> Dict:
        """Validate data against industry-specific rules"""
        templates = self.get_industry_specific_templates()
        
        if industry.lower() not in templates:
            return {"is_valid": True, "errors": [], "warnings": []}
        
        template = templates[industry.lower()]
        validation_rules = template.get("validation_rules", {})
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "industry_insights": template.get("ai_insights", [])
        }
        
        # Apply industry-specific validation rules
        for field, rule in validation_rules.items():
            value = company_data.get(field)
            
            if rule == "required" and not value:
                validation_results["errors"].append(f"{field} is required for {template['name']}")
                validation_results["is_valid"] = False
            
            elif rule == "positive_number" and value is not None:
                try:
                    if float(value) <= 0:
                        validation_results["errors"].append(f"{field} must be a positive number")
                        validation_results["is_valid"] = False
                except (ValueError, TypeError):
                    validation_results["warnings"].append(f"{field} should be a valid number")
            
            elif rule == "percentage_range" and value is not None:
                try:
                    percentage = float(value)
                    if percentage < 0 or percentage > 100:
                        validation_results["errors"].append(f"{field} must be between 0 and 100")
                        validation_results["is_valid"] = False
                except (ValueError, TypeError):
                    validation_results["warnings"].append(f"{field} should be a valid percentage")
        
        return validation_results

    def collect_onboarding_feedback(self, company_id: str, onboarding_data: Dict, feedback: Dict) -> Dict:
        """Collect comprehensive feedback for onboarding improvement"""
        feedback_entry = {
            "company_id": company_id,
            "timestamp": "2024-01-01T00:00:00Z",  # In production, use actual timestamp
            "onboarding_data": onboarding_data,
            "feedback": feedback,
            "metrics": self._calculate_onboarding_metrics(onboarding_data, feedback),
            "improvement_suggestions": []
        }
        
        # Analyze feedback for improvement opportunities
        improvement_suggestions = self._analyze_feedback_for_improvements(feedback_entry)
        feedback_entry["improvement_suggestions"] = improvement_suggestions
        
        # Store feedback
        self.feedback_log.append(feedback_entry)
        
        # Update learning patterns based on feedback
        self._update_learning_patterns_from_feedback(feedback_entry)
        
        return feedback_entry

    def _calculate_onboarding_metrics(self, onboarding_data: Dict, feedback: Dict) -> Dict:
        """Calculate onboarding success metrics"""
        metrics = {
            "completion_time": feedback.get("completion_time_minutes", 0),
            "data_quality_score": self.get_data_quality_score(onboarding_data)["percentage"],
            "user_satisfaction": feedback.get("satisfaction_score", 0),
            "questions_answered": len([f for f in onboarding_data.values() if f]),
            "required_fields_completed": 0,
            "optional_fields_completed": 0,
            "ai_suggestions_used": feedback.get("ai_suggestions_used", 0),
            "validation_errors": feedback.get("validation_errors", 0),
            "dropout_point": feedback.get("dropout_step", None)
        }
        
        # Calculate field completion rates
        total_fields = len(onboarding_data)
        completed_fields = len([f for f in onboarding_data.values() if f])
        metrics["completion_rate"] = (completed_fields / total_fields * 100) if total_fields > 0 else 0
        
        return metrics

    def _analyze_feedback_for_improvements(self, feedback_entry: Dict) -> List[str]:
        """Analyze feedback to generate improvement suggestions"""
        suggestions = []
        feedback = feedback_entry["feedback"]
        metrics = feedback_entry["metrics"]
        
        # Analyze completion time
        if metrics["completion_time"] > 30:
            suggestions.append("Consider simplifying onboarding flow - completion time is high")
        
        # Analyze satisfaction
        if metrics["user_satisfaction"] < 7:
            suggestions.append("User satisfaction is low - review onboarding experience")
        
        # Analyze dropout points
        if metrics["dropout_point"]:
            suggestions.append(f"High dropout at step {metrics['dropout_point']} - review step complexity")
        
        # Analyze AI suggestions usage
        if metrics["ai_suggestions_used"] == 0:
            suggestions.append("AI suggestions not being used - review suggestion relevance")
        
        # Analyze validation errors
        if metrics["validation_errors"] > 5:
            suggestions.append("High validation errors - review field requirements and validation rules")
        
        # Analyze specific feedback
        if feedback.get("difficult_questions"):
            suggestions.append("Users find some questions difficult - consider simplifying or adding help text")
        
        if feedback.get("missing_fields"):
            suggestions.append("Users report missing fields - review field coverage")
        
        if feedback.get("unclear_instructions"):
            suggestions.append("Instructions are unclear - improve field descriptions and help text")
        
        return suggestions

    def _update_learning_patterns_from_feedback(self, feedback_entry: Dict):
        """Update learning patterns based on user feedback"""
        onboarding_data = feedback_entry["onboarding_data"]
        feedback = feedback_entry["feedback"]
        
        # Update industry patterns
        industry = onboarding_data.get("industry", "").lower()
        if industry and industry in self.learning_patterns.get("industry", {}):
            industry_data = self.learning_patterns["industry"][industry]
            
            # Update common materials based on user input
            if onboarding_data.get("materials"):
                if "common_materials" not in industry_data:
                    industry_data["common_materials"] = []
                industry_data["common_materials"].extend(onboarding_data["materials"])
            
            # Update average company size
            if onboarding_data.get("employee_count"):
                current_avg = industry_data.get("avg_size", 0)
                company_count = len(industry_data.get("companies", []))
                if company_count > 0:
                    new_avg = (current_avg * company_count + onboarding_data["employee_count"]) / (company_count + 1)
                    industry_data["avg_size"] = new_avg

    def get_onboarding_analytics(self) -> Dict:
        """Get comprehensive onboarding analytics"""
        if not self.feedback_log:
            return {"message": "No feedback data available"}
        
        analytics = {
            "total_onboardings": len(self.feedback_log),
            "average_completion_time": 0,
            "average_satisfaction": 0,
            "average_completion_rate": 0,
            "industry_breakdown": {},
            "common_issues": [],
            "improvement_areas": [],
            "success_metrics": {}
        }
        
        # Calculate averages
        total_time = sum(f["metrics"]["completion_time"] for f in self.feedback_log)
        total_satisfaction = sum(f["metrics"]["user_satisfaction"] for f in self.feedback_log)
        total_completion_rate = sum(f["metrics"]["completion_rate"] for f in self.feedback_log)
        
        if self.feedback_log:
            analytics["average_completion_time"] = total_time / len(self.feedback_log)
            analytics["average_satisfaction"] = total_satisfaction / len(self.feedback_log)
            analytics["average_completion_rate"] = total_completion_rate / len(self.feedback_log)
        
        # Industry breakdown
        for feedback in self.feedback_log:
            industry = feedback["onboarding_data"].get("industry", "Unknown")
            if industry not in analytics["industry_breakdown"]:
                analytics["industry_breakdown"][industry] = {
                    "count": 0,
                    "avg_satisfaction": 0,
                    "avg_completion_rate": 0
                }
            analytics["industry_breakdown"][industry]["count"] += 1
        
        # Calculate industry averages
        for industry, data in analytics["industry_breakdown"].items():
            industry_feedbacks = [f for f in self.feedback_log if f["onboarding_data"].get("industry") == industry]
            if industry_feedbacks:
                data["avg_satisfaction"] = sum(f["metrics"]["user_satisfaction"] for f in industry_feedbacks) / len(industry_feedbacks)
                data["avg_completion_rate"] = sum(f["metrics"]["completion_rate"] for f in industry_feedbacks) / len(industry_feedbacks)
        
        # Common issues
        all_suggestions = []
        for feedback in self.feedback_log:
            all_suggestions.extend(feedback["improvement_suggestions"])
        
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        analytics["common_issues"] = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Success metrics
        analytics["success_metrics"] = {
            "high_satisfaction_rate": len([f for f in self.feedback_log if f["metrics"]["user_satisfaction"] >= 8]) / len(self.feedback_log) * 100,
            "high_completion_rate": len([f for f in self.feedback_log if f["metrics"]["completion_rate"] >= 90]) / len(self.feedback_log) * 100,
            "fast_completion_rate": len([f for f in self.feedback_log if f["metrics"]["completion_time"] <= 15]) / len(self.feedback_log) * 100
        }
        
        return analytics

    def improve_onboarding_templates(self) -> Dict:
        """Improve onboarding templates based on feedback analysis"""
        analytics = self.get_onboarding_analytics()
        improvements = {
            "template_updates": [],
            "new_templates": [],
            "validation_rule_updates": [],
            "ai_question_improvements": []
        }
        
        # Analyze common issues and suggest template improvements
        for issue, count in analytics.get("common_issues", []):
            if "completion time" in issue.lower():
                improvements["template_updates"].append({
                    "type": "simplify_flow",
                    "reason": issue,
                    "frequency": count,
                    "suggestion": "Reduce number of steps or combine related fields"
                })
            
            elif "difficult questions" in issue.lower():
                improvements["ai_question_improvements"].append({
                    "type": "simplify_questions",
                    "reason": issue,
                    "frequency": count,
                    "suggestion": "Add help text and simplify question language"
                })
            
            elif "validation errors" in issue.lower():
                improvements["validation_rule_updates"].append({
                    "type": "relax_validation",
                    "reason": issue,
                    "frequency": count,
                    "suggestion": "Review and potentially relax validation rules"
                })
        
        # Industry-specific improvements
        for industry, data in analytics.get("industry_breakdown", {}).items():
            if data["avg_satisfaction"] < 7:
                improvements["template_updates"].append({
                    "type": "industry_specific",
                    "industry": industry,
                    "reason": f"Low satisfaction in {industry}",
                    "suggestion": "Review and improve industry-specific questions"
                })
        
        return improvements

    def get_active_learning_recommendations(self, company_data: Dict) -> Dict:
        """Get active learning recommendations based on feedback analysis"""
        recommendations = {
            "template_suggestions": [],
            "question_improvements": [],
            "validation_adjustments": [],
            "success_patterns": []
        }
        
        # Analyze successful onboarding patterns
        successful_onboardings = [f for f in self.feedback_log if f["metrics"]["user_satisfaction"] >= 8 and f["metrics"]["completion_rate"] >= 90]
        
        if successful_onboardings:
            # Find common patterns in successful onboardings
            common_fields = {}
            for onboarding in successful_onboardings:
                for field, value in onboarding["onboarding_data"].items():
                    if value:
                        common_fields[field] = common_fields.get(field, 0) + 1
            
            # Suggest fields that are commonly completed
            for field, count in sorted(common_fields.items(), key=lambda x: x[1], reverse=True)[:5]:
                if count >= len(successful_onboardings) * 0.8:  # 80% completion rate
                    recommendations["success_patterns"].append({
                        "field": field,
                        "completion_rate": count / len(successful_onboardings) * 100,
                        "suggestion": f"Field '{field}' is commonly completed - consider making it required"
                    })
        
        # Analyze current company data against successful patterns
        missing_successful_fields = []
        for pattern in recommendations["success_patterns"]:
            field = pattern["field"]
            if not company_data.get(field) and pattern["completion_rate"] >= 90:
                missing_successful_fields.append(field)
        
        if missing_successful_fields:
            recommendations["template_suggestions"].append({
                "type": "add_fields",
                "fields": missing_successful_fields,
                "reason": "These fields are commonly completed in successful onboardings",
                "priority": "high"
            })
        
        return recommendations

    def update_templates_from_feedback(self, feedback_analysis: Dict):
        """Update templates based on feedback analysis"""
        templates = self.get_industry_specific_templates()
        
        for improvement in feedback_analysis.get("template_updates", []):
            if improvement["type"] == "simplify_flow":
                # Simplify onboarding flows by reducing steps
                for industry, template in templates.items():
                    if len(template["steps"]) > 4:
                        # Combine related steps
                        simplified_steps = []
                        for i in range(0, len(template["steps"]), 2):
                            if i + 1 < len(template["steps"]):
                                # Combine two steps
                                combined_step = {
                                    "step": len(simplified_steps) + 1,
                                    "title": f"{template['steps'][i]['title']} & {template['steps'][i+1]['title']}",
                                    "fields": template["steps"][i]["fields"] + template["steps"][i+1]["fields"]
                                }
                                simplified_steps.append(combined_step)
                            else:
                                simplified_steps.append(template["steps"][i])
                        
                        template["steps"] = simplified_steps
            
            elif improvement["type"] == "industry_specific":
                industry = improvement.get("industry", "").lower()
                if industry in templates:
                    # Add help text to industry-specific questions
                    for step in templates[industry]["steps"]:
                        for field in step["fields"]:
                            if "help_text" not in field:
                                field["help_text"] = f"Please provide detailed information about your {field['key']} to help us better understand your needs."
        
        return templates

    def get_feedback_summary(self) -> Dict:
        """Get a summary of feedback and improvements"""
        analytics = self.get_onboarding_analytics()
        improvements = self.improve_onboarding_templates()
        
        return {
            "analytics": analytics,
            "improvements": improvements,
            "total_feedback_entries": len(self.feedback_log),
            "last_updated": "2024-01-01T00:00:00Z",  # In production, use actual timestamp
            "recommendations": {
                "immediate_actions": self._get_immediate_improvement_actions(analytics),
                "long_term_improvements": self._get_long_term_improvement_actions(improvements)
            }
        }

    def _get_immediate_improvement_actions(self, analytics: Dict) -> List[str]:
        """Get immediate actions based on analytics"""
        actions = []
        
        if analytics["average_satisfaction"] < 7:
            actions.append("Review and simplify onboarding flow immediately")
        
        if analytics["average_completion_time"] > 30:
            actions.append("Reduce onboarding complexity to improve completion time")
        
        if analytics["success_metrics"]["high_completion_rate"] < 80:
            actions.append("Add more help text and guidance to improve completion rates")
        
        return actions

    def _get_long_term_improvement_actions(self, improvements: Dict) -> List[str]:
        """Get long-term improvement actions"""
        actions = []
        
        if improvements["template_updates"]:
            actions.append("Implement template improvements based on user feedback")
        
        if improvements["ai_question_improvements"]:
            actions.append("Enhance AI question generation based on difficulty feedback")
        
        if improvements["validation_rule_updates"]:
            actions.append("Review and optimize validation rules")
        
        return actions 

    def process_conversational_input(self, user_input: str, conversation_context: Dict) -> Dict:
        """Process conversational input and generate intelligent responses"""
        # Analyze user input using AI
        ai_analysis = ai_service.analyze_conversational_input(user_input, conversation_context)
        
        # Extract intent and entities
        intent = ai_analysis.get("intent", "general")
        entities = ai_analysis.get("entities", {})
        sentiment = ai_analysis.get("sentiment", "neutral")
        
        # Generate appropriate response based on intent
        response = self._generate_conversational_response(intent, entities, sentiment, conversation_context)
        
        # Update conversation context
        updated_context = self._update_conversation_context(conversation_context, user_input, intent, entities)
        
        return {
            "response": response,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "updated_context": updated_context,
            "next_questions": self._get_next_questions(updated_context),
            "confidence_score": ai_analysis.get("confidence", 0.7)
        }

    def _generate_conversational_response(self, intent: str, entities: Dict, sentiment: str, context: Dict) -> str:
        """Generate conversational response based on intent and context"""
        if intent == "provide_company_info":
            return self._handle_company_info_provision(entities, context)
        elif intent == "ask_question":
            return self._handle_question_asking(entities, context)
        elif intent == "clarify_information":
            return self._handle_clarification(entities, context)
        elif intent == "express_concern":
            return self._handle_concern_expression(entities, sentiment, context)
        elif intent == "request_help":
            return self._handle_help_request(entities, context)
        elif intent == "confirm_information":
            return self._handle_confirmation(entities, context)
        else:
            return self._handle_general_conversation(intent, entities, context)

    def _handle_company_info_provision(self, entities: Dict, context: Dict) -> str:
        """Handle when user provides company information"""
        company_data = context.get("company_data", {})
        
        # Update company data with provided information
        for entity_type, value in entities.items():
            if entity_type in ["company_name", "industry", "location", "size", "materials", "processes"]:
                company_data[entity_type] = value
        
        # Validate the provided information
        validation = self.validate_and_enrich_company_data(company_data)
        
        if validation["is_valid"]:
            return f"Thank you for providing that information about {entities.get('company_name', 'your company')}. I've updated your profile. What else would you like to tell me about your operations?"
        else:
            return f"I've noted that information, but I need a bit more detail. Could you please clarify: {', '.join(validation['errors'][:2])}"

    def _handle_question_asking(self, entities: Dict, context: Dict) -> str:
        """Handle when user asks questions"""
        question_topic = entities.get("question_topic", "general")
        
        if question_topic == "onboarding_process":
            return "The onboarding process helps us understand your company's operations so we can find the best circular economy opportunities for you. We'll ask about your industry, processes, materials, and sustainability goals."
        elif question_topic == "benefits":
            return "By joining our platform, you can reduce waste disposal costs, find new revenue streams from byproducts, meet sustainability goals, and connect with other companies for resource sharing."
        elif question_topic == "time_commitment":
            return "The onboarding typically takes 10-15 minutes. You can save and continue later if needed. The more detailed information you provide, the better matches we can find for you."
        else:
            return "I'm here to help you through the onboarding process. What specific questions do you have about our circular economy platform?"

    def _handle_clarification(self, entities: Dict, context: Dict) -> str:
        """Handle when user needs clarification"""
        unclear_topic = entities.get("unclear_topic", "general")
        
        clarifications = {
            "industry": "Your industry helps us understand your typical processes and waste streams. For example, textile manufacturing, food processing, or electronics manufacturing.",
            "materials": "List the main materials you use in production, such as cotton, steel, plastic, or chemicals. This helps us identify recycling opportunities.",
            "processes": "Describe your main production processes, like cutting, heating, assembly, or chemical reactions. This helps us understand your waste streams.",
            "waste": "Tell us about materials you currently dispose of or want to find better uses for, such as scrap metal, organic waste, or chemical byproducts."
        }
        
        return clarifications.get(unclear_topic, "Could you please rephrase your question? I want to make sure I understand exactly what you need help with.")

    def _handle_concern_expression(self, entities: Dict, sentiment: str, context: Dict) -> str:
        """Handle when user expresses concerns"""
        concern_type = entities.get("concern_type", "general")
        
        if sentiment == "negative":
            if concern_type == "data_security":
                return "I understand your concern about data security. We take this very seriously. All your information is encrypted and only used to find circular economy opportunities. We never share your data with third parties without permission."
            elif concern_type == "time":
                return "I appreciate your time is valuable. The onboarding is designed to be efficient, and the time investment now will save you money and create new opportunities later. We can also break it into shorter sessions."
            elif concern_type == "complexity":
                return "I understand this might seem complex at first. Let me guide you through it step by step. We can start with the basics and add more detail as we go. What would you like to begin with?"
            else:
                return "I hear your concern. Let me address it directly. What specific aspect of the onboarding process is worrying you? I'm here to make this as smooth as possible for you."
        else:
            return "Thank you for sharing that. Let me help you address any concerns you have about the onboarding process."

    def _handle_help_request(self, entities: Dict, context: Dict) -> str:
        """Handle when user requests help"""
        help_topic = entities.get("help_topic", "general")
        
        help_responses = {
            "start": "Let's start with the basics. What's your company name and what industry are you in?",
            "continue": "Great! Let's continue. What are your main production processes?",
            "save": "You can save your progress at any time. Your information is automatically saved as you go.",
            "skip": "You can skip optional questions and come back to them later. Required fields are marked with an asterisk (*).",
            "examples": "Here are some examples: For materials, you might list 'cotton, polyester, dyes'. For processes, you might say 'spinning, weaving, dyeing'."
        }
        
        return help_responses.get(help_topic, "I'm here to help! What specific assistance do you need with the onboarding process?")

    def _handle_confirmation(self, entities: Dict, context: Dict) -> str:
        """Handle when user confirms information"""
        confirmed_info = entities.get("confirmed_info", "general")
        
        return f"Perfect! I've confirmed your {confirmed_info} information. Let's continue with the next step. What would you like to tell me about your operations?"

    def _handle_general_conversation(self, intent: str, entities: Dict, context: Dict) -> str:
        """Handle general conversation"""
        return "I'm here to help you complete your onboarding. Let's work together to understand your company's circular economy potential. What would you like to start with?"

    def _update_conversation_context(self, context: Dict, user_input: str, intent: str, entities: Dict) -> Dict:
        """Update conversation context with new information"""
        updated_context = context.copy()
        
        # Update company data with extracted entities
        if "company_data" not in updated_context:
            updated_context["company_data"] = {}
        
        for entity_type, value in entities.items():
            if entity_type in ["company_name", "industry", "location", "size", "materials", "processes"]:
                updated_context["company_data"][entity_type] = value
        
        # Track conversation flow
        if "conversation_history" not in updated_context:
            updated_context["conversation_history"] = []
        
        updated_context["conversation_history"].append({
            "user_input": user_input,
            "intent": intent,
            "entities": entities,
            "timestamp": "2024-01-01T00:00:00Z"  # In production, use actual timestamp
        })
        
        # Update conversation state
        updated_context["current_step"] = self._determine_next_step(updated_context)
        updated_context["conversation_progress"] = self._calculate_conversation_progress(updated_context)
        
        return updated_context

    def _determine_next_step(self, context: Dict) -> str:
        """Determine the next step in the conversation flow"""
        company_data = context.get("company_data", {})
        
        if not company_data.get("company_name"):
            return "get_company_name"
        elif not company_data.get("industry"):
            return "get_industry"
        elif not company_data.get("location"):
            return "get_location"
        elif not company_data.get("materials"):
            return "get_materials"
        elif not company_data.get("processes"):
            return "get_processes"
        else:
            return "complete_onboarding"

    def _calculate_conversation_progress(self, context: Dict) -> float:
        """Calculate conversation progress percentage"""
        company_data = context.get("company_data", {})
        required_fields = ["company_name", "industry", "location", "materials", "processes"]
        
        completed_fields = sum(1 for field in required_fields if company_data.get(field))
        return (completed_fields / len(required_fields)) * 100

    def _get_next_questions(self, context: Dict) -> List[Dict]:
        """Get next questions based on conversation context"""
        current_step = context.get("current_step", "get_company_name")
        company_data = context.get("company_data", {})
        
        next_questions = []
        
        if current_step == "get_company_name":
            next_questions.append({
                "question": "What's your company name?",
                "type": "text",
                "key": "company_name",
                "help_text": "This helps us personalize your experience and track your progress."
            })
        
        elif current_step == "get_industry":
            next_questions.append({
                "question": "What industry are you in?",
                "type": "select",
                "key": "industry",
                "options": ["Textile Manufacturing", "Electronics Manufacturing", "Food Processing", "Chemical Manufacturing", "Other"],
                "help_text": "Your industry helps us understand your typical processes and waste streams."
            })
        
        elif current_step == "get_location":
            next_questions.append({
                "question": "Where is your company located?",
                "type": "text",
                "key": "location",
                "help_text": "Location helps us find nearby partners and understand local regulations."
            })
        
        elif current_step == "get_materials":
            next_questions.append({
                "question": "What are your main production materials?",
                "type": "textarea",
                "key": "materials",
                "help_text": "List materials like cotton, steel, plastic, chemicals, etc. This helps identify recycling opportunities."
            })
        
        elif current_step == "get_processes":
            next_questions.append({
                "question": "What are your main production processes?",
                "type": "textarea",
                "key": "processes",
                "help_text": "Describe processes like cutting, heating, assembly, chemical reactions, etc."
            })
        
        elif current_step == "complete_onboarding":
            next_questions.append({
                "question": "Great! I have enough information to start finding circular economy opportunities for you. Would you like to complete your profile or start exploring matches?",
                "type": "choice",
                "key": "next_action",
                "options": ["Complete Profile", "Start Exploring Matches"],
                "help_text": "You can always come back to add more details later."
            })
        
        return next_questions

    def generate_conversational_onboarding_flow(self, initial_context: Dict = None) -> Dict:
        """Generate a conversational onboarding flow"""
        if not initial_context:
            initial_context = {
                "company_data": {},
                "conversation_history": [],
                "current_step": "get_company_name",
                "conversation_progress": 0
            }
        
        # Get initial questions
        next_questions = self._get_next_questions(initial_context)
        
        return {
            "conversation_context": initial_context,
            "next_questions": next_questions,
            "welcome_message": "Hello! I'm here to help you join our circular economy platform. Let's start by learning about your company. What's your company name?",
            "estimated_time": "10-15 minutes",
            "can_save_progress": True,
            "ai_assisted": True
        }

    def get_conversational_insights(self, conversation_context: Dict) -> Dict:
        """Get insights based on conversational data"""
        company_data = conversation_context.get("company_data", {})
        conversation_history = conversation_context.get("conversation_history", [])
        
        insights = {
            "engagement_level": self._calculate_engagement_level(conversation_history),
            "completion_confidence": self._calculate_completion_confidence(company_data),
            "suggested_next_actions": self._get_suggested_next_actions(conversation_context),
            "potential_concerns": self._identify_potential_concerns(conversation_history),
            "optimization_suggestions": self._get_optimization_suggestions(conversation_context)
        }
        
        return insights

    def _calculate_engagement_level(self, conversation_history: List[Dict]) -> str:
        """Calculate user engagement level"""
        if not conversation_history:
            return "new"
        
        # Analyze conversation patterns
        total_exchanges = len(conversation_history)
        detailed_responses = sum(1 for exchange in conversation_history if len(exchange.get("user_input", "")) > 50)
        
        if total_exchanges < 3:
            return "new"
        elif detailed_responses / total_exchanges > 0.7:
            return "high"
        elif detailed_responses / total_exchanges > 0.3:
            return "medium"
        else:
            return "low"

    def _calculate_completion_confidence(self, company_data: Dict) -> float:
        """Calculate confidence in completion based on data quality"""
        required_fields = ["company_name", "industry", "location", "materials", "processes"]
        completed_fields = sum(1 for field in required_fields if company_data.get(field))
        
        # Base confidence on completion rate
        base_confidence = (completed_fields / len(required_fields)) * 100
        
        # Adjust based on data quality
        quality_score = self.get_data_quality_score(company_data)["percentage"]
        
        return (base_confidence + quality_score) / 2

    def _get_suggested_next_actions(self, context: Dict) -> List[str]:
        """Get suggested next actions based on conversation context"""
        actions = []
        company_data = context.get("company_data", {})
        
        if not company_data.get("materials"):
            actions.append("Ask about production materials to identify waste streams")
        
        if not company_data.get("processes"):
            actions.append("Inquire about production processes to understand operations")
        
        if company_data.get("industry") and company_data.get("materials"):
            actions.append("Provide industry-specific insights and opportunities")
        
        if context.get("conversation_progress", 0) > 80:
            actions.append("Suggest completing the profile to unlock full matching")
        
        return actions

    def _identify_potential_concerns(self, conversation_history: List[Dict]) -> List[str]:
        """Identify potential user concerns from conversation"""
        concerns = []
        
        for exchange in conversation_history:
            user_input = exchange.get("user_input", "").lower()
            intent = exchange.get("intent", "")
            
            if "time" in user_input and "long" in user_input:
                concerns.append("Time commitment concerns")
            
            if "complicated" in user_input or "complex" in user_input:
                concerns.append("Process complexity concerns")
            
            if "data" in user_input and ("security" in user_input or "private" in user_input):
                concerns.append("Data privacy concerns")
            
            if intent == "express_concern":
                concerns.append("General concerns expressed")
        
        return list(set(concerns))  # Remove duplicates

    def _get_optimization_suggestions(self, context: Dict) -> List[str]:
        """Get suggestions for optimizing the conversational flow"""
        suggestions = []
        conversation_history = context.get("conversation_history", [])
        
        if len(conversation_history) > 10:
            suggestions.append("Consider simplifying the flow - conversation is getting long")
        
        # Check for repeated questions
        questions_asked = [exchange.get("intent") for exchange in conversation_history]
        if len(set(questions_asked)) < len(questions_asked) * 0.7:
            suggestions.append("Users are asking similar questions - consider adding help text")
        
        # Check for low engagement
        engagement = self._calculate_engagement_level(conversation_history)
        if engagement == "low":
            suggestions.append("Low engagement detected - consider making questions more interactive")
        
        return suggestions

def generate_green_initiatives(company_data):
    """Generate personalized green initiatives based on company data"""
    try:
        industry = company_data.get('industry', '').lower()
        employee_count = company_data.get('employee_count', 0)
        
        # Base initiatives that apply to all companies
        base_initiatives = [
            {
                "id": str(uuid.uuid4()),
                "category": "Energy Efficiency",
                "question": "Do you have LED lighting installed throughout your facility?",
                "description": "LED lighting can reduce energy consumption by up to 75% compared to traditional lighting.",
                "impact": "high",
                "potential_savings": 15000,
                "carbon_reduction": 25,
                "implementation_time": "2-4 weeks",
                "difficulty": "easy",
                "status": "not_implemented"
            },
            {
                "id": str(uuid.uuid4()),
                "category": "Waste Management",
                "question": "Do you have a comprehensive recycling program for paper, plastic, and metal waste?",
                "description": "Implementing a recycling program can reduce waste disposal costs and generate revenue from recyclables.",
                "impact": "medium",
                "potential_savings": 8000,
                "carbon_reduction": 15,
                "implementation_time": "1-2 months",
                "difficulty": "medium",
                "status": "not_implemented"
            }
        ]
        
        # Industry-specific initiatives
        industry_initiatives = []
        if 'manufacturing' in industry:
            industry_initiatives.extend([
                {
                    "id": str(uuid.uuid4()),
                    "category": "Process Optimization",
                    "question": "Do you use energy-efficient motors and variable speed drives?",
                    "description": "Energy-efficient motors can reduce electricity consumption by 20-30%.",
                    "impact": "high",
                    "potential_savings": 25000,
                    "carbon_reduction": 40,
                    "implementation_time": "3-6 months",
                    "difficulty": "medium",
                    "status": "not_implemented"
                }
            ])
        elif 'textile' in industry:
            industry_initiatives.extend([
                {
                    "id": str(uuid.uuid4()),
                    "category": "Water Conservation",
                    "question": "Do you have a water recycling system for dyeing processes?",
                    "description": "Water recycling can reduce water consumption by 50-70% in textile manufacturing.",
                    "impact": "high",
                    "potential_savings": 30000,
                    "carbon_reduction": 35,
                    "implementation_time": "4-8 months",
                    "difficulty": "hard",
                    "status": "not_implemented"
                }
            ])
        
        # Size-based initiatives
        size_initiatives = []
        if employee_count > 100:
            size_initiatives.append({
                "id": str(uuid.uuid4()),
                "category": "Renewable Energy",
                "question": "Do you have solar panels or other renewable energy sources installed?",
                "description": "Renewable energy can provide long-term cost savings and reduce carbon footprint.",
                "impact": "high",
                "potential_savings": 50000,
                "carbon_reduction": 100,
                "implementation_time": "3-6 months",
                "difficulty": "hard",
                "status": "not_implemented"
            })
        
        all_initiatives = base_initiatives + industry_initiatives + size_initiatives
        
        # Calculate company profile
        company_profile = {
            "total_initiatives": len(all_initiatives),
            "implemented_count": 0,  # Will be calculated from database
            "in_progress_count": 0,  # Will be calculated from database
            "total_savings": 0,  # Will be calculated from database
            "total_carbon_reduction": 0,  # Will be calculated from database
            "sustainability_score": 25,  # Will be calculated from database
            "next_priorities": [
                "Implement LED lighting program",
                "Set up recycling infrastructure",
                "Install energy monitoring systems"
            ]
        }
        
        return {
            "success": True,
            "initiatives": all_initiatives,
            "companyProfile": company_profile
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_portfolio_recommendations(company_data, achievements=None):
    """Generate personalized portfolio recommendations"""
    try:
        industry = company_data.get('industry', '').lower()
        employee_count = company_data.get('employee_count', 0)
        location = company_data.get('location', '')
        
        # Calculate company overview
        if employee_count < 50:
            size_category = "Small Enterprise"
        elif employee_count < 200:
            size_category = "Medium Enterprise"
        else:
            size_category = "Large Enterprise"
        
        # Generate recommendations based on company data
        recommendations = []
        
        if 'manufacturing' in industry:
            recommendations.append({
                "id": str(uuid.uuid4()),
                "category": "Process Optimization",
                "title": "Implement Lean Manufacturing",
                "description": "Reduce waste and improve efficiency through lean manufacturing principles",
                "potential_impact": {
                    "savings": 25000,
                    "carbon_reduction": 30,
                    "efficiency_gain": 20
                },
                "implementation_difficulty": "medium",
                "time_to_implement": "3-6 months",
                "priority": "high",
                "ai_reasoning": f"Based on {company_data.get('name', 'your company')}'s size and industry, lean manufacturing could significantly improve operational efficiency."
            })
        
        if employee_count > 50:
            recommendations.append({
                "id": str(uuid.uuid4()),
                "category": "Supply Chain",
                "title": "Local Supplier Network",
                "description": "Develop partnerships with local suppliers to reduce transportation costs and carbon footprint",
                "potential_impact": {
                    "savings": 15000,
                    "carbon_reduction": 25,
                    "efficiency_gain": 10
                },
                "implementation_difficulty": "easy",
                "time_to_implement": "2-4 months",
                "priority": "medium",
                "ai_reasoning": f"Your location in {location} shows several potential local suppliers that could reduce logistics costs by 15-20%."
            })
        
        company_overview = {
            "size_category": size_category,
            "industry_position": "Established",
            "sustainability_rating": "B+",
            "growth_potential": "High"
        }
        
        return {
            "success": True,
            "company_overview": company_overview,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_insights(company_data):
    """Generate AI insights for the company"""
    try:
        industry = company_data.get('industry', '').lower()
        employee_count = company_data.get('employee_count', 0)
        
        insights = []
        
        # Generate industry-specific insights
        if 'manufacturing' in industry:
            insights.append({
                "id": str(uuid.uuid4()),
                "type": "opportunity",
                "title": "New Partnership Opportunity",
                "description": "Local textile manufacturer needs your waste materials",
                "impact": "high",
                "estimated_savings": 15000,
                "carbon_reduction": 25,
                "action_required": True,
                "priority": "high"
            })
        
        if employee_count > 100:
            insights.append({
                "id": str(uuid.uuid4()),
                "type": "suggestion",
                "title": "Process Optimization",
                "description": "Implement lean manufacturing to reduce waste by 30%",
                "impact": "medium",
                "estimated_savings": 8000,
                "carbon_reduction": 15,
                "action_required": True,
                "priority": "medium"
            })
        
        return {
            "success": True,
            "insights": insights
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Onboarding Engine')
    parser.add_argument('--action', required=True, help='Action to perform')
    parser.add_argument('--data', help='Company data for onboarding')
    parser.add_argument('--company_data', help='Company data for other actions')
    parser.add_argument('--achievements', help='Achievements data')
    
    args = parser.parse_args()
    
    if args.action == 'generate_listings':
        if not args.data:
            print(json.dumps({"success": False, "error": "No data provided"}))
            sys.exit(1)
        
        company_data = json.loads(args.data)
        result = generate_listings(company_data)
        print(json.dumps(result))
        
    elif args.action == 'generate_green_initiatives':
        if not args.company_data:
            print(json.dumps({"success": False, "error": "No company data provided"}))
            sys.exit(1)
        
        company_data = json.loads(args.company_data)
        result = generate_green_initiatives(company_data)
        print(json.dumps(result))
        
    elif args.action == 'generate_portfolio_recommendations':
        if not args.company_data:
            print(json.dumps({"success": False, "error": "No company data provided"}))
            sys.exit(1)
        
        company_data = json.loads(args.company_data)
        achievements = json.loads(args.achievements) if args.achievements else None
        result = generate_portfolio_recommendations(company_data, achievements)
        print(json.dumps(result))
        
    elif args.action == 'generate_insights':
        if not args.company_data:
            print(json.dumps({"success": False, "error": "No company data provided"}))
            sys.exit(1)
        
        company_data = json.loads(args.company_data)
        result = generate_insights(company_data)
        print(json.dumps(result))
        
    else:
        print(json.dumps({"success": False, "error": f"Unknown action: {args.action}"}))
        sys.exit(1)