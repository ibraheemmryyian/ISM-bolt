#!/usr/bin/env python3
"""
Demo Video Preparation Script

This script prepares the system for a demo video by:
1. Setting up the database with sample company data
2. Configuring the AI onboarding system to handle waste streams
3. Importing real company data with waste streams
4. Ensuring the material listings and matches are properly displayed
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import required modules
try:
    from backend.adaptive_ai_onboarding import AdaptiveAIOnboarding
    from backend.listing_inference_service import ListingInferenceService
    HAS_BACKEND_MODULES = True
except ImportError:
    logger.warning("Backend modules not found, will use direct database access instead")
    HAS_BACKEND_MODULES = False

# Try to import database modules
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import supabase
    HAS_DATABASE_MODULES = True
except ImportError:
    logger.warning("Database modules not found, will use file-based approach")
    HAS_DATABASE_MODULES = False

class DemoPreparation:
    """Prepares the system for a demo video"""
    
    def __init__(self, config_path=None):
        """Initialize with optional config path"""
        self.config = self._load_config(config_path)
        self.company_data = None
        self.materials_data = None
        self.matches_data = None
        
    def _load_config(self, config_path):
        """Load configuration from file or use defaults"""
        default_config = {
            "database": {
                "host": os.environ.get("DB_HOST", "localhost"),
                "port": os.environ.get("DB_PORT", "5432"),
                "name": os.environ.get("DB_NAME", "postgres"),
                "user": os.environ.get("DB_USER", "postgres"),
                "password": os.environ.get("DB_PASSWORD", "postgres")
            },
            "supabase": {
                "url": os.environ.get("SUPABASE_URL", ""),
                "key": os.environ.get("SUPABASE_KEY", "")
            },
            "sample_data": {
                "companies_csv": "data/sample_companies.csv",
                "waste_streams_csv": "data/sample_waste_streams.csv"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            return default_config
    
    def load_company_data(self, file_path=None):
        """Load company data from CSV file"""
        if not file_path:
            file_path = self.config["sample_data"]["companies_csv"]
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} not found, creating sample data")
                self._create_sample_company_data(file_path)
            
            self.company_data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.company_data)} companies from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading company data: {e}")
            return False
    
    def load_waste_streams(self, file_path=None):
        """Load waste stream data from CSV file"""
        if not file_path:
            file_path = self.config["sample_data"]["waste_streams_csv"]
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} not found, creating sample data")
                self._create_sample_waste_streams(file_path)
            
            self.materials_data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.materials_data)} waste streams from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading waste stream data: {e}")
            return False
    
    def _create_sample_company_data(self, output_path):
        """Create sample company data if none exists"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create sample data
        companies = []
        industries = ["Manufacturing", "Food & Beverage", "Chemicals", "Textiles", 
                      "Construction", "Electronics", "Automotive", "Pharmaceuticals"]
        locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", 
                     "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA"]
        
        for i in range(1, 21):
            company = {
                "id": f"company_{i}",
                "name": f"Demo Company {i}",
                "industry": random.choice(industries),
                "location": random.choice(locations),
                "employee_count": random.randint(10, 1000),
                "description": f"A sample company for demo purposes #{i}",
                "website": f"https://demo-company-{i}.example.com",
                "contact_email": f"contact@demo-company-{i}.example.com",
                "contact_phone": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "sustainability_goals": random.choice(["Carbon reduction", "Waste minimization", 
                                                      "Energy efficiency", "Water conservation", "None yet"])
            }
            companies.append(company)
        
        # Save to CSV
        pd.DataFrame(companies).to_csv(output_path, index=False)
        logger.info(f"Created sample company data at {output_path}")
    
    def _create_sample_waste_streams(self, output_path):
        """Create sample waste stream data if none exists"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create sample data
        waste_streams = []
        waste_types = {
            "Manufacturing": ["Metal scrap", "Plastic waste", "Wood waste", "Packaging materials", 
                             "Chemical waste", "Oil and lubricants"],
            "Food & Beverage": ["Organic waste", "Food scraps", "Packaging waste", "Wastewater", 
                               "Used cooking oil", "Expired products"],
            "Chemicals": ["Chemical byproducts", "Solvents", "Acids", "Alkalines", 
                         "Laboratory waste", "Packaging materials"],
            "Textiles": ["Fabric scraps", "Yarn waste", "Dye residues", "Packaging materials", 
                        "Thread waste", "Cutting waste"],
            "Construction": ["Concrete debris", "Wood scraps", "Metal waste", "Drywall", 
                            "Insulation materials", "Paint and coatings"],
            "Electronics": ["Electronic components", "Circuit boards", "Cables", "Packaging materials", 
                           "Metals", "Plastics"],
            "Automotive": ["Metal scraps", "Oil and lubricants", "Rubber waste", "Plastic components", 
                          "Glass", "Packaging materials"],
            "Pharmaceuticals": ["Expired medications", "Chemical waste", "Packaging materials", 
                               "Laboratory waste", "Organic compounds", "Filter materials"]
        }
        
        if not self.company_data is not None:
            self.load_company_data()
        
        for _, company in self.company_data.iterrows():
            industry = company["industry"]
            company_id = company["id"]
            
            # Get waste types for this industry
            industry_wastes = waste_types.get(industry, waste_types["Manufacturing"])
            
            # Create 2-5 waste streams per company
            for _ in range(random.randint(2, 5)):
                waste_type = random.choice(industry_wastes)
                waste = {
                    "company_id": company_id,
                    "waste_stream": waste_type,
                    "description": f"{waste_type} generated during production process",
                    "frequency": random.choice(["Daily", "Weekly", "Monthly", "Quarterly"]),
                    "handling_method": random.choice(["Recycling", "Landfill", "Incineration", "Unknown"]),
                    "hazardous": random.choice([True, False]),
                    "estimated_volume": f"{random.randint(1, 1000)} {random.choice(['kg', 'tons', 'liters'])}"
                }
                waste_streams.append(waste)
        
        # Save to CSV
        pd.DataFrame(waste_streams).to_csv(output_path, index=False)
        logger.info(f"Created sample waste stream data at {output_path}")
    
    def connect_to_database(self):
        """Connect to the database"""
        if not HAS_DATABASE_MODULES:
            logger.error("Database modules not available")
            return False
        
        try:
            # Try PostgreSQL connection
            conn_string = f"host={self.config['database']['host']} " \
                          f"port={self.config['database']['port']} " \
                          f"dbname={self.config['database']['name']} " \
                          f"user={self.config['database']['user']} " \
                          f"password={self.config['database']['password']}"
            
            self.conn = psycopg2.connect(conn_string)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Connected to PostgreSQL database")
            
            # Try Supabase connection if credentials available
            if self.config['supabase']['url'] and self.config['supabase']['key']:
                self.supabase_client = supabase.create_client(
                    self.config['supabase']['url'],
                    self.config['supabase']['key']
                )
                logger.info("Connected to Supabase")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def import_companies_to_database(self):
        """Import company data to database"""
        if not HAS_DATABASE_MODULES:
            logger.error("Database modules not available")
            return False
        
        if self.company_data is None:
            logger.error("No company data loaded")
            return False
        
        try:
            # Check if we have a database connection
            if not hasattr(self, 'conn') or self.conn.closed:
                if not self.connect_to_database():
                    return False
            
            # Import companies
            for _, company in self.company_data.iterrows():
                # Check if company exists
                self.cursor.execute(
                    "SELECT id FROM companies WHERE id = %s",
                    (company['id'],)
                )
                if self.cursor.fetchone():
                    # Update existing company
                    self.cursor.execute(
                        """
                        UPDATE companies 
                        SET name = %s, industry = %s, location = %s, 
                            employee_count = %s, description = %s,
                            website = %s, contact_email = %s, contact_phone = %s,
                            sustainability_goals = %s, updated_at = %s
                        WHERE id = %s
                        """,
                        (
                            company['name'], company['industry'], company['location'],
                            company['employee_count'], company['description'],
                            company.get('website', ''), company.get('contact_email', ''),
                            company.get('contact_phone', ''), company.get('sustainability_goals', ''),
                            datetime.now(), company['id']
                        )
                    )
                else:
                    # Insert new company
                    self.cursor.execute(
                        """
                        INSERT INTO companies 
                        (id, name, industry, location, employee_count, description,
                         website, contact_email, contact_phone, sustainability_goals, 
                         created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            company['id'], company['name'], company['industry'],
                            company['location'], company['employee_count'], company['description'],
                            company.get('website', ''), company.get('contact_email', ''),
                            company.get('contact_phone', ''), company.get('sustainability_goals', ''),
                            datetime.now(), datetime.now()
                        )
                    )
            
            self.conn.commit()
            logger.info(f"Imported {len(self.company_data)} companies to database")
            return True
        except Exception as e:
            logger.error(f"Error importing companies to database: {e}")
            if hasattr(self, 'conn'):
                self.conn.rollback()
            return False
    
    def import_waste_streams_to_database(self):
        """Import waste stream data to database"""
        if not HAS_DATABASE_MODULES:
            logger.error("Database modules not available")
            return False
        
        if self.materials_data is None:
            logger.error("No waste stream data loaded")
            return False
        
        try:
            # Check if we have a database connection
            if not hasattr(self, 'conn') or self.conn.closed:
                if not self.connect_to_database():
                    return False
            
            # Import waste streams as materials
            for _, waste in self.materials_data.iterrows():
                # Parse quantity and unit from estimated_volume
                parts = waste['estimated_volume'].split(' ')
                quantity = parts[0] if len(parts) > 0 else "0"
                unit = parts[1] if len(parts) > 1 else "units"
                
                # Generate a unique ID for the material
                material_id = f"material_{waste['company_id']}_{random.randint(1000, 9999)}"
                
                # Check if material exists (by company_id and waste_stream)
                self.cursor.execute(
                    """
                    SELECT id FROM materials 
                    WHERE company_id = %s AND name = %s
                    """,
                    (waste['company_id'], waste['waste_stream'])
                )
                if self.cursor.fetchone():
                    # Update existing material
                    self.cursor.execute(
                        """
                        UPDATE materials 
                        SET description = %s, quantity = %s, unit = %s,
                            type = 'waste', category = %s, 
                            hazardous = %s, frequency = %s,
                            handling_method = %s, updated_at = %s
                        WHERE company_id = %s AND name = %s
                        """,
                        (
                            waste['description'], quantity, unit,
                            waste['waste_stream'].split(' ')[0] if ' ' in waste['waste_stream'] else 'General',
                            waste['hazardous'], waste['frequency'],
                            waste['handling_method'], datetime.now(),
                            waste['company_id'], waste['waste_stream']
                        )
                    )
                else:
                    # Insert new material
                    self.cursor.execute(
                        """
                        INSERT INTO materials 
                        (id, company_id, name, description, quantity, unit,
                         type, category, hazardous, frequency, handling_method,
                         ai_generated, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            material_id, waste['company_id'], waste['waste_stream'],
                            waste['description'], quantity, unit,
                            'waste', waste['waste_stream'].split(' ')[0] if ' ' in waste['waste_stream'] else 'General',
                            waste['hazardous'], waste['frequency'],
                            waste['handling_method'], True, datetime.now(), datetime.now()
                        )
                    )
            
            self.conn.commit()
            logger.info(f"Imported {len(self.materials_data)} waste streams to database")
            return True
        except Exception as e:
            logger.error(f"Error importing waste streams to database: {e}")
            if hasattr(self, 'conn'):
                self.conn.rollback()
            return False
    
    def generate_matches(self):
        """Generate potential matches between materials"""
        if not HAS_DATABASE_MODULES:
            logger.error("Database modules not available")
            return False
        
        try:
            # Check if we have a database connection
            if not hasattr(self, 'conn') or self.conn.closed:
                if not self.connect_to_database():
                    return False
            
            # Get all materials
            self.cursor.execute(
                """
                SELECT id, company_id, name, type, category, quantity, unit
                FROM materials
                """
            )
            materials = self.cursor.fetchall()
            
            # Group by category
            materials_by_category = {}
            for material in materials:
                category = material['category']
                if category not in materials_by_category:
                    materials_by_category[category] = []
                materials_by_category[category].append(material)
            
            # Generate matches within same category
            matches = []
            for category, category_materials in materials_by_category.items():
                # Find waste materials
                waste_materials = [m for m in category_materials if m['type'] == 'waste']
                # Find requirement materials
                req_materials = [m for m in category_materials if m['type'] == 'requirement']
                
                # If we don't have requirements, create some synthetic ones
                if not req_materials and waste_materials:
                    # Create synthetic requirements for other companies
                    companies = [m['company_id'] for m in waste_materials]
                    for waste in waste_materials:
                        # Find a company that doesn't have this waste
                        other_companies = [c for c in companies if c != waste['company_id']]
                        if other_companies:
                            req_company = random.choice(other_companies)
                            req_id = f"material_req_{req_company}_{random.randint(1000, 9999)}"
                            
                            # Insert synthetic requirement
                            self.cursor.execute(
                                """
                                INSERT INTO materials 
                                (id, company_id, name, description, quantity, unit,
                                 type, category, ai_generated, created_at, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    req_id, req_company, waste['name'],
                                    f"Required {waste['name']} for production",
                                    waste['quantity'], waste['unit'],
                                    'requirement', waste['category'], True, 
                                    datetime.now(), datetime.now()
                                )
                            )
                            
                            # Add to req_materials
                            req_materials.append({
                                'id': req_id,
                                'company_id': req_company,
                                'name': waste['name'],
                                'type': 'requirement',
                                'category': waste['category'],
                                'quantity': waste['quantity'],
                                'unit': waste['unit']
                            })
                
                # Match waste to requirements
                for waste in waste_materials:
                    for req in req_materials:
                        # Don't match within same company
                        if waste['company_id'] == req['company_id']:
                            continue
                        
                        # Calculate match score (0.7-1.0)
                        match_score = 0.7 + random.random() * 0.3
                        
                        # Create match
                        match_id = f"match_{waste['id']}_{req['id']}"
                        
                        # Check if match exists
                        self.cursor.execute(
                            """
                            SELECT id FROM matches 
                            WHERE material_id = %s AND matched_material_id = %s
                            """,
                            (waste['id'], req['id'])
                        )
                        if self.cursor.fetchone():
                            # Update existing match
                            self.cursor.execute(
                                """
                                UPDATE matches 
                                SET match_score = %s, updated_at = %s
                                WHERE material_id = %s AND matched_material_id = %s
                                """,
                                (match_score, datetime.now(), waste['id'], req['id'])
                            )
                        else:
                            # Insert new match
                            self.cursor.execute(
                                """
                                INSERT INTO matches 
                                (id, material_id, matched_material_id, match_score,
                                 ai_generated, created_at, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    match_id, waste['id'], req['id'], match_score,
                                    True, datetime.now(), datetime.now()
                                )
                            )
                        
                        matches.append({
                            'id': match_id,
                            'material_id': waste['id'],
                            'matched_material_id': req['id'],
                            'match_score': match_score
                        })
            
            self.conn.commit()
            logger.info(f"Generated {len(matches)} matches")
            self.matches_data = pd.DataFrame(matches)
            return True
        except Exception as e:
            logger.error(f"Error generating matches: {e}")
            if hasattr(self, 'conn'):
                self.conn.rollback()
            return False
    
    def setup_demo_account(self, email=None, password=None):
        """Set up a demo account for the video"""
        if email is None:
            email = f"demo_{int(datetime.now().timestamp())}@example.com"
        if password is None:
            password = "demo123456"
        
        try:
            # If we have Supabase client, use it
            if hasattr(self, 'supabase_client'):
                # Create user in Supabase
                response = self.supabase_client.auth.sign_up({
                    "email": email,
                    "password": password
                })
                user_id = response.user.id
                logger.info(f"Created demo user in Supabase: {email}")
            else:
                # Otherwise, just generate a user ID
                user_id = f"user_{int(datetime.now().timestamp())}"
                logger.info(f"Generated demo user ID: {user_id}")
            
            # If we have database connection, create company
            if hasattr(self, 'conn') and not self.conn.closed:
                # Create company for user
                company_name = "Demo Video Company"
                
                self.cursor.execute(
                    """
                    INSERT INTO companies 
                    (id, name, created_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, company_name, datetime.now(), datetime.now())
                )
                self.conn.commit()
                logger.info(f"Created demo company: {company_name}")
            
            return {
                "user_id": user_id,
                "email": email,
                "password": password
            }
        except Exception as e:
            logger.error(f"Error setting up demo account: {e}")
            return None
    
    def prepare_demo(self):
        """Run the full demo preparation process"""
        logger.info("Starting demo preparation")
        
        # Load data
        self.load_company_data()
        self.load_waste_streams()
        
        # Connect to database
        if HAS_DATABASE_MODULES:
            self.connect_to_database()
            self.import_companies_to_database()
            self.import_waste_streams_to_database()
            self.generate_matches()
        
        # Set up demo account
        demo_account = self.setup_demo_account()
        
        logger.info("Demo preparation complete")
        return demo_account

if __name__ == "__main__":
    # Check for config file path in arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run demo preparation
    demo = DemoPreparation(config_path)
    demo_account = demo.prepare_demo()
    
    if demo_account:
        print("\n" + "=" * 50)
        print("DEMO ACCOUNT CREATED")
        print("=" * 50)
        print(f"Email: {demo_account['email']}")
        print(f"Password: {demo_account['password']}")
        print(f"User ID: {demo_account['user_id']}")
        print("=" * 50)
        print("\nUse these credentials to log in for your demo video.")
        print("The system has been prepared with sample data and matches.")