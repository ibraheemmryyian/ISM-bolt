#!/usr/bin/env python3
"""
Demo Environment Setup Script
Prepares the entire system for video capture demo
"""

import asyncio
import json
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from demo_data_import_service import DemoDataImportService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoEnvironmentSetup:
    """
    Comprehensive demo environment setup for video capture
    """
    
    def __init__(self, data_file: str = None, force_setup: bool = False):
        self.data_file = data_file
        self.force_setup = force_setup
        self.demo_service = DemoDataImportService()
        self.setup_results = {}
        
    async def setup_complete_demo_environment(self):
        """
        Setup the complete demo environment for video capture
        """
        logger.info("🚀 Starting complete demo environment setup...")
        
        steps = [
            ("Validating demo data file", self._validate_demo_data),
            ("Importing and enhancing company data", self._import_company_data),
            ("Generating realistic material listings", self._generate_materials),
            ("Creating demo matches", self._create_matches),
            ("Setting up demo user accounts", self._setup_demo_accounts),
            ("Optimizing onboarding flow", self._optimize_onboarding),
            ("Preparing marketplace data", self._prepare_marketplace),
            ("Validating demo flow", self._validate_demo_flow),
            ("Creating demo summary", self._create_demo_summary)
        ]
        
        for i, (step_name, step_func) in enumerate(steps, 1):
            logger.info(f"📋 Step {i}/{len(steps)}: {step_name}")
            try:
                result = await step_func()
                self.setup_results[step_name] = {"success": True, "result": result}
                logger.info(f"✅ {step_name} completed successfully")
            except Exception as e:
                logger.error(f"❌ {step_name} failed: {e}")
                self.setup_results[step_name] = {"success": False, "error": str(e)}
                if step_name in ["Importing and enhancing company data"]:
                    # Critical step - abort if failed
                    raise Exception(f"Critical step failed: {step_name}")
        
        # Generate final demo report
        await self._generate_demo_report()
        
        logger.info("🎬 Demo environment setup completed! Ready for video capture.")
        return self.setup_results

    async def _validate_demo_data(self):
        """Validate the demo data file exists and is properly formatted"""
        if not self.data_file:
            # Look for common data file locations
            possible_files = [
                "data/company_data.json",
                "data/real_company_data.json", 
                "data/gulf_companies.json",
                "../data/company_data.json",
                "company_data.json"
            ]
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    self.data_file = file_path
                    break
            
            if not self.data_file:
                # Create sample data file
                sample_data = self._create_sample_company_data()
                self.data_file = "data/demo_company_data.json"
                os.makedirs("data", exist_ok=True)
                with open(self.data_file, 'w') as f:
                    json.dump(sample_data, f, indent=2)
                logger.info(f"Created sample data file: {self.data_file}")
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Demo data file not found: {self.data_file}")
        
        # Validate file format
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            companies = data
        elif isinstance(data, dict) and 'companies' in data:
            companies = data['companies']
        else:
            raise ValueError("Invalid data format - expected list of companies or object with 'companies' key")
        
        logger.info(f"Validated {len(companies)} companies in {self.data_file}")
        return {"file_path": self.data_file, "company_count": len(companies)}

    def _create_sample_company_data(self):
        """Create sample company data for demo if no real data provided"""
        sample_companies = [
            {
                "name": "Gulf Manufacturing Co.",
                "industry": "Manufacturing",
                "location": "Dubai, UAE",
                "employee_count": 250,
                "waste_streams": ["metal shavings", "plastic offcuts", "packaging waste"],
                "description": "Leading manufacturing company specializing in industrial components"
            },
            {
                "name": "Arabian Textiles Ltd.",
                "industry": "Textiles",
                "location": "Abu Dhabi, UAE", 
                "employee_count": 180,
                "waste_streams": ["fabric scraps", "yarn waste", "dyeing chemicals"],
                "description": "Premium textile manufacturer with sustainable practices"
            },
            {
                "name": "Emirates Food Processing",
                "industry": "Food & Beverage",
                "location": "Sharjah, UAE",
                "employee_count": 120,
                "waste_streams": ["organic waste", "packaging materials", "expired products"],
                "description": "Food processing company serving the Gulf region"
            },
            {
                "name": "Qatar Chemical Industries",
                "industry": "Chemicals",
                "location": "Doha, Qatar",
                "employee_count": 300,
                "waste_streams": ["chemical byproducts", "contaminated materials", "solvents"],
                "description": "Chemical processing and manufacturing facility"
            },
            {
                "name": "Saudi Construction Group",
                "industry": "Construction",
                "location": "Riyadh, Saudi Arabia",
                "employee_count": 450,
                "waste_streams": ["concrete waste", "metal scraps", "wood waste"],
                "description": "Major construction company with infrastructure projects"
            }
        ]
        
        return sample_companies

    async def _import_company_data(self):
        """Import and enhance company data using the demo service"""
        result = await self.demo_service.import_real_company_data(
            self.data_file, 
            demo_mode=True
        )
        
        logger.info(f"Imported {result['companies_imported']} companies")
        return result

    async def _generate_materials(self):
        """Generate realistic material listings"""
        # Materials are generated as part of the import process
        material_count = len(self.demo_service.generated_materials)
        logger.info(f"Generated {material_count} material listings")
        
        return {
            "total_materials": material_count,
            "waste_materials": len([m for m in self.demo_service.generated_materials if m['type'] == 'waste']),
            "requirement_materials": len([m for m in self.demo_service.generated_materials if m['type'] == 'requirement'])
        }

    async def _create_matches(self):
        """Create demo matches between materials"""
        # Matches are created as part of the import process
        match_count = len(self.demo_service.created_matches)
        logger.info(f"Created {match_count} demo matches")
        
        return {
            "total_matches": match_count,
            "average_match_score": sum(m['match_score'] for m in self.demo_service.created_matches) / max(1, match_count),
            "total_potential_savings": sum(m['potential_savings'] for m in self.demo_service.created_matches)
        }

    async def _setup_demo_accounts(self):
        """Setup demo user accounts for testing"""
        demo_accounts = [
            {
                "email": "demo.user@symbioflows.com",
                "name": "Demo User",
                "company": "Demo Company",
                "role": "sustainability_manager"
            },
            {
                "email": "test.company@symbioflows.com", 
                "name": "Test User",
                "company": "Test Industries",
                "role": "operations_manager"
            }
        ]
        
        logger.info("Demo accounts configuration ready")
        return {"demo_accounts": demo_accounts}

    async def _optimize_onboarding(self):
        """Optimize the onboarding flow for demo capture"""
        
        # Create optimized onboarding configuration
        onboarding_config = {
            "demo_mode": True,
            "fast_ai_responses": True,
            "skip_email_verification": True,
            "auto_generate_portfolio": True,
            "demo_data_prefill": {
                "industry": "Manufacturing",
                "products": "Industrial components and machinery parts",
                "production_volume": "500 tonnes per month",
                "processes": "CNC machining, assembly, quality control, packaging"
            }
        }
        
        # Write onboarding config for frontend to use
        config_path = "frontend/src/config/demo-config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(onboarding_config, f, indent=2)
        
        logger.info("Onboarding flow optimized for demo")
        return onboarding_config

    async def _prepare_marketplace(self):
        """Prepare marketplace with demo data"""
        
        # Save demo materials to a file the frontend can access
        marketplace_data = {
            "materials": self.demo_service.generated_materials,
            "matches": self.demo_service.created_matches,
            "companies": [
                {
                    "id": company.get("id", f"demo_{company.get('demo_index')}"),
                    "name": company["name"],
                    "industry": company["industry"],
                    "location": company["location"],
                    "sustainability_score": company["sustainability_score"],
                    "employee_count": company["employee_count"]
                }
                for company in self.demo_service.imported_companies[:10]
            ]
        }
        
        # Write marketplace data
        marketplace_path = "frontend/src/data/demo-marketplace.json"
        os.makedirs(os.path.dirname(marketplace_path), exist_ok=True)
        with open(marketplace_path, 'w') as f:
            json.dump(marketplace_data, f, indent=2)
        
        logger.info("Marketplace prepared with demo data")
        return {
            "materials_count": len(marketplace_data["materials"]),
            "matches_count": len(marketplace_data["matches"]),
            "companies_count": len(marketplace_data["companies"])
        }

    async def _validate_demo_flow(self):
        """Validate the complete demo flow"""
        
        validation_checks = {
            "data_import": len(self.demo_service.imported_companies) > 0,
            "material_generation": len(self.demo_service.generated_materials) > 0,
            "match_creation": len(self.demo_service.created_matches) > 0,
            "onboarding_config": os.path.exists("frontend/src/config/demo-config.json"),
            "marketplace_data": os.path.exists("frontend/src/data/demo-marketplace.json")
        }
        
        all_checks_passed = all(validation_checks.values())
        
        if not all_checks_passed:
            failed_checks = [check for check, passed in validation_checks.items() if not passed]
            raise Exception(f"Demo validation failed: {failed_checks}")
        
        logger.info("All demo flow validations passed")
        return validation_checks

    async def _create_demo_summary(self):
        """Create final demo summary"""
        summary = self.demo_service.get_demo_summary()
        
        # Add additional demo info
        summary.update({
            "setup_timestamp": datetime.now().isoformat(),
            "data_source": self.data_file,
            "video_capture_ready": True,
            "demo_flow_steps": [
                "1. Navigate to homepage (localhost:3000)",
                "2. Click 'Get Started' to create account",
                "3. Complete sign-up with demo credentials", 
                "4. Go through AI onboarding (/onboarding)",
                "5. View generated portfolio (/dashboard)",
                "6. Browse marketplace (/marketplace)",
                "7. View material matches and messaging"
            ],
            "recommended_demo_script": {
                "account_creation": "30 seconds - Quick sign-up with realistic company info",
                "ai_onboarding": "90 seconds - Show AI generating portfolio from company description",
                "material_listings": "60 seconds - Browse through intelligent material listings",
                "matching_system": "90 seconds - Demonstrate AI matching and potential savings",
                "total_demo_time": "4-5 minutes"
            }
        })
        
        return summary

    async def _generate_demo_report(self):
        """Generate comprehensive demo report"""
        
        report = {
            "demo_setup_complete": True,
            "timestamp": datetime.now().isoformat(),
            "setup_results": self.setup_results,
            "system_ready": all(result.get("success", False) for result in self.setup_results.values()),
            "demo_statistics": self.demo_service.get_demo_summary(),
            "next_steps": [
                "Start frontend development server (npm run dev)",
                "Start backend services if needed",
                "Navigate to localhost:3000 for demo",
                "Begin video capture"
            ]
        }
        
        # Save report
        report_path = "demo_setup_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Demo setup report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎬 DEMO ENVIRONMENT SETUP COMPLETE!")
        print("="*60)
        print(f"📊 Companies Imported: {report['demo_statistics']['statistics']['companies_imported']}")
        print(f"📦 Materials Generated: {report['demo_statistics']['statistics']['materials_generated']}")
        print(f"🔗 Matches Created: {report['demo_statistics']['statistics']['matches_created']}")
        print(f"✅ System Ready: {'Yes' if report['system_ready'] else 'No'}")
        print("\n📋 Demo Flow:")
        for step in report['demo_statistics']['demo_flow_steps']:
            print(f"   {step}")
        print("\n" + "="*60)

async def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup demo environment for video capture")
    parser.add_argument("--data-file", type=str, help="Path to company data JSON file")
    parser.add_argument("--force", action="store_true", help="Force setup even if already configured")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = DemoEnvironmentSetup(
        data_file=args.data_file,
        force_setup=args.force
    )
    
    try:
        await setup.setup_complete_demo_environment()
        print("\n🚀 Ready for video capture! Start your frontend and begin demo.")
        return 0
    except Exception as e:
        logger.error(f"Demo setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())