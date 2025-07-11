#!/usr/bin/env python3
"""
Complete ISM AI Pipeline Runner
Imports 50 companies, generates AI listings, creates matches, and runs symbiosis analysis
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class CompletePipelineRunner:
    def __init__(self):
        self.backend_url = "http://localhost:5001"
        self.companies_data = []
        self.imported_companies = []
        self.generated_listings = []
        self.created_matches = []
        self.symbiosis_networks = []
        
    async def load_company_data(self):
        """Load or generate 50 companies data"""
        print("📦 Step 1: Loading company data...")
        
        # Try to load from existing file first
        try:
            with open("data/50_real_gulf_companies_cleaned.json", 'r', encoding='utf-8') as f:
                self.companies_data = json.load(f)
                print(f"✅ Loaded {len(self.companies_data)} companies from file")
                return True
        except:
            print("⚠️ No existing company file found, generating fake data...")
            self.companies_data = self.generate_fake_companies(50)
            print(f"✅ Generated {len(self.companies_data)} fake companies")
            return True
    
    def generate_fake_companies(self, count=50):
        """Generate realistic fake company data"""
        industries = [
            "Plastics Manufacturing", "Chemical Processing", "Steel Production", 
            "Textile Manufacturing", "Food Processing", "Electronics Manufacturing",
            "Automotive Parts", "Construction Materials", "Pharmaceuticals",
            "Petrochemicals", "Mining", "Paper Manufacturing", "Glass Manufacturing",
            "Cement Production", "Aluminum Processing", "Copper Refining",
            "Waste Management", "Recycling", "Energy Production", "Water Treatment"
        ]
        
        locations = [
            "Dubai, UAE", "Abu Dhabi, UAE", "Riyadh, Saudi Arabia", "Jeddah, Saudi Arabia",
            "Doha, Qatar", "Kuwait City, Kuwait", "Muscat, Oman", "Manama, Bahrain",
            "Sharjah, UAE", "Al Khobar, Saudi Arabia", "Dammam, Saudi Arabia"
        ]
        
        companies = []
        
        for i in range(1, count + 1):
            company = {
                "name": f"Gulf Industrial Company {i:03d}",
                "industry": industries[i % len(industries)],
                "location": locations[i % len(locations)],
                "email": f"company{i:03d}@example.com",
                "employee_count": 50 + (i * 20) % 2000,
                "sustainability_score": 30 + (i * 2) % 65,
                "carbon_footprint": 500 + (i * 50) % 5000,
                "water_usage": 100 + (i * 20) % 2000,
                "subscription_tier": "pro",
                "subscription_status": "active",
                "role": "user",
                "products": f"Product A{i}, Product B{i}, Product C{i}",
                "main_materials": f"Material X{i}, Material Y{i}, Material Z{i}",
                "process_description": f"Manufacturing process for {industries[i % len(industries)]}",
                "waste_quantity": 100 + (i * 30) % 5000,
                "waste_unit": "kg",
                "waste_frequency": "monthly",
                "onboarding_completed": True
            }
            companies.append(company)
        
        return companies
    
    async def import_companies(self):
        """Import all companies into the system"""
        print(f"\n🏢 Step 2: Importing {len(self.companies_data)} companies...")
        
        async with aiohttp.ClientSession() as session:
            for i, company in enumerate(self.companies_data, 1):
                try:
                    async with session.post(
                        f"{self.backend_url}/api/companies",
                        json=company,
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            company_id = result.get("id")
                            if company_id:
                                self.imported_companies.append({
                                    "id": company_id,
                                    "data": company
                                })
                                print(f"✅ Imported {i}/{len(self.companies_data)}: {company['name']}")
                            else:
                                print(f"⚠️ No ID for {company['name']}")
                        else:
                            print(f"❌ Failed to import {company['name']}: {response.status}")
                            
                except Exception as e:
                    print(f"❌ Error importing {company['name']}: {e}")
        
        print(f"✅ Successfully imported {len(self.imported_companies)} companies")
        return len(self.imported_companies) > 0
    
    async def generate_ai_listings(self):
        """Generate AI listings for all companies"""
        print(f"\n🤖 Step 3: Generating AI listings for {len(self.imported_companies)} companies...")
        
        async with aiohttp.ClientSession() as session:
            for i, company in enumerate(self.imported_companies, 1):
                try:
                    company_id = company["id"]
                    company_data = company["data"]
                    
                    # Generate waste listings
                    waste_request = {
                        "company_id": company_id,
                        "type": "waste",
                        "industry": company_data["industry"],
                        "location": company_data["location"],
                        "employee_count": company_data["employee_count"],
                        "sustainability_score": company_data["sustainability_score"]
                    }
                    
                    async with session.post(
                        f"{self.backend_url}/api/ai/listings/generate",
                        json=waste_request,
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            waste_result = await response.json()
                            self.generated_listings.extend(waste_result.get("listings", []))
                            print(f"✅ Generated waste listings for {company_data['name']}")
                        
                    # Generate requirement listings
                    req_request = {
                        "company_id": company_id,
                        "type": "requirement",
                        "industry": company_data["industry"],
                        "location": company_data["location"],
                        "employee_count": company_data["employee_count"],
                        "sustainability_score": company_data["sustainability_score"]
                    }
                    
                    async with session.post(
                        f"{self.backend_url}/api/ai/listings/generate",
                        json=req_request,
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            req_result = await response.json()
                            self.generated_listings.extend(req_result.get("listings", []))
                            print(f"✅ Generated requirement listings for {company_data['name']}")
                            
                except Exception as e:
                    print(f"❌ Error generating listings for {company['data']['name']}: {e}")
        
        print(f"✅ Generated {len(self.generated_listings)} AI listings")
        return len(self.generated_listings) > 0
    
    async def run_ai_matching(self):
        """Run AI matching between companies"""
        print(f"\n🔗 Step 4: Running AI matching for {len(self.imported_companies)} companies...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_url}/api/ai/matching/run",
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.created_matches = result.get("matches", [])
                        print(f"✅ AI matching completed. Created {len(self.created_matches)} matches")
                        return True
                    else:
                        print(f"❌ AI matching failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Error in AI matching: {e}")
            return False
    
    async def run_symbiosis_analysis(self):
        """Run multi-hop symbiosis analysis"""
        print(f"\n🌐 Step 5: Running symbiosis network analysis...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get symbiosis opportunities
                async with session.get(
                    f"{self.backend_url}/api/ai/symbiosis/opportunities",
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.symbiosis_networks = result.get("networks", [])
                        print(f"✅ Symbiosis analysis completed. Found {len(self.symbiosis_networks)} networks")
                        return True
                    else:
                        print(f"⚠️ Symbiosis analysis failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"⚠️ Error in symbiosis analysis: {e}")
            return False
    
    async def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n📊 Step 6: Generating summary report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_summary": {
                "total_companies": len(self.imported_companies),
                "total_listings": len(self.generated_listings),
                "total_matches": len(self.created_matches),
                "total_networks": len(self.symbiosis_networks)
            },
            "companies_by_industry": {},
            "listings_by_type": {
                "waste": 0,
                "requirement": 0
            },
            "matches_by_strength": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "environmental_impact": {
                "total_carbon_reduction": 0,
                "total_waste_diverted": 0,
                "total_economic_value": 0
            }
        }
        
        # Analyze companies by industry
        for company in self.imported_companies:
            industry = company["data"]["industry"]
            report["companies_by_industry"][industry] = report["companies_by_industry"].get(industry, 0) + 1
        
        # Analyze listings by type
        for listing in self.generated_listings:
            listing_type = listing.get("type", "unknown")
            if listing_type in report["listings_by_type"]:
                report["listings_by_type"][listing_type] += 1
        
        # Save report
        with open("pipeline_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("✅ Summary report generated: pipeline_report.json")
        return report
    
    async def run_complete_pipeline(self):
        """Run the complete AI pipeline"""
        print("🚀 ISM AI - COMPLETE PIPELINE RUNNER")
        print("=" * 60)
        print("This will process 50 companies through the full AI pipeline:")
        print("1. Import companies")
        print("2. Generate AI listings")
        print("3. Create AI matches")
        print("4. Run symbiosis analysis")
        print("5. Generate summary report")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Load company data
            if not await self.load_company_data():
                return False
            
            # Step 2: Import companies
            if not await self.import_companies():
                return False
            
            # Step 3: Generate AI listings
            if not await self.generate_ai_listings():
                return False
            
            # Step 4: Run AI matching
            if not await self.run_ai_matching():
                return False
            
            # Step 5: Run symbiosis analysis
            await self.run_symbiosis_analysis()
            
            # Step 6: Generate report
            report = await self.generate_summary_report()
            
            # Final summary
            end_time = time.time()
            duration = end_time - start_time
            
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"⏱️ Total time: {duration:.2f} seconds")
            print(f"📊 Results:")
            print(f"   • Companies processed: {report['pipeline_summary']['total_companies']}")
            print(f"   • AI listings generated: {report['pipeline_summary']['total_listings']}")
            print(f"   • AI matches created: {report['pipeline_summary']['total_matches']}")
            print(f"   • Symbiosis networks: {report['pipeline_summary']['total_networks']}")
            print(f"\n🌐 Check your system:")
            print(f"   • Admin Dashboard: http://localhost:5001/api/admin/stats")
            print(f"   • Frontend: http://localhost:5173")
            print(f"   • Report: pipeline_report.json")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return False

async def main():
    """Main function"""
    runner = CompletePipelineRunner()
    success = await runner.run_complete_pipeline()
    exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main()) 