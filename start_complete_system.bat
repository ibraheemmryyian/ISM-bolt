@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 SYMBIOFLOWS - COMPLETE SYSTEM STARTUP
echo ========================================
echo 🎯 Starting ALL Advanced AI Microservices with Full Validation
echo ========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

REM Suppress warnings globally
set PYTHONWARNINGS=ignore
set PYTHONPATH=%cd%;%PYTHONPATH%

echo %BLUE%🔍 Pre-flight validation...%RESET%
echo.

REM ========================================
REM PREREQUISITE CHECKS
REM ========================================

echo %YELLOW%Checking prerequisites...%RESET%

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Node.js is not installed or not in PATH%RESET%
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Python is not installed or not in PATH%RESET%
    pause
    exit /b 1
)

REM Check npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ npm is not installed or not in PATH%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Prerequisites OK%RESET%
echo.

REM ========================================
REM COMPREHENSIVE SERVICE VALIDATION
REM ========================================

echo %YELLOW%Validating all microservice files...%RESET%

REM Check backend services (25+ services)
set BACKEND_SERVICES=0
if exist "backend\adaptive_onboarding_server.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_listings_generator.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_monitoring_dashboard.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_pricing_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_pricing_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_matchmaking_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\meta_learning_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\materials_bert_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\materials_bert_service_simple.py" set /a BACKEND_SERVICES+=1
if exist "backend\ultra_ai_listings_generator.py" set /a BACKEND_SERVICES+=1
if exist "backend\regulatory_compliance.py" set /a BACKEND_SERVICES+=1
if exist "backend\proactive_opportunity_engine.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_feedback_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_production_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_retraining_pipeline.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_production_ai_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_production_ai_system_fixed.py" set /a BACKEND_SERVICES+=1
if exist "backend\system_health_monitor.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_enhanced_materials_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_service_integration.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_hyperparameter_optimizer.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_fusion_layer.py" set /a BACKEND_SERVICES+=1
if exist "backend\value_function_arbiter.py" set /a BACKEND_SERVICES+=1
if exist "backend\advanced_ai_integration.py" set /a BACKEND_SERVICES+=1
if exist "backend\advanced_ai_prompts_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\advanced_description_generator.py" set /a BACKEND_SERVICES+=1
if exist "backend\advanced_listings_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\advanced_orchestration_engine.py" set /a BACKEND_SERVICES+=1
if exist "backend\advanced_quality_assessment_engine.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_material_analysis_engine.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_pricing_integration.py" set /a BACKEND_SERVICES+=1
if exist "backend\buyer_seller_differentiation_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\complete_logistics_platform.py" set /a BACKEND_SERVICES+=1
if exist "backend\complete_system_integration.py" set /a BACKEND_SERVICES+=1
if exist "backend\deepseek_r1_semantic_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\demo_no_hardcoded_data.py" set /a BACKEND_SERVICES+=1
if exist "backend\demo_ultra_advanced_ai.py" set /a BACKEND_SERVICES+=1
if exist "backend\distributed_tracing.py" set /a BACKEND_SERVICES+=1
if exist "backend\dynamic_materials_integration_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\enhanced_materials_integration_demo.py" set /a BACKEND_SERVICES+=1
if exist "backend\enhanced_materials_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\error_recovery_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\event_driven_architecture.py" set /a BACKEND_SERVICES+=1
if exist "backend\federated_meta_learning.py" set /a BACKEND_SERVICES+=1
if exist "backend\generate_ai_listings_deepseek.py" set /a BACKEND_SERVICES+=1
if exist "backend\revolutionary_ai_matching.py" set /a BACKEND_SERVICES+=1
if exist "backend\run_complete_pipeline.py" set /a BACKEND_SERVICES+=1
if exist "backend\run_complete_setup.py" set /a BACKEND_SERVICES+=1
if exist "backend\run_system_tests.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_bulletproof_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_core_services.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_fixed_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_materials_bert_simple.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_optimized_ai_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_pipeline.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_adaptive_fix.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_ai_features.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_deepseek_integration.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_materials_quality.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_now.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_perfect_flow.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_python_ai.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_real_data_complete.py" set /a BACKEND_SERVICES+=1
if exist "backend\test_startup.py" set /a BACKEND_SERVICES+=1
if exist "backend\validate_all_services.py" set /a BACKEND_SERVICES+=1

REM Check AI Service Flask services (8 services)
set AI_FLASK_SERVICES=0
if exist "ai_service_flask\ai_gateway.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\advanced_analytics_service.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\ai_pricing_service_wrapper.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\gnn_inference_service.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\logistics_service_wrapper.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\multi_hop_symbiosis_service.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\federated_learning_service.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\find_duplicates.py" set /a AI_FLASK_SERVICES+=1

REM Check root services (15+ services)
set ROOT_SERVICES=0
if exist "logistics_cost_service.py" set /a ROOT_SERVICES+=1
if exist "optimize_deepseek_r1.py" set /a ROOT_SERVICES+=1
if exist "financial_analysis_engine.py" set /a ROOT_SERVICES+=1
if exist "advanced_ai_quality_analysis.py" set /a ROOT_SERVICES+=1
if exist "advanced_analytics_engine.py" set /a ROOT_SERVICES+=1
if exist "ai_listings_generator.py" set /a ROOT_SERVICES+=1
if exist "ai_onboarding_questions_generator.py" set /a ROOT_SERVICES+=1
if exist "ai_quality_monitor.py" set /a ROOT_SERVICES+=1
if exist "bulletproof_ai_generator.py" set /a ROOT_SERVICES+=1
if exist "carbon_calculation_engine.py" set /a ROOT_SERVICES+=1
if exist "check_api_response.py" set /a ROOT_SERVICES+=1
if exist "check_database_state.py" set /a ROOT_SERVICES+=1
if exist "check_materials_table.py" set /a ROOT_SERVICES+=1
if exist "comprehensive_match_analyzer.py" set /a ROOT_SERVICES+=1
if exist "conversational_b2b_agent.py" set /a ROOT_SERVICES+=1
if exist "deepseek_quality_analysis.py" set /a ROOT_SERVICES+=1
if exist "enhanced_ai_generator.py" set /a ROOT_SERVICES+=1
if exist "find_duplicates.py" set /a ROOT_SERVICES+=1
if exist "fix_ai_generator.py" set /a ROOT_SERVICES+=1
if exist "generate_combined_report.py" set /a ROOT_SERVICES+=1
if exist "generate_marketplace_listings.py" set /a ROOT_SERVICES+=1
if exist "gulf_benchmark_selection_fixed.py" set /a ROOT_SERVICES+=1
if exist "gulf_benchmark_selection.py" set /a ROOT_SERVICES+=1
if exist "list_all_endpoints.py" set /a ROOT_SERVICES+=1
if exist "merge_deduplicate.py" set /a ROOT_SERVICES+=1
if exist "multi_hop_symbiosis_network.py" set /a ROOT_SERVICES+=1
if exist "populate_marketplace.py" set /a ROOT_SERVICES+=1
if exist "real_ai_matching_engine.py" set /a ROOT_SERVICES+=1
if exist "refinement_analysis_engine.py" set /a ROOT_SERVICES+=1
if exist "run_complete_pipeline.py" set /a ROOT_SERVICES+=1
if exist "run_system_tests.py" set /a ROOT_SERVICES+=1
if exist "waste_tracking_engine.py" set /a ROOT_SERVICES+=1

echo %GREEN%✅ Found %BACKEND_SERVICES% backend services%RESET%
echo %GREEN%✅ Found %AI_FLASK_SERVICES% AI Flask services%RESET%
echo %GREEN%✅ Found %ROOT_SERVICES% root services%RESET%

set TOTAL_SERVICES=%BACKEND_SERVICES%
set /a TOTAL_SERVICES+=%AI_FLASK_SERVICES%
set /a TOTAL_SERVICES+=%ROOT_SERVICES%

echo %BLUE%🎯 Total microservices found: %TOTAL_SERVICES%%RESET%
echo.

REM ========================================
REM CORE SERVICES (ALWAYS START)
REM ========================================

echo %BLUE%📦 Starting Core Services...%RESET%

REM 1. Backend (Node.js) on port 3001
echo %YELLOW%1. Starting Node.js Backend (port 3001)...%RESET%
start "Backend API" cmd /k "cd backend && npm run dev"

REM 2. Frontend (React) on port 5173
echo %YELLOW%2. Starting React Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 8 /nobreak >nul

REM ========================================
REM CRITICAL AI SERVICES (PRIORITY 1)
REM ========================================

echo %BLUE%🤖 Starting Critical AI Services...%RESET%

REM 3. AI Gateway (Python) on port 5000
echo %YELLOW%3. Starting AI Gateway (port 5000)...%RESET%
start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"

REM 4. Advanced Analytics Service (Python) on port 5004
echo %YELLOW%4. Starting Advanced Analytics Service (port 5004)...%RESET%
start "Advanced Analytics" cmd /k "cd ai_service_flask && python advanced_analytics_service.py"

REM 5. GNN Inference Service (Python) on port 5001
echo %YELLOW%5. Starting GNN Inference Service (port 5001)...%RESET%
start "GNN Inference" cmd /k "cd ai_service_flask && python gnn_inference_service.py"

REM 6. Multi-Hop Symbiosis Service (Python) on port 5003
echo %YELLOW%6. Starting Multi-Hop Symbiosis Service (port 5003)...%RESET%
start "Multi-Hop Symbiosis" cmd /k "cd ai_service_flask && python multi_hop_symbiosis_service.py"

REM 7. Federated Learning Service (Python) on port 5002
echo %YELLOW%7. Starting Federated Learning Service (port 5002)...%RESET%
start "Federated Learning" cmd /k "cd ai_service_flask && python federated_learning_service.py"

timeout /t 10 /nobreak >nul

REM ========================================
REM BACKEND MICROSERVICES (PRIORITY 2)
REM ========================================

echo %BLUE%🔧 Starting Backend Microservices...%RESET%

REM 8. Adaptive Onboarding Server (Python) on port 5006
echo %YELLOW%8. Starting Adaptive Onboarding Server (port 5006)...%RESET%
start "Adaptive Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

REM 9. System Health Monitor (Python) on port 5018
echo %YELLOW%9. Starting System Health Monitor (port 5018)...%RESET%
start "System Health Monitor" cmd /k "cd backend && python system_health_monitor.py"

REM 10. AI Monitoring Dashboard (Python) on port 5007
echo %YELLOW%10. Starting AI Monitoring Dashboard (port 5007)...%RESET%
start "AI Monitoring Dashboard" cmd /k "cd backend && python ai_monitoring_dashboard.py"

REM 11. AI Pricing Service (Python) on port 5008
echo %YELLOW%11. Starting AI Pricing Service (port 5008)...%RESET%
start "AI Pricing Service" cmd /k "cd backend && python ai_pricing_service.py"

REM 12. AI Pricing Orchestrator (Python) on port 8030
echo %YELLOW%12. Starting AI Pricing Orchestrator (port 8030)...%RESET%
start "AI Pricing Orchestrator" cmd /k "cd backend && python ai_pricing_orchestrator.py"

REM 13. Meta-Learning Orchestrator (Python) on port 8010
echo %YELLOW%13. Starting Meta-Learning Orchestrator (port 8010)...%RESET%
start "Meta-Learning Orchestrator" cmd /k "cd backend && python meta_learning_orchestrator.py"

REM 14. AI Matchmaking Service (Python) on port 8020
echo %YELLOW%14. Starting AI Matchmaking Service (port 8020)...%RESET%
start "AI Matchmaking Service" cmd /k "cd backend && python ai_matchmaking_service.py"

REM 15. MaterialsBERT Service (Python) on port 5009
echo %YELLOW%15. Starting MaterialsBERT Service (port 5009)...%RESET%
start "MaterialsBERT Service" cmd /k "cd backend && python materials_bert_service.py"

REM 16. MaterialsBERT Simple Service (Python) on port 5010
echo %YELLOW%16. Starting MaterialsBERT Simple Service (port 5010)...%RESET%
start "MaterialsBERT Simple" cmd /k "cd backend && python materials_bert_service_simple.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM ADVANCED AI SERVICES (PRIORITY 3)
REM ========================================

echo %BLUE%🧠 Starting Advanced AI Services...%RESET%

REM 17. Ultra AI Listings Generator (Python) on port 5011
echo %YELLOW%17. Starting Ultra AI Listings Generator (port 5011)...%RESET%
start "Ultra AI Listings Generator" cmd /k "cd backend && python ultra_ai_listings_generator.py"

REM 18. AI Listings Generator (Python) on port 5012
echo %YELLOW%18. Starting AI Listings Generator (port 5012)...%RESET%
start "AI Listings Generator" cmd /k "cd backend && python ai_listings_generator.py"

REM 19. Enhanced AI Generator (Python) on port 5013
echo %YELLOW%19. Starting Enhanced AI Generator (port 5013)...%RESET%
start "Enhanced AI Generator" cmd /k "python enhanced_ai_generator.py"

REM 20. Bulletproof AI Generator (Python) on port 5014
echo %YELLOW%20. Starting Bulletproof AI Generator (port 5014)...%RESET%
start "Bulletproof AI Generator" cmd /k "python bulletproof_ai_generator.py"

REM 21. Real AI Matching Engine (Python) on port 5015
echo %YELLOW%21. Starting Real AI Matching Engine (port 5015)...%RESET%
start "Real AI Matching Engine" cmd /k "python real_ai_matching_engine.py"

REM 22. Revolutionary AI Matching (Python) on port 5016
echo %YELLOW%22. Starting Revolutionary AI Matching (port 5016)...%RESET%
start "Revolutionary AI Matching" cmd /k "python revolutionary_ai_matching.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM SPECIALIZED SERVICES (PRIORITY 4)
REM ========================================

echo %BLUE%🔬 Starting Specialized Services...%RESET%

REM 23. Regulatory Compliance (Python) on port 5017
echo %YELLOW%23. Starting Regulatory Compliance Service (port 5017)...%RESET%
start "Regulatory Compliance" cmd /k "cd backend && python regulatory_compliance.py"

REM 24. Proactive Opportunity Engine (Python) on port 5019
echo %YELLOW%24. Starting Proactive Opportunity Engine (port 5019)...%RESET%
start "Proactive Opportunity Engine" cmd /k "cd backend && python proactive_opportunity_engine.py"

REM 25. AI Feedback Orchestrator (Python) on port 5020
echo %YELLOW%25. Starting AI Feedback Orchestrator (port 5020)...%RESET%
start "AI Feedback Orchestrator" cmd /k "cd backend && python ai_feedback_orchestrator.py"

REM 26. AI Production Orchestrator (Python) on port 5021
echo %YELLOW%26. Starting AI Production Orchestrator (port 5021)...%RESET%
start "AI Production Orchestrator" cmd /k "cd backend && python ai_production_orchestrator.py"

REM 27. AI Retraining Pipeline (Python) on port 5022
echo %YELLOW%27. Starting AI Retraining Pipeline (port 5022)...%RESET%
start "AI Retraining Pipeline" cmd /k "cd backend && python ai_retraining_pipeline.py"

REM 28. AI Service Integration (Python) on port 5023
echo %YELLOW%28. Starting AI Service Integration (port 5023)...%RESET%
start "AI Service Integration" cmd /k "cd backend && python ai_service_integration.py"

REM 29. AI Hyperparameter Optimizer (Python) on port 5024
echo %YELLOW%29. Starting AI Hyperparameter Optimizer (port 5024)...%RESET%
start "AI Hyperparameter Optimizer" cmd /k "cd backend && python ai_hyperparameter_optimizer.py"

REM 30. AI Fusion Layer (Python) on port 5025
echo %YELLOW%30. Starting AI Fusion Layer (port 5025)...%RESET%
start "AI Fusion Layer" cmd /k "cd backend && python ai_fusion_layer.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM ANALYTICS & MONITORING (PRIORITY 5)
REM ========================================

echo %BLUE%📊 Starting Analytics & Monitoring Services...%RESET%

REM 31. Advanced Analytics Engine (Python) on port 5026
echo %YELLOW%31. Starting Advanced Analytics Engine (port 5026)...%RESET%
start "Advanced Analytics Engine" cmd /k "python advanced_analytics_engine.py"

REM 32. AI Quality Monitor (Python) on port 5027
echo %YELLOW%32. Starting AI Quality Monitor (port 5027)...%RESET%
start "AI Quality Monitor" cmd /k "python ai_quality_monitor.py"

REM 33. Advanced AI Quality Analysis (Python) on port 5028
echo %YELLOW%33. Starting Advanced AI Quality Analysis (port 5028)...%RESET%
start "Advanced AI Quality Analysis" cmd /k "python advanced_ai_quality_analysis.py"

REM 34. DeepSeek Quality Analysis (Python) on port 5029
echo %YELLOW%34. Starting DeepSeek Quality Analysis (port 5029)...%RESET%
start "DeepSeek Quality Analysis" cmd /k "python deepseek_quality_analysis.py"

REM 35. Comprehensive Match Analyzer (Python) on port 5030
echo %YELLOW%35. Starting Comprehensive Match Analyzer (port 5030)...%RESET%
start "Comprehensive Match Analyzer" cmd /k "python comprehensive_match_analyzer.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM LOGISTICS & FINANCIAL (PRIORITY 6)
REM ========================================

echo %BLUE%🚚 Starting Logistics & Financial Services...%RESET%

REM 36. Logistics Cost Service (Python) on port 5031
echo %YELLOW%36. Starting Logistics Cost Service (port 5031)...%RESET%
start "Logistics Cost Service" cmd /k "python logistics_cost_service.py"

REM 37. Financial Analysis Engine (Python) on port 5032
echo %YELLOW%37. Starting Financial Analysis Engine (port 5032)...%RESET%
start "Financial Analysis Engine" cmd /k "python financial_analysis_engine.py"

REM 38. Carbon Calculation Engine (Python) on port 5033
echo %YELLOW%38. Starting Carbon Calculation Engine (port 5033)...%RESET%
start "Carbon Calculation Engine" cmd /k "python carbon_calculation_engine.py"

REM 39. Waste Tracking Engine (Python) on port 5034
echo %YELLOW%39. Starting Waste Tracking Engine (port 5034)...%RESET%
start "Waste Tracking Engine" cmd /k "python waste_tracking_engine.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM INTEGRATION & UTILITY SERVICES (PRIORITY 7)
REM ========================================

echo %BLUE%🔗 Starting Integration & Utility Services...%RESET%

REM 40. Conversational B2B Agent (Python) on port 5035
echo %YELLOW%40. Starting Conversational B2B Agent (port 5035)...%RESET%
start "Conversational B2B Agent" cmd /k "python conversational_b2b_agent.py"

REM 41. Multi-Hop Symbiosis Network (Python) on port 5036
echo %YELLOW%41. Starting Multi-Hop Symbiosis Network (port 5036)...%RESET%
start "Multi-Hop Symbiosis Network" cmd /k "python multi_hop_symbiosis_network.py"

REM 42. Value Function Arbiter (Python) on port 5037
echo %YELLOW%42. Starting Value Function Arbiter (port 5037)...%RESET%
start "Value Function Arbiter" cmd /k "cd backend && python value_function_arbiter.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM HEALTH CHECK AND VALIDATION
REM ========================================

echo.
echo %BLUE%🔍 Running comprehensive health checks...%RESET%

REM Test core services
echo %YELLOW%Testing core services...%RESET%
cd backend
python validate_all_services.py
cd ..

echo.
echo ========================================
echo 🎉 COMPLETE SYSTEM STARTED!
echo ========================================
echo.
echo 📊 SYSTEM SUMMARY:
echo ✅ Backend API: http://localhost:3001
echo ✅ Frontend: http://localhost:5173
echo ✅ AI Gateway: http://localhost:5000
echo ✅ Advanced Analytics: http://localhost:5004
echo ✅ GNN Inference: http://localhost:5001
echo ✅ Multi-Hop Symbiosis: http://localhost:5003
echo ✅ Federated Learning: http://localhost:5002
echo ✅ Adaptive Onboarding: http://localhost:5006
echo ✅ System Health Monitor: http://localhost:5018
echo ✅ AI Monitoring Dashboard: http://localhost:5007
echo ✅ AI Pricing Service: http://localhost:5008
echo ✅ AI Pricing Orchestrator: http://localhost:8030
echo ✅ Meta-Learning Orchestrator: http://localhost:8010
echo ✅ AI Matchmaking Service: http://localhost:8020
echo ✅ MaterialsBERT Service: http://localhost:5009
echo ✅ MaterialsBERT Simple: http://localhost:5010
echo ✅ Ultra AI Listings Generator: http://localhost:5011
echo ✅ AI Listings Generator: http://localhost:5012
echo ✅ Enhanced AI Generator: http://localhost:5013
echo ✅ Bulletproof AI Generator: http://localhost:5014
echo ✅ Real AI Matching Engine: http://localhost:5015
echo ✅ Revolutionary AI Matching: http://localhost:5016
echo ✅ Regulatory Compliance: http://localhost:5017
echo ✅ Proactive Opportunity Engine: http://localhost:5019
echo ✅ AI Feedback Orchestrator: http://localhost:5020
echo ✅ AI Production Orchestrator: http://localhost:5021
echo ✅ AI Retraining Pipeline: http://localhost:5022
echo ✅ AI Service Integration: http://localhost:5023
echo ✅ AI Hyperparameter Optimizer: http://localhost:5024
echo ✅ AI Fusion Layer: http://localhost:5025
echo ✅ Advanced Analytics Engine: http://localhost:5026
echo ✅ AI Quality Monitor: http://localhost:5027
echo ✅ Advanced AI Quality Analysis: http://localhost:5028
echo ✅ DeepSeek Quality Analysis: http://localhost:5029
echo ✅ Comprehensive Match Analyzer: http://localhost:5030
echo ✅ Logistics Cost Service: http://localhost:5031
echo ✅ Financial Analysis Engine: http://localhost:5032
echo ✅ Carbon Calculation Engine: http://localhost:5033
echo ✅ Waste Tracking Engine: http://localhost:5034
echo ✅ Conversational B2B Agent: http://localhost:5035
echo ✅ Multi-Hop Symbiosis Network: http://localhost:5036
echo ✅ Value Function Arbiter: http://localhost:5037
echo.
echo 🎯 TOTAL MICROSERVICES: %TOTAL_SERVICES%
echo 🚀 ALL SERVICES STARTED IN SEPARATE WINDOWS
echo.
echo 💡 Each service window shows real-time logs and errors
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped. 