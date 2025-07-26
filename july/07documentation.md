# ISM [AI] – Full Code Audit & Project Documentation

---

## 1. **Project Structure Overview**

### **Root Directories**
- `backend/` – Node.js and Python backend services, endpoints, business logic, models, and tests.
- `ai_service_flask/` – Python Flask microservices (AI, analytics, federated learning, GNN, etc.).
- `frontend/` – React frontend, UI components, API service layers, and configuration.
- `scripts/` – Setup, deployment, and test automation scripts.
- `supabase/` – Database migrations and schema.
- `model_storage/`, `monitoring/`, `infrastructure/`, etc. – Supporting infrastructure.

---

## 2. **Backend: Endpoints, Call Points, and File Map**

### **A. Node.js Backend (`backend/app.js` and submodules)**

#### **API Endpoints (from `app.js` and related files)**
| Endpoint                        | Method | Handler File/Function                        | Features/Notes                        |
|----------------------------------|--------|----------------------------------------------|---------------------------------------|
| `/api/health`                   | GET    | `backend/app.js`                             | Health check                          |
| `/api/ai-infer-listings`        | POST   | `backend/ai_listings_generator.py`           | AI listing generation                 |
| `/api/match`                    | POST   | `backend/ai_matchmaking_service.py`          | AI recommendations/matching           |
| `/api/feedback`                 | POST   | `backend/ai_feedback_orchestrator.py`        | User feedback                         |
| `/api/ai-pipeline`              | POST   | `backend/advanced_ai_integration.py`         | AI pipeline orchestration             |
| `/api/ai-chat`                  | POST   | `backend/conversational_b2b_agent.py`        | AI chat                               |
| `/api/real-time-recommendations`| POST   | `backend/revolutionary_ai_matching.py`       | Real-time recommendations             |
| `/api/explain-match`            | POST   | `backend/advanced_ai_prompts_service.py`     | Explainable AI                        |
| `/api/plugins`                  | GET    | `backend/ai_service_integration.py`          | Plugin list                           |
| `/api/cost-breakdown`           | POST   | `backend/logistics_cost_engine.js`           | Cost breakdown                        |
| `/api/intelligent-matching`     | POST   | `backend/services/intelligentMatchingService.js` | Intelligent matching             |
| `/api/ai-analysis`              | POST   | `backend/ai_monitoring_dashboard.py`         | AI evolution/analysis                 |
| `/api/freightos/rates`          | POST   | `backend/services/freightosService.js`       | Freightos integration                 |
| `/api/real-data/process-company`| POST   | `backend/real_data_bulk_importer.py`         | Real data processing                  |
| `/api/ai/generate-listings/:companyId` | POST | `backend/ai_listings_generator.py`      | Company-specific listing generation    |
| `/api/ai/gnn/create-graph`      | POST   | `backend/gnn_reasoning_engine.py`            | GNN graph creation                    |
| `/api/ai/revolutionary/match`   | POST   | `backend/revolutionary_ai_matching.py`       | Revolutionary matching                |
| `/api/ai/services/status`       | GET    | `backend/system_health_monitor.py`           | AI service status                     |
| `/api/adaptive-onboarding/start`| POST   | `backend/adaptive_ai_onboarding.py`          | Adaptive onboarding (Python/Flask)    |
| `/api/adaptive-onboarding/respond`| POST | `backend/adaptive_ai_onboarding.py`          | Adaptive onboarding (Python/Flask)    |
| `/api/adaptive-onboarding/complete`| POST| `backend/adaptive_ai_onboarding.py`          | Adaptive onboarding (Python/Flask)    |
| `/api/enhanced-matching`        | POST   | `backend/enhanced_materials_integration_demo.py` | Enhanced matching                |
| `/api/ai/gnn-predictions`       | GET    | `backend/gnn_reasoning_engine.py`            | GNN predictions                       |
| `/api/ai/multi-hop-paths`       | GET    | `backend/multi_hop_symbiosis_network.py`     | Multi-hop paths                       |
| `/api/ai/insights`              | GET    | `backend/advanced_analytics_engine.py`       | AI insights                            |
| `/api/ai/symbiosis-network`     | GET    | `backend/multi_hop_symbiosis_network.py`     | Symbiosis network                     |

#### **Call Chain Example: `/api/ai-infer-listings`**
- **Entry:** `backend/app.js` (Express route or Flask route)
- **Handler:** `backend/ai_listings_generator.py` (`infer_listings()` function)
- **Logic:** Calls AI model, possibly via `ai_service_flask/advanced_analytics_service.py`
- **Database:** Reads/writes via `backend/models/`
- **Returns:** JSON response to frontend

#### **Test Coverage**
- **Test scripts:**  
  - `test_complete_system.ps1` → `backend/tests/setup.js` → calls `/api/ai-infer-listings` and others.
  - `test_ai_features.bat`, `test_materials_quality.js`, etc. call endpoints directly.

#### **Supporting Files**
- **Config:** `backend/config/`
- **Utils:** `backend/utils/`
- **Models:** `backend/models/`
- **Migrations:** `supabase/migrations/`, `backend/*.sql`

---

### **B. Python Flask Microservices (`ai_service_flask/`)**

#### **Endpoints**
| Endpoint (Path)         | Method | File/Function                                 | Feature/Notes                |
|-------------------------|--------|-----------------------------------------------|------------------------------|
| `/materials`            | GET    | `ai_service_flask/advanced_analytics_service.py` | MaterialsBERT service    |
| `/analyze`              | POST   | `ai_service_flask/advanced_analytics_service.py` | MaterialsBERT analysis   |
| `/properties`           | GET    | `ai_service_flask/advanced_analytics_service.py` | MaterialsBERT properties |
| `/gnn/<endpoint>`       | POST   | `ai_service_flask/gnn_inference_service.py`      | GNN Gateway              |
| `/federated/<endpoint>` | POST   | `ai_service_flask/federated_learning_service.py` | Federated learning        |
| `/symbiosis/<endpoint>` | POST   | `ai_service_flask/multi_hop_symbiosis_service.py`| Symbiosis analysis        |
| `/analytics/<endpoint>` | POST   | `ai_service_flask/advanced_analytics_service.py` | Analytics                 |
| `/orchestrate`          | POST   | `ai_service_flask/ai_gateway.py`                 | Orchestration             |

#### **Call Chain Example: `/analyze`**
- **Entry:** `ai_service_flask/advanced_analytics_service.py` (Flask route)
- **Handler:** `analyze()` function
- **Logic:** Calls AI model, returns analysis

---

## 3. **Frontend: Components, API Calls, and File Map**

### **A. UI Components (`frontend/src/components/`)**

| Component File                        | Feature/Screen                        | API Service Used (file)         | Endpoints Called                        |
|---------------------------------------|---------------------------------------|---------------------------------|-----------------------------------------|
| `AIOnboardingWizard.tsx`              | AI Onboarding Wizard                  | `aiService.ts`                  | `/api/ai-infer-listings`, `/api/match`  |
| `AIInferenceMatching.tsx`             | AI Matching UI                        | `aiService.ts`                  | `/api/match`                            |
| `ComprehensiveMatchAnalysis.tsx`      | Match Analysis                        | `aiService.ts`                  | `/api/explain-match`                    |
| `ChatInterface.tsx`                   | AI Chat                               | `aiService.ts`                  | `/api/ai-chat`                          |
| `EnhancedMatchingInterface.tsx`       | Enhanced Matching                     | `aiService.ts`                  | `/api/enhanced-matching`                |
| `AdminAccessPage.tsx`                 | Admin Access                          | `adminAccess.ts`                | `/api/plugins`                          |
| `PaymentProcessor.tsx`                | Payment Processing                    | `paymentService.ts`             | `/api/cost-breakdown`                   |
| `Dashboard.tsx`                       | Main Dashboard                        | `aiPreviewService.ts`           | `/api/ai-pipeline`                      |
| ...                                   | ...                                   | ...                             | ...                                     |

### **B. API Service Layer (`frontend/src/lib/`)**

| Service File            | Function Name         | Endpoint Called                        | Used By (Component)                  |
|------------------------|----------------------|----------------------------------------|--------------------------------------|
| `aiService.ts`         | `inferListings()`    | `/api/ai-infer-listings`               | `AIOnboardingWizard.tsx`             |
|                        | `match()`           | `/api/match`                           | `AIInferenceMatching.tsx`            |
|                        | `explainMatch()`    | `/api/explain-match`                   | `ComprehensiveMatchAnalysis.tsx`     |
|                        | `aiChat()`          | `/api/ai-chat`                         | `ChatInterface.tsx`                  |
|                        | ...                 | ...                                    | ...                                  |
| `aiPreviewService.ts`  | `runPipeline()`     | `/api/ai-pipeline`                     | `Dashboard.tsx`                      |
| `adminAccess.ts`       | `getPlugins()`      | `/api/plugins`                         | `AdminAccessPage.tsx`                |
| `paymentService.ts`    | `getCostBreakdown()`| `/api/cost-breakdown`                  | `PaymentProcessor.tsx`               |
| ...                    | ...                 | ...                                    | ...                                  |

---

## 4. **End-to-End Feature Flow Example**

### **AI Onboarding Wizard (Full Call Chain)**

1. **User Action:**  
   - Clicks "Generate Listings" in `AIOnboardingWizard.tsx`
2. **Frontend Service Call:**  
   - Calls `inferListings()` in `aiService.ts`
3. **API Request:**  
   - `POST /api/ai-infer-listings`
4. **Backend Entry:**  
   - `backend/app.js` (Express/Flask route)
5. **Backend Handler:**  
   - `backend/ai_listings_generator.py` (`infer_listings()` function)
6. **AI Logic:**  
   - Calls AI model, processes data
7. **Response:**  
   - Returns listings to frontend
8. **Frontend UI Update:**  
   - `AIOnboardingWizard.tsx` updates state/UI

---

## 5. **Test and Setup Scripts**

| Script File                        | Purpose/Entry Point                    | Calls/Tests                       |
|------------------------------------|----------------------------------------|-----------------------------------|
| `test_complete_system.ps1`         | Full system test (PowerShell)          | Calls backend endpoints           |
| `backend/tests/setup.js`           | Node.js test setup                     | Calls `/api/ai-infer-listings`, etc. |
| `test_ai_features.bat`             | Batch test for AI features             | Calls backend endpoints           |
| `test_materials_quality.js`        | JS test for materials quality          | Calls `/analyze`, `/materials`    |
| ...                                | ...                                    | ...                               |

---

## 6. **Redundant/Orphaned/Legacy Files**

- **Orphaned:**  
  - Any file not referenced in any endpoint, import, or test (e.g., old scripts in `backend/`, unused components in `frontend/`).
- **Legacy:**  
  - Files with similar names but older timestamps (e.g., `materials_bert_service_simple.py` vs. `materials_bert_service_advanced.py`).
- **Redundant:**  
  - Multiple files implementing the same endpoint or logic (e.g., both Node and Python versions of the same service).

---

## 7. **Diagrams**

### **A. End-to-End Sequence (AI Onboarding Wizard Example)**

```mermaid
sequenceDiagram
    participant User
    participant Frontend:AIOnboardingWizard.tsx
    participant Service:aiService.ts
    participant Backend:app.js
    participant Logic:ai_listings_generator.py

    User->>Frontend: Clicks "Generate Listings"
    Frontend->>Service: inferListings(data)
    Service->>Backend: POST /api/ai-infer-listings
    Backend->>Logic: infer_listings(request)
    Logic-->>Backend: AI results
    Backend-->>Service: JSON response
    Service-->>Frontend: Return data
    Frontend-->>User: Show listings
```

---

## 8. **API Consistency and Mapping**

- **See `API_Consistency_Mapping.md`** for a full table of endpoints, backend existence, frontend usage, and notes.
- **Critical mismatches** are highlighted for immediate attention.

---

## 9. **Onboarding and Setup**

- **README.md** should include:
  - Setup steps for backend, frontend, and database.
  - How to run tests and validate setup.
  - Architecture diagrams
  - API mapping reference
  - Troubleshooting and FAQ

---

**This documentation is intended as a living reference for all developers, ensuring clarity, maintainability, and production-grade standards across the ISM [AI] platform.** 