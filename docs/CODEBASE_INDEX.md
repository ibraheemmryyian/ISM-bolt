# SymbioFlows Codebase Index

> Last updated: <!--DATE_PLACEHOLDER-->

This document provides a high-level overview of the main directories and files in the repository. Use it as a starting point for navigation and onboarding. It **does not** replace the in-depth documentation found elsewhere in `docs/`.  

---

## Top-Level Structure

| Path | Type | Brief Description |
| --- | --- | --- |
| `ai_service/` | Python pkg | Core AI service abstractions and helpers. |
| `ai_service_flask/` | Microservices | Flask-based AI service wrappers (pricing, analytics, logistics, federated learning, etc.). |
| `ai-service/` | 🔄 Legacy | Earlier iteration of `ai_service/` retained for migration reference. |
| `archive/` | Scripts & Logs | Historical scripts, experiments, and generated artefacts. |
| `backend/` | Monorepo | Main backend containing orchestration engines, services, ML models, and utilities. |
| `data/` | Data | Raw or sample datasets. |
| `database/` | SQL | Database schemas & migrations. |
| `docs/` | Docs | Architectural and API documentation. |
| `federated_models/` | JSON | Global models for federated learning. |
| `frontend/` | React | Modern web front-end (TypeScript, Vite, Tailwind). |
| `frontendold/` | Legacy | Archived React front-end kept for reference. |
| `gnn_models/` | Models | Graph Neural Network test models. |
| `infrastructure/` | IaC | Docker, FastAPI, and infra deployment scripts. |
| `july/` | Notes | Monthly planning notes. |
| `k8s/` | Kubernetes | Deployment and CronJob YAMLs. |
| `model_storage/` | Models | Centralised model registry and backups. |
| `models/` | Models | Organised model categories (base, production, staging, etc.). |
| `monitoring/` | Observability | Prometheus & Grafana configs. |
| `scripts/` | Automation | Utility scripts for setup, deduplication, etc. |
| `supabase/` | SQL | Supabase migration files. |

---

## Detailed Breakdown

### 1. ai_service_flask/
Key Flask microservices that wrap AI functionality behind HTTP endpoints.

- `advanced_analytics_service.py` – Exposure of advanced analytics models.
- `ai_gateway.py` – Gateway orchestrator for routing requests to underlying AI services.
- `ai_pricing_service_wrapper.py` – Price optimisation models wrapper.
- `federated_learning_service.py` – Federated training coordination.
- `gnn_inference_service.py` – GNN-based inference service.
- `logistics_service_wrapper.py` – Logistics cost & optimisation service.
- `multi_hop_symbiosis_service.py` – Multi-hop graph symbiosis inference.
- `model_storage/` – On-disk artefacts for the Flask services above.

### 2. backend/
Monolithic repository of production-grade back-end components, including orchestration layers, AI engines, utilities, and service integrations.

Key highlights:

- **Orchestration Engines** – `advanced_orchestration_engine.py`, `workflow_orchestrator.py`, etc.
- **AI Engines & Services** – `ai_material_analysis_engine.py`, `gnn_reasoning_engine.py`, `materials_bert_service*.py`.
- **Pipelines** – `ai_retraining_pipeline.py`, `generate_supervised_materials_and_matches.py`, etc.
- **Integrations** – `service_mesh_proxy.py`, `ai_service_integration.py`.
- **Utilities** – `utils/`, `middleware/`, `scripts/`.
- **Models & Storage** – `model_storage/`, `fusion_models/`, `models/`.

### 3. frontend/
React (TypeScript) SPA powered by Vite.

- `src/components/` – UI components, including `AdaptiveAIOnboarding.tsx`, `AdminHub.tsx`, and shared `ui/` primitives.
- `src/lib/` – Client-side API & service wrappers.
- `src/types/` – Shared TypeScript types.
- `supabase/` – Front-end migrations.

### 4. infrastructure/
Infrastructure-as-Code assets.

- `docker-compose.yml` – Local orchestration of core services.
- `fastapi-service/` – Stand-alone FastAPI wrapper used in certain deployments.

### 5. k8s/
Kubernetes deployment descriptors and CronJobs for production clusters.

### 6. monitoring/
Prometheus rules and Grafana dashboards to monitor system health.

---

## See Also

- `docs/ARCHITECTURE_OVERVIEW.md` – Big-picture architecture.
- `docs/PRODUCTION_DEPLOYMENT_GUIDE.md` – Deployment instructions.
- `README.md` – Quick start.

---

*Generated automatically by the indexing assistant.*