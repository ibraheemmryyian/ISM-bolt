import json
import time
from collections import defaultdict
from typing import List, Dict, Any

# --- Model Wrappers ---
class MaterialsBERTWrapper:
    def __init__(self, service):
        self.service = service
    def predict_topics(self, text: str) -> List[str]:
        # TODO: Implement chunking if needed
        props = self.service.predict_material_properties('industry', text)
        return props.get('predicted_properties', [])
    def predict_properties(self, material: str) -> List[str]:
        props = self.service.predict_material_properties(material, '')
        return props.get('predicted_properties', [])
    def suggest_applications(self, material: str, properties: List[str]) -> List[str]:
        apps = self.service.suggest_applications(material, properties)
        return [a['application'] for a in apps]

class OMAT24Wrapper:
    def __init__(self):
        # TODO: Initialize OMAT24 model or API client
        pass
    def predict_topics(self, text: str) -> List[str]:
        # TODO: Implement OMAT24 topic extraction
        return []
    def predict_properties(self, material: str) -> List[str]:
        # TODO: Implement OMAT24 property prediction
        return []
    def suggest_applications(self, material: str, properties: List[str]) -> List[str]:
        # TODO: Implement OMAT24 application suggestion
        return []

# --- Evaluation ---
def evaluate(pred: List[str], gold: List[str]) -> Dict[str, float]:
    pred_set, gold_set = set(pred), set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {'precision': precision, 'recall': recall, 'f1': f1}

# --- Main Benchmark ---
def main():
    # Load company data
    with open('data/50_gulf_companies_fixed.json', 'r', encoding='utf-8') as f:
        companies = json.load(f)

    # TODO: Load ground truth labels for topics/properties/applications if available
    ground_truth = defaultdict(dict)  # {company_name: {material: {props, apps}, 'topics': [...]}}

    # Initialize models
    from .materials_bert_service import MaterialsBertService
    mb_service = MaterialsBertService()
    mb = MaterialsBERTWrapper(mb_service)
    omat = OMAT24Wrapper()  # TODO: Plug in OMAT24

    results = []

    for company in companies:
        name = company.get('name')
        industry = company.get('industry')
        materials = company.get('materials', [])
        products = company.get('products', [])
        # --- Trending Topic Extraction (from news) ---
        # TODO: Fetch or simulate news text for the company/industry
        news_text = f"{name} is a major player in {industry}."  # Placeholder
        for model_name, model in [('MaterialsBERT', mb), ('OMAT24', omat)]:
            start = time.time()
            pred_topics = model.predict_topics(news_text)
            latency = time.time() - start
            gold_topics = ground_truth[name].get('topics', [])
            metrics = evaluate(pred_topics, gold_topics) if gold_topics else {}
            results.append({
                'company': name,
                'task': 'topic_extraction',
                'model': model_name,
                'input': news_text,
                'predicted': pred_topics,
                'ground_truth': gold_topics,
                'latency': latency,
                **metrics
            })
        # --- Property Prediction & Application Suggestion ---
        for material in materials:
            for model_name, model in [('MaterialsBERT', mb), ('OMAT24', omat)]:
                # Property Prediction
                start = time.time()
                pred_props = model.predict_properties(material)
                latency = time.time() - start
                gold_props = ground_truth[name].get(material, {}).get('properties', [])
                metrics = evaluate(pred_props, gold_props) if gold_props else {}
                results.append({
                    'company': name,
                    'material': material,
                    'task': 'property_prediction',
                    'model': model_name,
                    'input': material,
                    'predicted': pred_props,
                    'ground_truth': gold_props,
                    'latency': latency,
                    **metrics
                })
                # Application Suggestion
                start = time.time()
                pred_apps = model.suggest_applications(material, pred_props)
                latency = time.time() - start
                gold_apps = ground_truth[name].get(material, {}).get('applications', [])
                metrics = evaluate(pred_apps, gold_apps) if gold_apps else {}
                results.append({
                    'company': name,
                    'material': material,
                    'task': 'application_suggestion',
                    'model': model_name,
                    'input': material,
                    'predicted': pred_apps,
                    'ground_truth': gold_apps,
                    'latency': latency,
                    **metrics
                })
    # Save results
    with open('data/benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark complete. Results saved to data/benchmark_results.json")

if __name__ == '__main__':
    main() 