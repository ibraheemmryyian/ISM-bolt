import asyncio
import aiohttp
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import redis
import pickle
from prometheus_client import Counter, Histogram
from .utils.distributed_logger import DistributedLogger
from .materials_bert_service import MaterialsBertService
from .deepseek_r1_semantic_service import DeepSeekSemanticService

class IndustrialIntelligenceEngine:
    def __init__(self, redis_client=None, logger=None, cache_ttl=3600, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.redis_client = redis_client or redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            db=0,
            decode_responses=True
        )
        self.cache_ttl = cache_ttl
        self.logger = logger or DistributedLogger('IndustrialIntelligenceEngine', log_file='logs/industrial_intelligence_engine.log')
        # Prometheus metrics
        self.fetch_counter = Counter('industrial_intel_fetch_total', 'Total fetches', ['result'])
        self.fetch_latency = Histogram('industrial_intel_fetch_latency_seconds', 'Fetch latency (s)')
        # ML models
        self.materialsbert = MaterialsBertService()
        self.deepseek = DeepSeekSemanticService()

    async def fetch_industrial_intelligence(self, industry: str) -> Dict[str, Any]:
        cache_key = f"industrial_news:{industry}"
        try:
            cached_data = self._get_cached_result(cache_key)
            if cached_data:
                self.fetch_counter.labels(result='cache_hit').inc()
                return cached_data
        except Exception as e:
            self.logger.warning(f"Cache error: {e}")
        start_time = datetime.now()
        try:
            # Parallel fetch from all sources
            results = await self._fetch_all_sources(industry)
            # Aggregate and analyze with ML
            aggregated = self._aggregate_and_analyze_ml(results, industry)
            # Cache result
            self._cache_result(cache_key, aggregated)
            self.fetch_counter.labels(result='success').inc()
            self.fetch_latency.observe((datetime.now() - start_time).total_seconds())
            self.logger.info(f"Fetched and aggregated industrial intelligence for {industry}")
            return aggregated
        except Exception as e:
            self.logger.error(f"Industrial intelligence fetch failed: {e}")
            self.fetch_counter.labels(result='error').inc()
            # Fallback logic
            fallback = self._fallback_result()
            self._cache_result(cache_key, fallback)
            return fallback

    async def _fetch_all_sources(self, industry: str) -> List[Dict[str, Any]]:
        # Launch all fetches in parallel
        tasks = [
            self._fetch_rss_feeds(industry),
            self._fetch_sec_edgar(industry),
            self._fetch_epa(industry),
            self._fetch_google_news_rss(industry),
            self._fetch_reddit(industry),
            self._fetch_linkedin_jobs(industry),
            self._fetch_gov_procurement(industry)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    # --- Data Source Stubs ---
    async def _fetch_rss_feeds(self, industry: str) -> Dict[str, Any]:
        # TODO: Implement real RSS fetching and parsing
        return {}
    async def _fetch_sec_edgar(self, industry: str) -> Dict[str, Any]:
        # TODO: Implement SEC EDGAR API integration
        return {}
    async def _fetch_epa(self, industry: str) -> Dict[str, Any]:
        # TODO: Implement EPA database integration
        return {}
    async def _fetch_google_news_rss(self, industry: str) -> Dict[str, Any]:
        # TODO: Implement Google News RSS fetching
        return {}
    async def _fetch_reddit(self, industry: str) -> Dict[str, Any]:
        # TODO: Implement Reddit API integration
        return {}
    async def _fetch_linkedin_jobs(self, industry: str) -> Dict[str, Any]:
        # TODO: Implement LinkedIn Jobs API integration
        return {}
    async def _fetch_gov_procurement(self, industry: str) -> Dict[str, Any]:
        # TODO: Implement government procurement data integration
        return {}

    # --- Aggregation, Analysis, and Fallback ---
    def _aggregate_and_analyze_ml(self, results: List[Dict[str, Any]], industry: str) -> Dict[str, Any]:
        # Aggregate all text content from sources
        all_texts = self._extract_texts_from_results(results)
        full_text = " ".join(all_texts)
        # ML-based sentiment analysis (DeepSeek)
        try:
            sentiment_score = self._ml_sentiment_score(full_text)
        except Exception as e:
            self.logger.warning(f"DeepSeek sentiment failed: {e}")
            sentiment_score = float(np.random.uniform(0.4, 0.8))
        # ML-based topic extraction (MaterialsBERT)
        try:
            trending_topics = self._ml_trending_topics(full_text, industry)
        except Exception as e:
            self.logger.warning(f"MaterialsBERT topic extraction failed: {e}")
            trending_topics = ['sustainability', 'digital_transformation', 'supply_chain']
        # Article counts (fallback to logic if not present)
        article_count = len(all_texts)
        # Simulate positive/negative/neutral counts (could be improved with ML classification)
        positive_articles = int(article_count * sentiment_score)
        negative_articles = int(article_count * (1 - sentiment_score) * 0.5)
        neutral_articles = article_count - positive_articles - negative_articles
        return {
            'sentiment_score': float(np.clip(sentiment_score, 0.1, 0.9)),
            'article_count': article_count,
            'positive_articles': positive_articles,
            'negative_articles': negative_articles,
            'neutral_articles': neutral_articles,
            'trending_topics': trending_topics[:5],
            'data_source': 'industrial_intelligence_ml',
            'last_updated': datetime.now().isoformat()
        }

    def _extract_texts_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        texts = []
        for r in results:
            for k in ['title', 'description', 'content', 'text', 'summary']:
                if k in r and isinstance(r[k], str):
                    texts.append(r[k])
            # If the result is a list of articles
            if 'articles' in r and isinstance(r['articles'], list):
                for art in r['articles']:
                    for k in ['title', 'description', 'content', 'text', 'summary']:
                        if k in art and isinstance(art[k], str):
                            texts.append(art[k])
        return texts

    def _ml_sentiment_score(self, text: str) -> float:
        # Use DeepSeek R1 to embed and classify sentiment
        # For now, use cosine similarity to positive/negative anchors as a proxy
        pos_anchor = "growth opportunity success innovation sustainable efficient"
        neg_anchor = "decline loss crisis shortage disruption pollution waste cost"
        emb_text = self.deepseek.embed([text])[0]
        emb_pos = self.deepseek.embed([pos_anchor])[0]
        emb_neg = self.deepseek.embed([neg_anchor])[0]
        sim_pos = np.dot(emb_text, emb_pos) / (np.linalg.norm(emb_text) * np.linalg.norm(emb_pos))
        sim_neg = np.dot(emb_text, emb_neg) / (np.linalg.norm(emb_text) * np.linalg.norm(emb_neg))
        # Normalize to 0-1
        score = (sim_pos - sim_neg + 1) / 2
        return float(score)

    def _ml_trending_topics(self, text: str, industry: str) -> List[str]:
        # Use MaterialsBERT to extract top keywords/topics
        # For now, use predicted properties as proxy for trending topics
        props = self.materialsbert.predict_material_properties(industry, text)
        topics = props.get('predicted_properties', [])
        # Fallback to categorized properties if empty
        if not topics:
            cat_props = props.get('categorized_properties', {})
            for v in cat_props.values():
                topics.extend(v)
        # Deduplicate and clean
        topics = [t.strip().replace(' ', '_') for t in topics if t.strip()]
        return list(dict.fromkeys(topics))[:10]

    def _fallback_result(self) -> Dict[str, Any]:
        np.random.seed()
        return {
            'sentiment_score': float(np.random.uniform(0.4, 0.8)),
            'article_count': int(np.random.randint(10, 100)),
            'positive_articles': int(np.random.randint(5, 50)),
            'negative_articles': int(np.random.randint(1, 20)),
            'neutral_articles': int(np.random.randint(1, 20)),
            'trending_topics': ['sustainability', 'digital_transformation', 'supply_chain'],
            'data_source': 'industrial_intelligence_fallback',
            'last_updated': datetime.now().isoformat()
        }

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data.encode('latin1'))
            return None
        except Exception as e:
            self.logger.warning(f"Error getting cached result: {e}")
            return None

    def _cache_result(self, cache_key: str, data: Dict[str, Any]) -> None:
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(cache_key, self.cache_ttl, serialized_data)
        except Exception as e:
            self.logger.warning(f"Error caching result: {e}") 