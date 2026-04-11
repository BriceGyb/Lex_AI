# LexAI — Résultats RAGAS — Sprint 2

## Baseline : Hybrid BM25 + FAISS (RRF)

| Métrique | Score | Interprétation |
|---|---|---|
| **Faithfulness** | **0.8322** | Le LLM reste fidèle aux sources dans 83% des cas — peu d'hallucinations |
| **Context Precision** | **0.8935** | Les bons articles sont bien classés en tête du retrieval (89%) |
| **Context Recall** | **0.9944** | Quasiment tous les articles pertinents sont récupérés (99%) |
| Answer Relevancy | N/A | Non calculé (incompatibilité embeddings ragas 0.4.x) |

> **Questions évaluées :** 60 paires Q/A synthétiques générées depuis le corpus officiel Légifrance  
> **Pipeline évalué :** Hybrid Search BM25Okapi + FAISS (text-embedding-3-small) + RRF (0.65/0.35) → GPT-4o-mini

---

## Analyse

### Points forts
- **Context Recall 0.9944** : excellent — le retrieval retrouve presque systématiquement tous les articles pertinents. La recherche hybride BM25+FAISS est très efficace sur ce corpus.
- **Context Precision 0.8935** : les articles les plus pertinents arrivent bien en tête du Top-5 — le ranking RRF fonctionne bien.
- **Faithfulness 0.8322** : le LLM hallucine peu, la majorité des réponses sont bien ancrées dans les sources récupérées.

### Points à améliorer (objectifs Sprint 2)
- **Faithfulness < 1.0** : ~17% des réponses contiennent des éléments non sourcés → Cross-Encoder Reranking (4.3) et HyDE (4.4) devraient améliorer la qualité des contextes injectés.
- **Context Precision < 1.0** : certains articles non pertinents passent dans le Top-5 → le Reranking BGE-Reranker devrait filtrer ces faux positifs.

---

## Tableau comparatif (à compléter au fil du Sprint 2)

| Variante | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| **Baseline Hybrid BM25+FAISS** | 0.8322 | 0.8935 | 0.9944 |
| + Cross-Encoder Reranking | _à venir_ | _à venir_ | _à venir_ |
| + HyDE | _à venir_ | _à venir_ | _à venir_ |
| + Fine-tuning embeddings | _à venir_ | _à venir_ | _à venir_ |

---

*Généré le 10 avril 2026 — ragas 0.4.3 — corpus : 60 articles (6 codes français)*
