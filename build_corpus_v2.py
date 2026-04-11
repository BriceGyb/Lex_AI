"""
LexAI — Construction du corpus étendu (Sprint 2)
=================================================
Cible : Code Pénal + Code Civil uniquement
Limite : 200 articles par code → ~400 articles au total
Sortie : corpus_penal_civil.json  (lois_francaises.json NON modifié)

Usage :
  python build_corpus_v2.py              # fetch + génère eval dataset
  python build_corpus_v2.py --no-eval   # fetch uniquement, sans générer le dataset Q/A

Datasets générés (TOUS conservés pour le rapport) :
  lois_francaises.json     ← corpus v1 Sprint 1  (NON touché)
  eval_dataset.json        ← eval v1 Sprint 1    (NON touché)
  corpus_penal_civil.json  ← corpus v2 Sprint 2
  eval_dataset_v2.json     ← eval v2 Sprint 2
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OUTPUT_CORPUS = "corpus_penal_civil.json"
OUTPUT_EVAL   = "eval_dataset_v2.json"
MAX_PAR_CODE  = 200   # 200 × 2 codes = ~400 articles

# On ne cible que Code Pénal et Code Civil pour ce corpus étendu
CODES_CIBLES_V2 = {
    "Code Civil":  "LEGITEXT000006070721",
    "Code Pénal":  "LEGITEXT000006070719",
}


# ── Fetch ─────────────────────────────────────────────────────────────────────

async def fetch_corpus() -> list[dict]:
    """Récupère les articles Code Pénal + Code Civil via l'API Légifrance."""
    from ingestion.legifrance_fetcher import LegiFranceFetcher

    client_id     = os.getenv("PISTE_CLIENT_ID")
    client_secret = os.getenv("PISTE_CLIENT_SECRET")

    if not client_id or not client_secret:
        print(
            "[STOP] Variables PISTE_CLIENT_ID et PISTE_CLIENT_SECRET manquantes dans .env\n"
            "       Ajoutez-les et relancez le script."
        )
        return []

    print("=" * 62)
    print("  LexAI — Ingestion corpus étendu (Sprint 2)")
    print(f"  Codes cibles : Code Civil + Code Pénal")
    print(f"  Max articles par code : {MAX_PAR_CODE}")
    print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 62)

    # Patch temporaire : restreindre CODES_CIBLES au sous-ensemble voulu
    import ingestion.legifrance_fetcher as lf_module
    codes_originaux = lf_module.CODES_CIBLES.copy()
    lf_module.CODES_CIBLES = CODES_CIBLES_V2

    fetcher  = LegiFranceFetcher(client_id, client_secret, max_articles_par_code=MAX_PAR_CODE)
    articles = []

    try:
        async for article in fetcher.fetch_all():
            articles.append(article)
            print(f"  [{len(articles):>4}] {article['code']} — {article['article'][:60]}")
    except Exception as e:
        print(f"\n[ERREUR] Ingestion interrompue : {e}")
        if articles:
            print(f"  -> {len(articles)} articles partiels sauvegardes.")
    finally:
        # Restaurer les codes originaux
        lf_module.CODES_CIBLES = codes_originaux

    return articles


# ── Génération dataset Q/A ────────────────────────────────────────────────────

def generer_eval_dataset(articles: list[dict]) -> list[dict]:
    """
    Génère 1 paire Q/A par article via GPT-4o-mini.
    Sauvegarde dans eval_dataset_v2.json.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    paires = []

    print(f"\n[~] Génération dataset Q/A ({len(articles)} articles)...")

    for i, article in enumerate(articles):
        print(f"  [{i+1}/{len(articles)}] {article['code']} — {article['article']}")
        texte = article["texte"][:800]

        prompt = f"""Voici un article de loi français. Génère UNE question juridique précise et sa réponse correcte.

Article : {article['article']} ({article['code']})
Texte : {texte}

Retourne un JSON avec exactement ces deux champs :
- "question" : une question juridique spécifique en français à laquelle cet article répond directement
- "ground_truth" : la réponse précise basée UNIQUEMENT sur ce texte (1-3 phrases en français)

Retourne UNIQUEMENT le JSON, sans explication."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            paires.append({
                "question":    data["question"],
                "ground_truth": data["ground_truth"],
                "article_ref": f"{article['code']} — {article['article']}",
            })
        except Exception as e:
            print(f"    [!] Erreur: {e}")

    return paires


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(generer_eval: bool = True):
    parser = argparse.ArgumentParser(description="LexAI — Corpus étendu Sprint 2")
    parser.add_argument("--no-eval", action="store_true", help="Ne pas générer le dataset Q/A")
    args = parser.parse_args()

    # 1. Fetch
    articles = await fetch_corpus()
    if not articles:
        return

    # 2. Sauvegarde corpus
    corpus = {"corpus_juridique": articles}
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Corpus sauvegarde : {OUTPUT_CORPUS} ({len(articles)} articles)")

    # 3. Dataset Q/A
    if not args.no_eval:
        paires = generer_eval_dataset(articles)
        with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
            json.dump(paires, f, ensure_ascii=False, indent=2)
        print(f"[+] Dataset eval sauvegarde : {OUTPUT_EVAL} ({len(paires)} paires Q/A)")

    print("\n" + "=" * 62)
    print("  Recapitulatif des datasets (pour le rapport)")
    print("=" * 62)
    print(f"  lois_francaises.json   : corpus v1 Sprint 1 (NON modifie)")
    print(f"  eval_dataset.json      : eval v1 Sprint 1  (NON modifie)")
    print(f"  {OUTPUT_CORPUS:<23}: corpus v2 Sprint 2 ({len(articles)} articles)")
    if not args.no_eval:
        print(f"  {OUTPUT_EVAL:<23}: eval v2 Sprint 2  ({len(paires)} paires)")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
