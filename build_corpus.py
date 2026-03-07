"""
LexAI — Construction du corpus depuis l'API Légifrance
Usage : python build_corpus.py

Ce script remplace le JSON statique par des données réelles de Légifrance.
Il écrase lois_francaises.json avec les articles récupérés.
Si l'API n'est pas configurée, il conserve le JSON existant.
"""

import asyncio
import json
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv
from ingestion.legifrance_fetcher import LegiFranceFetcher

load_dotenv()

OUTPUT_FILE    = "lois_francaises.json"
BACKUP_FILE    = "lois_francaises.backup.json"
MAX_PAR_CODE   = int(os.getenv("MAX_ARTICLES_PAR_CODE", "30"))


async def build():
    client_id     = os.getenv("PISTE_CLIENT_ID")
    client_secret = os.getenv("PISTE_CLIENT_SECRET")

    if not client_id or not client_secret:
        print(
            "[STOP] Variables PISTE_CLIENT_ID et PISTE_CLIENT_SECRET manquantes dans .env\n"
            "       → Créez un compte sur https://piste.gouv.fr pour obtenir vos identifiants.\n"
            "       → Le JSON statique existant est conservé."
        )
        return

    print("=" * 60)
    print("  LexAI — Ingestion API Légifrance (PISTE)")
    print(f"  Max articles par code : {MAX_PAR_CODE}")
    print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Backup du JSON actuel
    if os.path.exists(OUTPUT_FILE):
        shutil.copy(OUTPUT_FILE, BACKUP_FILE)
        print(f"[Backup] {OUTPUT_FILE} → {BACKUP_FILE}")

    fetcher  = LegiFranceFetcher(client_id, client_secret, max_articles_par_code=MAX_PAR_CODE)
    articles = []

    try:
        async for article in fetcher.fetch_all():
            articles.append(article)
            print(f"  [{len(articles):>4}] {article['code']} — {article['article'][:60]}")
    except Exception as e:
        print(f"\n[ERREUR] Ingestion interrompue : {e}")
        if articles:
            print(f"  → {len(articles)} articles déjà récupérés, sauvegarde partielle.")
        else:
            print("  → Aucun article récupéré. Le JSON existant est restauré.")
            if os.path.exists(BACKUP_FILE):
                shutil.copy(BACKUP_FILE, OUTPUT_FILE)
            return

    if not articles:
        print("[WARN] Aucun article récupéré. Le JSON existant est conservé.")
        return

    # Sauvegarde
    corpus = {"corpus_juridique": articles}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"[OK] {len(articles)} articles sauvegardés dans {OUTPUT_FILE}")
    print(f"     Backup conservé dans {BACKUP_FILE}")
    print("     Supprimez le dossier chroma_db/ avant de relancer app.py")
    print("     pour forcer la réindexation du nouveau corpus.")
    print("=" * 60)

    # Supprimer le vectorstore existant pour forcer la réindexation
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        print("[Reset] chroma_db/ supprimé → réindexation automatique au prochain lancement.")


if __name__ == "__main__":
    asyncio.run(build())
