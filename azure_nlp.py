"""
azure_nlp.py
------------
Minimal NLP script for the Roche POC: takes text in, returns sentiment,
key phrases and named entities using Azure AI Language (Text Analytics).

Azure resource (see .env.example):
    resource        hai-roche-ai-nlp
    resource group  hai-roche-poc
    API kind        TextAnalytics
    region          East US
    pricing tier    Standard

If the Azure endpoint/key is not configured (or the call fails), the script
falls back to the offline model in local_nlp.py so the demo still runs.

Usage:
    python azure_nlp.py "your text here"
    python azure_nlp.py --local "your text here"     # force offline model
    echo "your text" | python azure_nlp.py
"""

from __future__ import annotations

import json
import os
import sys

try:  # optional at runtime; only needed for the Azure path
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
except ImportError:  # pragma: no cover
    TextAnalyticsClient = None
    AzureKeyCredential = None

from local_nlp import TinySentimentModel

# --- Configuration --------------------------------------------------------
# Never hardcode the key. Set it in the environment or a local .env file.
AZURE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")
AZURE_KEY = os.getenv("AZURE_LANGUAGE_KEY", "")


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader so there is no python-dotenv dependency."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_client() -> "TextAnalyticsClient | None":
    endpoint, key = os.getenv("AZURE_LANGUAGE_ENDPOINT"), os.getenv("AZURE_LANGUAGE_KEY")
    if not (endpoint and key and TextAnalyticsClient):
        return None
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def analyse_azure(client: "TextAnalyticsClient", text: str) -> dict:
    """One document in, one structured result out."""
    docs = [text]
    sentiment = client.analyze_sentiment(documents=docs)[0]
    key_phrases = client.extract_key_phrases(documents=docs)[0]
    entities = client.recognize_entities(documents=docs)[0]

    return {
        "engine": "azure-ai-language",
        "text": text,
        "sentiment": sentiment.sentiment,
        "scores": {
            "positive": round(sentiment.confidence_scores.positive, 4),
            "neutral": round(sentiment.confidence_scores.neutral, 4),
            "negative": round(sentiment.confidence_scores.negative, 4),
        },
        "key_phrases": list(key_phrases.key_phrases),
        "entities": [
            {"text": e.text, "category": e.category, "confidence": round(e.confidence_score, 4)}
            for e in entities.entities
        ],
    }


def analyse_local(text: str) -> dict:
    result = TinySentimentModel().predict(text)
    result["engine"] = "local-tiny-model"
    return result


def main() -> None:
    _load_dotenv()

    args = [a for a in sys.argv[1:] if a != "--local"]
    force_local = "--local" in sys.argv
    text = " ".join(args).strip() or (sys.stdin.read().strip() if not sys.stdin.isatty() else "")

    if not text:
        print('usage: python azure_nlp.py "text to analyse"', file=sys.stderr)
        sys.exit(1)

    if force_local:
        print(json.dumps(analyse_local(text), indent=2))
        return

    client = get_client()
    if client is None:
        print("[warn] Azure endpoint/key not configured - using local model", file=sys.stderr)
        print(json.dumps(analyse_local(text), indent=2))
        return

    try:
        print(json.dumps(analyse_azure(client, text), indent=2))
    except Exception as exc:  # noqa: BLE001 - demo-level handling
        print(f"[warn] Azure call failed ({exc.__class__.__name__}) - using local model", file=sys.stderr)
        print(json.dumps(analyse_local(text), indent=2))


if __name__ == "__main__":
    main()
