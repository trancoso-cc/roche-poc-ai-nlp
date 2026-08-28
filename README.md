# Roche AI NLP PoC

Minimal NLP demo for the Roche POC. Text in, structured NLP output out.

Two interchangeable engines:

| Engine | File | Needs network | Notes |
|---|---|---|---|
| Azure AI Language | `azure_nlp.py` | yes | sentiment + key phrases + entities |
| Tiny local model | `local_nlp.py` | no | binary sentiment, stdlib only |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env      # then paste the key from the Azure portal
python azure_nlp.py "the documentation is well written and easy to follow"
```

Offline / air-gapped:

```bash
python local_nlp.py "the provisioning failed with an unclear error"
# or
python azure_nlp.py --local "..."
```

`azure_nlp.py` falls back to the local model automatically if the endpoint or
key is missing, or if the Azure call fails.

## Azure resource

| Field | Value |
|---|---|
| Resource | `hai-roche-ai-nlp` |
| Resource group | `hai-roche-poc` |
| Subscription ID | `0d237316-7492-425d-881e-7375ef7317fc` |
| API kind | TextAnalytics |
| Region | East US |
| Pricing tier | Standard |
| Endpoint | `https://hai-roche-ai-nlp.cognitiveservices.azure.com/` |

The endpoint above is the conventional form for a Cognitive Services resource
of this name. The portal currently reports **Provision failed** with a blank
endpoint, so confirm the real value under *Keys and Endpoint* once the resource
provisions successfully.

Keys are read from environment variables only. Nothing secret is committed.

## Model artefact

`models/tiny_sentiment.json` (~19 KB) holds a TF-IDF + logistic regression
binary sentiment classifier: vocabulary, IDF vector, coefficients, intercept.

- Inference is reimplemented in plain Python in `local_nlp.py`, so the file
  loads with no scikit-learn or numpy at runtime.
- Regenerate with `python train_tiny_model.py`. The training corpus is inline
  and human-readable, so the artefact is fully auditable.
- Trained on ~70 short English sentences. It is a demo artefact for pipeline
  and governance testing, not a production-grade classifier.

## Layout

```
azure_nlp.py             Azure client + CLI, falls back to local
local_nlp.py             offline inference, standard library only
train_tiny_model.py      regenerates the weight file
models/tiny_sentiment.json   the committed model weights
.env.example             config template
```
