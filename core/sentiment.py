"""Source-agnostic sentiment scoring with a transformer model.

Uses cardiffnlp/twitter-roberta-base-sentiment-latest — a RoBERTa model
fine-tuned on tweets, well suited to short social text (comments, transcript
chunks). It replaces the earlier VADER scorer but keeps the SAME interface:

    score(records) -> records   # sets sentiment_label + sentiment_score in place

So the rest of the pipeline (and any future model swap) stays unchanged.

  * sentiment_label: "positive" | "neutral" | "negative" (argmax class)
  * sentiment_score: signed confidence in [-1, 1] = P(positive) - P(negative),
    so it stays comparable to the old VADER compound (positive => +, negative => -).

The model is downloaded from the Hugging Face Hub on first use and cached
locally; no API key or per-call cost.
"""

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Normalize whatever label strings the model emits to our three classes.
_LABEL_MAP = {
    "negative": "negative", "neutral": "neutral", "positive": "positive",
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
}

_pipeline = None


def _get_pipeline():
    """Lazily build the HF sentiment pipeline (loads/caches the model once)."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            top_k=None,        # return scores for ALL classes
            truncation=True,
            max_length=512,
        )
    return _pipeline


def _resolve(class_scores: dict):
    """From {label: prob} return (label, signed_score)."""
    if not class_scores:
        return "neutral", 0.0
    label = max(class_scores, key=class_scores.get)
    signed = class_scores.get("positive", 0.0) - class_scores.get("negative", 0.0)
    return label, round(float(signed), 4)


def score(records, batch_size: int = 32):
    """Attach `sentiment_label` and `sentiment_score` to each record in place.

    `records` is a list of common-record dicts. Returns the same list.
    """
    if not records:
        return records

    # Empty-text records get a neutral default and are skipped by the model.
    texts, index = [], []
    for i, rec in enumerate(records):
        text = " ".join((rec.get("text") or "").split())
        if text:
            texts.append(text)
            index.append(i)
        else:
            rec["sentiment_label"] = "neutral"
            rec["sentiment_score"] = 0.0

    if texts:
        clf = _get_pipeline()
        results = clf(texts, batch_size=batch_size)
        for i, result in zip(index, results):
            class_scores = {
                _LABEL_MAP.get(item["label"].lower(), item["label"].lower()): item["score"]
                for item in result
            }
            label, signed = _resolve(class_scores)
            records[i]["sentiment_label"] = label
            records[i]["sentiment_score"] = signed

    return records
