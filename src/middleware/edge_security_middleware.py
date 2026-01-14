import json
import re
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException
import spacy

# Try to load spaCy model, make it optional for graceful degradation
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Model not found - prompt injection detection will be disabled
    # but structural payload detection will still work
    pass


# -------------------------
# Normalization
# -------------------------

def normalize(text: str) -> str:
    text = text.strip()
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    return text


# -------------------------
# Regex structure checks
# -------------------------
# These do NOT rely on keywords — only on shapes of code / templates / blocks

STRUCTURAL_PATTERNS = [
    r"`{3,}[\s\S]*?`{3,}",      # fenced code blocks
    r"<[^>]+>",                # HTML / XML tags
    r"\{[\s\S]*?\}",           # template / payload blocks
    r"\[[\s\S]*?\]",           # bracketed payloads
    r"\([\s\S]*?\)",           # script-like calls
    r";\s*$",                  # command termination
    r"\$\{[\s\S]*?\}",         # variable interpolation
]


def has_structural_payload(text: str) -> bool:
    for pattern in STRUCTURAL_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def sanitize(text: str) -> str:
    text = re.sub(r"`{3,}[\s\S]*?`{3,}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[\s\S]*?\}", "", text)
    return text.strip()


# -------------------------
# spaCy-based intent analysis
# -------------------------

def imperative_ratio(doc) -> float:
    """Measures how command-like the sentence is."""
    if not doc:
        return 0.0
    imperatives = 0
    for sent in doc.sents:
        if sent and sent[0].pos_ == "VERB":
            imperatives += 1
    return imperatives / max(len(list(doc.sents)), 1)


def meta_reference_score(doc) -> float:
    """
    Detects whether user is talking *to the system* instead of
    *about a business object* using dependency structure.
    """
    score = 0.0
    for token in doc:
        # Pronouns acting as subjects of root verbs
        if token.dep_ == "nsubj" and token.pos_ == "PRON":
            score += 0.25
        # Verbs that take clauses (control intent)
        if token.dep_ in {"ccomp", "xcomp"}:
            score += 0.15
    return min(score, 1.0)


def execution_control_score(doc) -> float:
    """
    Measures how much the sentence is about *how to act*
    rather than *what outcome is needed*.
    """
    score = 0.0
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            if token.morph.get("Mood") == ["Imp"]:
                score += 0.3
        if token.dep_ in {"advcl", "ccomp"}:
            score += 0.1
    return min(score, 1.0)


def business_object_score(doc) -> float:
    """
    Business queries usually reference concrete objects:
    amounts, entities, nouns with numbers, etc.
    """
    score = 0.0
    for token in doc:
        if token.pos_ == "NOUN":
            score += 0.1
        if token.pos_ == "NUM":
            score += 0.2
    return min(score, 1.0)


def is_prompt_injection(text: str) -> bool:
    """
    Pure NLP-based classification.
    No keywords. No phrases. No rules about meaning.
    Only syntax, grammar, and discourse intent.
    
    Returns False if spaCy model is not available (graceful degradation).
    """
    if nlp is None:
        # If model not available, only use structural checks
        return False
    
    doc = nlp(text)

    risk = 0.0

    risk += imperative_ratio(doc) * 0.4
    risk += meta_reference_score(doc) * 0.3
    risk += execution_control_score(doc) * 0.2

    business_score = business_object_score(doc)

    # If business grounding is weak, raise risk
    if business_score < 0.3:
        risk += 0.2

    return risk >= 0.6


# -------------------------
# Request helpers
# -------------------------

async def extract_body(request: Request) -> Dict[str, Any]:
    try:
        raw = await request.body()
        return json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        return {}


def rebuild_request_body(request: Request, body: Dict[str, Any]) -> None:
    request._body = json.dumps(body).encode("utf-8")


# -------------------------
# Middleware
# -------------------------

async def prompt_security_middleware(request: Request, call_next):
    body = await extract_body(request)

    user_query = (
        body.get("query")
        or body.get("message")
        or body.get("user_query")
    )

    if not user_query:
        return await call_next(request)

    normalized = normalize(user_query)

    # Structural payload check (regex only) - always active
    if has_structural_payload(normalized):
        raise HTTPException(
            status_code=400,
            detail="Input format is not allowed. Please submit only a natural language request."
        )

    # NLP intent detection (spaCy only) - only if model is available
    # if nlp is not None and is_prompt_injection(normalized):
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Instructional or control-style input detected. Please submit only your business request."
    #     )

    # Sanitize before forwarding
    safe_query = sanitize(normalized)

    body["query"] = safe_query
    body["user_query"] = safe_query

    rebuild_request_body(request, body)
    return await call_next(request)
