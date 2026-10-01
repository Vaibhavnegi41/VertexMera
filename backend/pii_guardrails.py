import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class PIIEntity:
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float = 1.0

@dataclass
class PIIResult:
    original_text: str
    masked_text: str
    entities: List[PIIEntity] = field(default_factory=list)
    pii_map: Dict[str, str] = field(default_factory=dict)

# Deterministic regex patterns for structured PII
PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)\d{3,4}[\s\-.]?\d{3,4}(?!\w)")),
    ("SSN", re.compile(r"\b(?!000|666|9\d{2})\d{3}[\s\-]?(?!00)\d{2}[\s\-]?(?!0000)\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2})|35\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{3,4}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")),
    ("DATE_OF_BIRTH", re.compile(r"\b(?:DOB|Birth Date|born on)[\s:]+(?:(?:0[1-9]|[12]\d|3[01])[\/\-\.](?:0[1-9]|1[0-2])[\/\-\.]\d{2,4})\b", re.IGNORECASE)),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("PASSPORT", re.compile(r"\b[A-PR-WY][1-9]\d{6,8}\b", re.IGNORECASE)),
    ("SECRET_KEY", re.compile(r"\b(?:api[_\-]?key|secret|token|password|auth[_\-]?token)[\s:=]+['\"]?([A-Za-z0-9_\-\.]{16,})['\"]?\b", re.IGNORECASE)),
]

# Common label words that should never be tagged as PII entities by model
EXCLUDED_WORDS = {
    "ssn", "pan", "dob", "email", "mail", "phone", "tel", "fax", 
    "id", "url", "pin", "name", "address", "zip", "aadhaar", "passport", "token"
}

SPACY_LABELS = {
    "PERSON": "PERSON_NAME",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "NORP": "NATIONALITY",
}

_nlp = None

def get_nlp_model():
    """Load lightweight spaCy model with only NER enabled for maximum speed."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Disable unnecessary pipeline components to run 10x faster
            _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        except Exception:
            _nlp = None
    return _nlp


class PIIGuardrails:
    """Hybrid PII Guardrail: Fast Regex Pass followed by lightweight Semantic NER Pass."""

    def __init__(self, enable_model: bool = True):
        self.enable_model = enable_model
        self._pii_map: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}

    def _get_placeholder(self, entity_type: str, original: str) -> str:
        key = f"{entity_type}::{original.lower().strip()}"
        if key not in self._pii_map:
            self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
            self._pii_map[key] = f"[{entity_type}_{self._counters[entity_type]}]"
        return self._pii_map[key]

    def mask(self, text: str) -> PIIResult:
        if not text or not isinstance(text, str):
            return PIIResult(original_text=text or "", masked_text=text or "")

        entities: List[PIIEntity] = []

        # 1. Deterministic Pass (Regex)
        for entity_type, pattern in PATTERNS:
            for match in pattern.finditer(text):
                val = match.group().strip()
                if len(val) >= 3:
                    entities.append(PIIEntity(entity_type, match.start(), match.end(), val, 1.0))

        # 2. Model-Based Pass (Fast NER)
        if self.enable_model:
            nlp = get_nlp_model()
            if nlp:
                try:
                    doc = nlp(text[:25000])
                    for ent in doc.ents:
                        if ent.label_ in SPACY_LABELS:
                            val = ent.text.strip()
                            if len(val) >= 2 and val.lower() not in EXCLUDED_WORDS:
                                entities.append(PIIEntity(SPACY_LABELS[ent.label_], ent.start_char, ent.end_char, val, 0.85))
                except Exception:
                    pass

        # 3. Deduplicate overlapping spans
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        merged: List[PIIEntity] = []
        for ent in entities:
            if not merged or ent.start >= merged[-1].end:
                merged.append(ent)

        # 4. Mask right-to-left
        masked = text
        applied_map = {}
        for ent in sorted(merged, key=lambda e: e.start, reverse=True):
            placeholder = self._get_placeholder(ent.entity_type, ent.text)
            masked = masked[:ent.start] + placeholder + masked[ent.end:]
            applied_map[placeholder] = ent.text

        return PIIResult(original_text=text, masked_text=masked, entities=merged, pii_map=applied_map)

    @property
    def pii_map(self) -> Dict[str, str]:
        return dict(self._pii_map)


# Global singleton instance
_guard = None

def get_guardrails() -> PIIGuardrails:
    global _guard
    if _guard is None:
        _guard = PIIGuardrails()
    return _guard

def mask_text(text: str) -> str:
    """Mask PII in a single string."""
    return get_guardrails().mask(text).masked_text

def mask_documents(docs: list) -> list:
    """Mask PII in a list of Document objects in-place."""
    guard = get_guardrails()
    for doc in docs:
        if hasattr(doc, "page_content"):
            doc.page_content = guard.mask(doc.page_content).masked_text
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                doc.metadata["pii_redacted"] = True
    return docs
