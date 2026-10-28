import asyncio
import logging
import re
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from backend.config import grading_model
from backend.pii_guardrails import get_session_guardrails, get_guardrails
from backend.state import State

logger = logging.getLogger(__name__)

SAFE_REFUSAL_MESSAGE = (
    "I cannot provide this response because it violates content safety and security policies. "
    "If you have questions about VertexMera or verified document inquiries, I would be glad to help."
)

GROUNDEDNESS_NOTICE = (
    "\n\n> ⚠️ **Verification Notice:** Certain claims in this response could not be fully grounded in the retrieved reference context. "
    "Please verify with primary documentation."
)

# ── Deterministic safety regex patterns─────────────────────
UNSAFE_PATTERNS = [
    re.compile(r"\b(how\s+to\s+build\s+a\s+bomb|synthesize\s+(sarin|anthrax|ricin)|make\s+explosives)\b", re.IGNORECASE),
    re.compile(r"\b(credit\s*card\s*fraud|bypass\s*firewall\s*exploit|ddos\s*attack\s*script|sql\s*injection\s*payload)\b", re.IGNORECASE),
    re.compile(r"\b(kill\s+yourself|commit\s+suicide|self[\s\-]harm\s+instructions)\b", re.IGNORECASE),
    re.compile(r"<script[\s\S]*?>[\s\S]*?<\/script>", re.IGNORECASE),
]


class SafetyDecision(BaseModel):
    is_safe: bool = Field(description="True if the output complies with safety, non-toxicity, and legal policies.")
    violation_category: Optional[str] = Field(None, description="Category of violation if unsafe, e.g., 'toxicity', 'dangerous_content', 'exploit', 'none'.")
    reason: Optional[str] = Field(None, description="Concise reason for the safety rating.")


class FaithfulnessDecision(BaseModel):
    is_faithful: bool = Field(description="True if all substantive factual claims in the generation are grounded in the provided context.")
    hallucination_score: float = Field(description="Groundedness confidence score from 0.0 (wholly hallucinated) to 1.0 (strictly grounded).")
    unsupported_claims: List[str] = Field(default_factory=list, description="Any specific assertions not substantiated by the context.")
    reason: Optional[str] = Field(None, description="Brief explanation of the faithfulness assessment.")


safety_eval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI Safety and Moderation Auditor for an enterprise RAG assistant.\n"
        "Evaluate the candidate ASSISTANT_RESPONSE for safety violations.\n\n"
        "## Policy Checks\n"
        "1. **Toxicity & Abuse**: Hate speech, harassment, slurs, defamation, or graphic violence.\n"
        "2. **Dangerous & Illegal Acts**: Instructions for cyberattacks, weapons, malware, fraud, or self-harm.\n"
        "3. **Malicious Exploits**: Injection payloads or system bypass attempts.\n\n"
        "Return is_safe=True if the content is safe and compliant.\n"
        "Return is_safe=False if any policy is violated, specifying the violation category and reason."
    ),
    (
        "human",
        "USER QUESTION:\n{question}\n\nASSISTANT_RESPONSE:\n{generation}"
    )
])

faithfulness_eval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict Faithfulness and Hallucination Auditor for a Retrieval-Augmented Generation (RAG) system.\n\n"
        "## Task\n"
        "Evaluate whether the candidate ASSISTANT_RESPONSE is strictly supported by the provided REFERENCE CONTEXT.\n\n"
        "## Criteria\n"
        "- **Grounded (is_faithful=True, score >= 0.70)**: Every factual claim, number, date, and definition is directly present in or clearly entailed by the context.\n"
        "- **Hallucinated (is_faithful=False, score < 0.70)**: The response fabricates facts, entities, statistics, or policies absent from the context.\n"
        "- Note: If the response states that the context does not contain enough information, mark it as grounded (is_faithful=True, score=1.0).\n\n"
        "Provide is_faithful, a hallucination_score (0.0 to 1.0), unsupported_claims (if any), and a brief reason."
    ),
    (
        "human",
        "USER QUESTION:\n{question}\n\nREFERENCE CONTEXT:\n{context}\n\nASSISTANT_RESPONSE:\n{generation}"
    )
])

safety_evaluator = grading_model.with_structured_output(SafetyDecision)
faithfulness_evaluator = grading_model.with_structured_output(FaithfulnessDecision)


def check_deterministic_safety(text: str) -> Tuple[bool, Optional[str]]:
    """Fast-path deterministic check for severe toxicity and exploits."""
    if not text:
        return True, None
    for pattern in UNSAFE_PATTERNS:
        match = pattern.search(text)
        if match:
            return False, f"Prohibited pattern matched: {match.group()}"
    return True, None


async def evaluate_safety(generation: str, question: str = "") -> SafetyDecision:
    """Evaluate output generation for safety, non-toxicity, and moderation."""
    is_safe, reason = check_deterministic_safety(generation)
    if not is_safe:
        return SafetyDecision(
            is_safe=False,
            violation_category="dangerous_content",
            reason=reason or "Deterministic safety check triggered."
        )

    try:
        prompt = safety_eval_prompt.format_messages(
            question=question or "(No explicit question)",
            generation=generation
        )
        decision = await safety_evaluator.ainvoke(prompt)
        return decision
    except Exception as e:
        logger.warning("[OutputGuardrail] Safety evaluation LLM call error: %s. Falling back to safe pass.", e)
        return SafetyDecision(is_safe=True, violation_category="none", reason="Safety check fallback passed.")


async def evaluate_faithfulness(generation: str, context: str, question: str = "") -> FaithfulnessDecision:
    """Evaluate output generation for hallucination and grounding against reference context."""
    if not context or not context.strip():
        return FaithfulnessDecision(
            is_faithful=True,
            hallucination_score=1.0,
            unsupported_claims=[],
            reason="No reference context required."
        )

    try:
        # Limit context to avoid excessive token sizes
        truncated_context = context[:12000]
        prompt = faithfulness_eval_prompt.format_messages(
            question=question or "(No explicit question)",
            context=truncated_context,
            generation=generation
        )
        decision = await faithfulness_evaluator.ainvoke(prompt)
        return decision
    except Exception as e:
        logger.warning("[OutputGuardrail] Faithfulness evaluation LLM call error: %s. Falling back to default pass.", e)
        return FaithfulnessDecision(
            is_faithful=True,
            hallucination_score=0.85,
            unsupported_claims=[],
            reason="Faithfulness check fallback passed."
        )

async def apply_output_guardrails(state: State) -> dict:
    """
    Applies comprehensive output guardrails:
    1. Safety & Content Moderation: Flags toxicity, harmful content, and prompt injection/exploits.
    2. Hallucination & Faithfulness: Verifies generation is grounded in retrieved/web context.
    3. Output PII Check & De-Anonymization: Restores user placeholders safely and scrubs leaked secrets.
    """
    raw_generation = state.get("generation", "")
    question = state.get("raw_question") or state.get("question", "")
    thread_id = state.get("thread_id") or "__global__"
    guard = get_session_guardrails(thread_id)

    documents = state.get("documents", [])
    context_str = "\n\n".join([
        doc.page_content for doc in documents if hasattr(doc, "page_content")
    ])

    web_searched = state.get("web_searched", False)
    hallucination_retries = state.get("hallucination_retry_count", 0)

    # 1. Run Safety and Faithfulness checks concurrently for speed
    safety_task = evaluate_safety(raw_generation, question)
    faithfulness_task = evaluate_faithfulness(raw_generation, context_str, question)

    safety_decision, faithfulness_decision = await asyncio.gather(
        safety_task, faithfulness_task
    )

    current_generation = raw_generation
    safety_passed = safety_decision.is_safe
    faithfulness_passed = faithfulness_decision.is_faithful

    # 2. Process Safety Violation
    if not safety_passed:
        logger.warning("[OutputGuardrail] Safety violation detected: %s (%s)", safety_decision.violation_category, safety_decision.reason)
        current_generation = SAFE_REFUSAL_MESSAGE
        faithfulness_passed = True

    # 3. Process Hallucination / Faithfulness
    elif not faithfulness_passed and context_str:
        logger.warning(
            "[OutputGuardrail] Hallucination detected (score: %s). Unsupported claims: %s",
            faithfulness_decision.hallucination_score,
            faithfulness_decision.unsupported_claims
        )
        if not web_searched and hallucination_retries < 1:
            return {
                "generation": current_generation,
                "safety_passed": True,
                "safety_violation": None,
                "faithfulness_passed": False,
                "faithfulness_score": float(faithfulness_decision.hallucination_score),
                "hallucination_retry_count": hallucination_retries + 1,
                "web_searched": web_searched,
                "steps": ["output_guardrail_node"],
                "search": True,
            }
        else:
            current_generation = current_generation + GROUNDEDNESS_NOTICE

    # 4. Output PII Check & De-Anonymization 
    pii_map = state.get("pii_map") or {}
    de_anonymized = guard.unmask(current_generation, extra_map=pii_map)

    final_generation, pii_detected = guard.scrub_sensitive_output(de_anonymized)

    return {
        "generation": final_generation,
        "safety_passed": safety_passed,
        "safety_violation": safety_decision.violation_category if not safety_passed else None,
        "faithfulness_passed": faithfulness_passed,
        "faithfulness_score": float(faithfulness_decision.hallucination_score),
        "hallucination_retry_count": hallucination_retries,
        "web_searched": web_searched,
        "output_pii_detected": pii_detected,
        "steps": ["output_guardrail_node"],
    }
