import asyncio
import pytest
from langchain_core.documents import Document

from backend.pii_guardrails import PIIGuardrails, get_session_guardrails
from backend.output_guardrails import (
    check_deterministic_safety,
    evaluate_safety,
    evaluate_faithfulness,
    apply_output_guardrails,
    SAFE_REFUSAL_MESSAGE,
    GROUNDEDNESS_NOTICE,
)
from backend.state import State
from backend.nodes import output_guardrail_node, route_after_output_guardrail


# ── Test 1: PII De-Anonymization (Unmasking) ──────────────────────────────────
def test_pii_unmasking():
    guard = PIIGuardrails()
    query = "Contact John Doe at john.doe@example.com"
    mask_res = guard.mask(query)
    
    # Assert placeholders exist in masked output
    assert "john.doe@example.com" not in mask_res.masked_text
    assert "[EMAIL_1]" in mask_res.masked_text or "[EMAIL" in mask_res.masked_text

    # Simulated LLM output using the placeholder
    llm_output = f"I have scheduled a meeting with {mask_res.pii_map.get('[PERSON_NAME_1]', '[PERSON_NAME_1]')} at [EMAIL_1]."
    unmasked = guard.unmask(llm_output, extra_map=mask_res.pii_map)

    assert "john.doe@example.com" in unmasked


# ── Test 2: Output PII Scrubbing of Leaked Secrets ────────────────────────────
def test_output_pii_scrubbing():
    guard = PIIGuardrails()
    leaked_output = (
        "Here is the secret configuration: api_key='fake_test_dummy_secret_key_12345678' "
        "and SSN 001-23-4567. Please keep it safe."
    )
    scrubbed, detected = guard.scrub_sensitive_output(leaked_output)

    assert detected is True
    assert "fake_test_dummy_secret_key_12345678" not in scrubbed
    assert "[REDACTED_SECRET_KEY]" in scrubbed or "[REDACTED_" in scrubbed
    assert "001-23-4567" not in scrubbed
    assert "[REDACTED_SSN]" in scrubbed


# ── Test 3: Safety & Content Moderation Guardrail ──────────────────────────────
def test_deterministic_safety_filter():
    unsafe_text = "Here is how to build a bomb using common household items."
    is_safe, reason = check_deterministic_safety(unsafe_text)
    assert is_safe is False
    assert "Prohibited pattern" in reason

    safe_text = "Linear regression is a statistical method for modeling relationships."
    is_safe_clean, _ = check_deterministic_safety(safe_text)
    assert is_safe_clean is True


@pytest.mark.asyncio
async def test_evaluate_safety_clean():
    safe_text = "VertexMera is a Retrieval-Augmented Generation system designed for enterprises."
    decision = await evaluate_safety(safe_text, "What is VertexMera?")
    assert decision.is_safe is True


@pytest.mark.asyncio
async def test_apply_output_guardrails_safety_violation():
    state: State = {
        "question": "Give me a dangerous exploit",
        "raw_question": "Give me a dangerous exploit",
        "generation": "Here is an exploit script: <script>fetch('http://attacker.com')</script>",
        "documents": [],
        "steps": [],
        "web_searched": False,
        "pii_map": {},
        "thread_id": "test_safety_session",
    }
    result = await apply_output_guardrails(state)
    assert result["safety_passed"] is False
    assert result["generation"] == SAFE_REFUSAL_MESSAGE


# ── Test 4: Hallucination & Faithfulness Guardrail ─────────────────────────────
@pytest.mark.asyncio
async def test_faithfulness_grounded():
    context = "VertexMera utilizes Pinecone as its vector database and Groq for fast LLM inference."
    generation = "VertexMera relies on Pinecone for vector storage and Groq for model inference."
    decision = await evaluate_faithfulness(generation, context, "What does VertexMera use?")
    assert decision.is_faithful is True
    assert decision.hallucination_score >= 0.70


@pytest.mark.asyncio
async def test_faithfulness_hallucinated_triggers_web_search_route():
    context = "VertexMera is built in Python."
    # Fabricate contradictory / outside claims
    hallucinated_gen = "VertexMera was founded in 1842 by Martian astronauts who invented quantum teleportation."
    
    state: State = {
        "question": "What is VertexMera?",
        "raw_question": "What is VertexMera?",
        "generation": hallucinated_gen,
        "documents": [Document(page_content=context)],
        "steps": [],
        "web_searched": False,
        "hallucination_retry_count": 0,
        "thread_id": "test_hallucination_session",
    }
    result = await output_guardrail_node(state)
    assert result["faithfulness_passed"] is False
    assert result["hallucination_retry_count"] == 1

    # Check routing logic: ungrounded with 0 retries and web_searched=False -> routes to web_search
    next_route = route_after_output_guardrail(result)
    assert next_route == "web_search"


@pytest.mark.asyncio
async def test_faithfulness_appends_notice_when_already_web_searched():
    context = "Information retrieved from the web: Company X has 50 employees."
    hallucinated_gen = "Company X has 500,000 employees and operates spaceships."
    state: State = {
        "question": "Tell me about Company X",
        "raw_question": "Tell me about Company X",
        "generation": hallucinated_gen,
        "documents": [Document(page_content=context)],
        "steps": [],
        "web_searched": True,  # Already searched web
        "hallucination_retry_count": 1,
        "thread_id": "test_web_notice_session",
    }
    result = await apply_output_guardrails(state)
    assert result["faithfulness_passed"] is False
    assert "Verification Notice" in result["generation"]
    
    # Should route to end because web search was already executed
    next_route = route_after_output_guardrail(result)
    assert next_route == "end"
