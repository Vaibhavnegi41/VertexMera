"""
Generator Evaluation using DeepEval
=====================================
Evaluates the VertexMera RAG generator using two metrics:
  - Faithfulness     : Is the generated answer grounded in the retrieved context?
  - AnswerRelevancy  : Does the generated answer actually address the question?

LLM Judge  : Groq (qwen/qwen3-8b) — same infra as the pipeline
Dataset    : tests/evals/generator_dataset.json
Run with   : python -m pytest tests/evals/test_generator_eval.py -v
          OR python tests/evals/test_generator_eval.py  (standalone)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ── DeepEval imports ──────────────────────────────────────────────────────────
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from langchain_groq import ChatGroq


# ── LLM Judge (Groq) ──────────────────────────────────────────────────────────
class GroqDeepEvalModel(DeepEvalBaseLLM):
    """Wraps ChatGroq so DeepEval can use it as an LLM judge."""

    def __init__(self):
        self.model = ChatGroq(
            model="qwen/qwen3.8-27b",
            api_key=os.getenv("GROQ_API_KEY", ""),
            temperature=0,
            max_tokens=512,
        )

    def get_model_name(self) -> str:
        return "groq/qwen3.8-27b"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        """Sync generate with retry on Groq rate-limit errors."""
        from groq import RateLimitError
        delay = 30
        for attempt in range(5):
            try:
                return self.model.invoke(prompt).content
            except RateLimitError as exc:
                if attempt == 4:
                    raise
                print(f"[WARN] Rate limit (sync), retrying in {delay}s … ({exc})")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    async def a_generate(self, prompt: str) -> str:
        """Async generate with retry on Groq rate-limit errors."""
        from groq import RateLimitError
        delay = 30
        for attempt in range(5):
            try:
                response = await self.model.ainvoke(prompt)
                return response.content
            except RateLimitError as exc:
                if attempt == 4:
                    raise
                print(f"[WARN] Rate limit (async), retrying in {delay}s … ({exc})")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)


# ── Generate an actual answer using the RAG pipeline ─────────────────────────
async def generate_answer(question: str, context: list[str]) -> str:
    """
    Calls the project's RAG generation model (rag_prompt + model) with the
    given question and pre-supplied context chunks, and returns the answer.
    """
    from backend.config import model
    from backend.prompts import rag_prompt

    context_str = "\n\n".join(context)
    response = await model.ainvoke(
        rag_prompt.format_messages(
            question=question,
            documents=context_str,
            chat_history=[]
        )
    )
    return response.content


# ── Build test cases ──────────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "generator_dataset.json"


async def build_test_cases() -> list[LLMTestCase]:
    """
    Loads samples from the JSON dataset, runs the generator for each sample,
    and returns a list of DeepEval LLMTestCase objects.

    Each test case has:
      input            – the user question
      actual_output    – the LLM's generated answer (from the pipeline)
      expected_output  – the reference answer (from the dataset)
      retrieval_context – the context chunks fed to the generator
    """
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n[INFO]  Loaded {len(dataset)} samples from {DATASET_PATH.name}")
    print("[INFO]  Generating answers via RAG pipeline ...\n")

    test_cases: list[LLMTestCase] = []

    for i, sample in enumerate(dataset, start=1):
        question: str = sample["input"]
        context: list[str] = sample["context"]
        expected_output: str = sample["expected_output"]

        print(f"  [{i}/{len(dataset)}] Generating answer for: {question[:70]}…")
        actual_output = await generate_answer(question, context)
        print(f"         Done. ({len(actual_output)} chars)\n")

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=context,
            )
        )

    return test_cases


# ── Metrics ───────────────────────────────────────────────────────────────────
def build_metrics(judge: GroqDeepEvalModel, threshold: float = 0.5):
    """Returns Faithfulness and AnswerRelevancy metrics using the Groq judge."""
    shared = dict(model=judge, threshold=threshold, include_reason=True)
    return [
        FaithfulnessMetric(**shared),
        AnswerRelevancyMetric(**shared),
    ]


# ── Results printer ───────────────────────────────────────────────────────────
def print_results(all_test_results):
    """Prints a formatted summary table of evaluation results."""
    sep = "─" * 90
    print(f"\n{'═' * 90}")
    print("  GENERATOR EVALUATION RESULTS  —  VertexMera RAG Pipeline")
    print(f"{'═' * 90}\n")

    overall_scores: dict[str, list[float]] = {}

    for idx, tc in enumerate(all_test_results, start=1):
        print(f"  Test Case {idx}: {tc.input[:65]}...")
        print(sep)
        for md in tc.metrics_data:
            score = md.score if md.score is not None else 0.0
            status = "[PASS]" if md.success else "[FAIL]"
            print(f"    {md.name:<30} score={score:.2f}   {status}")
            print(f"    {'':30} reason: {md.reason or '—'}")
            overall_scores.setdefault(md.name, []).append(score)
        print()

    print(f"{'═' * 90}")
    print("  AGGREGATE SCORES")
    print(sep)
    for name, scores in overall_scores.items():
        avg = sum(scores) / len(scores)
        print(f"    {name:<30} avg = {avg:.3f}   (over {len(scores)} cases)")
    print(f"{'═' * 90}\n")


# ── pytest entry point ────────────────────────────────────────────────────────
def test_generator_evaluation():
    """
    Entry point for pytest. Builds test cases, evaluates them one at a time
    (to respect Groq free-tier rate limits), and asserts all metrics pass.
    """
    judge = GroqDeepEvalModel()
    metrics = build_metrics(judge, threshold=0.5)

    test_cases = asyncio.run(build_test_cases())
    test_cases = test_cases[:3]  # cap at 3 to stay within Groq free-tier OTPM limits

    print("\n[INFO]  Running DeepEval generator metrics (sequential, 1 case at a time) ...\n")

    all_test_results = []
    for i, tc in enumerate(test_cases, start=1):
        print(f"[INFO]  Evaluating case {i}/{len(test_cases)}: {tc.input[:60]}...")
        result = evaluate(
            test_cases=[tc],
            metrics=metrics,
            async_config=AsyncConfig(run_async=False),
            cache_config=CacheConfig(write_cache=True, use_cache=True),
            display_config=DisplayConfig(print_results=False, inspect_after_run=False),
        )
        all_test_results.extend(result.test_results)
        if i < len(test_cases):
            print("[INFO]  Cooling down 30s before next case...")
            time.sleep(30)

    print_results(all_test_results)

    # Assert every metric passed
    for tc_result in all_test_results:
        for md in tc_result.metrics_data:
            assert md.success, (
                f"Metric '{md.name}' FAILED for input: '{tc_result.input}'\n"
                f"  Score  : {md.score}\n"
                f"  Reason : {md.reason}"
            )


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VertexMera — Generator Evaluation (DeepEval)")
    print("=" * 60)
    test_generator_evaluation()
