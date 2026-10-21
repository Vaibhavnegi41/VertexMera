"""
Retriever Evaluation using DeepEval
=====================================
Evaluates the VertexMera RAG retriever using three contextual metrics:
  - ContextualPrecision   : Are the retrieved chunks ranked usefully?
  - ContextualRecall      : Is the expected answer covered by the retrieved context?
  - ContextualRelevancy   : Is each retrieved chunk relevant to the query?

LLM Judge  : Groq (openai/gpt-oss-120b) — same model used in the pipeline
Dataset    : tests/evals/retriever_dataset.json
Run with   : python -m pytest tests/evals/test_retriever_eval.py -v
          OR python tests/evals/test_retriever_eval.py  (standalone)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ── DeepEval imports ──────────────────────────────────────────────────────────
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM

# ── LangChain / project imports ───────────────────────────────────────────────
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Custom DeepEval LLM wrapper around Groq
# ─────────────────────────────────────────────────────────────────────────────

class GroqDeepEvalModel(DeepEvalBaseLLM):
    """
    Wraps the project's ChatGroq instance so DeepEval can use it as its
    LLM judge without needing an OpenAI key.
    """

    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY", ""),
            temperature=0,
            max_tokens=1024,
        )

    def get_model_name(self) -> str:
        return "groq/openai/gpt-oss-120b"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Retriever helper — calls the actual Pinecone vector store
# ─────────────────────────────────────────────────────────────────────────────

async def retrieve_contexts(query: str, k: int = 4) -> list[str]:
    """
    Uses the project's configured vector store to retrieve the top-k
    document chunks for a given query and returns their page_content strings.
    """
    # Import lazily so heavy models are only loaded when needed
    from backend.config import vector_store  # noqa: PLC0415

    docs_with_scores = await vector_store.asimilarity_search_with_score(query, k=k)
    return [doc.page_content for doc, _score in docs_with_scores]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build test cases from the dataset JSON
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent / "retriever_dataset.json"


async def build_test_cases() -> list[LLMTestCase]:
    """
    Loads queries & expected outputs from the JSON dataset, runs the
    retriever for each query, and returns a list of DeepEval LLMTestCase
    objects ready for metric evaluation.
    """
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n📂  Loaded {len(dataset)} samples from {DATASET_PATH.name}")
    print("🔍  Retrieving contexts from Pinecone …\n")

    test_cases: list[LLMTestCase] = []

    for i, sample in enumerate(dataset, start=1):
        query: str = sample["input"]
        expected_output: str = sample["expected_output"]

        # Pre-populated context (if any) takes priority; otherwise live retrieval
        if sample.get("context"):
            retrieved_contexts: list[str] = sample["context"]
        else:
            retrieved_contexts = await retrieve_contexts(query)

        print(f"  [{i}/{len(dataset)}] Query : {query[:70]}…")
        print(f"         Chunks : {len(retrieved_contexts)} retrieved\n")

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=expected_output,   # retriever eval — no generation needed
                expected_output=expected_output,
                retrieval_context=retrieved_contexts,
            )
        )

    return test_cases


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics(judge: GroqDeepEvalModel, threshold: float = 0.5):
    """
    Returns the three contextual retriever metrics configured with the
    project's Groq judge and a pass/fail threshold.
    """
    shared_kwargs = dict(model=judge, threshold=threshold, include_reason=True)
    return [
        ContextualPrecisionMetric(**shared_kwargs),
        ContextualRecallMetric(**shared_kwargs),
        ContextualRelevancyMetric(**shared_kwargs),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pretty-print results
# ─────────────────────────────────────────────────────────────────────────────

def print_results(test_cases: list[LLMTestCase], metrics):
    """Prints a formatted summary table of all metric scores per test case."""
    metric_names = [m.__class__.__name__.replace("Metric", "") for m in metrics]

    separator = "─" * 90
    print(f"\n{'═' * 90}")
    print("  RETRIEVER EVALUATION RESULTS  —  VertexMera RAG Pipeline")
    print(f"{'═' * 90}\n")

    overall_scores: dict[str, list[float]] = {n: [] for n in metric_names}

    for idx, tc in enumerate(test_cases, start=1):
        print(f"  Test Case {idx}: {tc.input[:65]}…")
        print(separator)
        for metric, name in zip(metrics, metric_names):
            score = metric.score if hasattr(metric, "score") else "N/A"
            passed = metric.is_successful() if hasattr(metric, "is_successful") else "?"
            reason = getattr(metric, "reason", "—")
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"    {name:<25} score={score:.2f}   {status}")
            print(f"    {'':25} reason: {reason}")
            if isinstance(score, float):
                overall_scores[name].append(score)
        print()

    # ── Aggregate summary ──────────────────────────────────────────────────
    print(f"{'═' * 90}")
    print("  AGGREGATE SCORES")
    print(separator)
    for name, scores in overall_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"    {name:<25} avg = {avg:.3f}   (over {len(scores)} cases)")
    print(f"{'═' * 90}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  pytest-compatible test function  (also runnable as __main__)
# ─────────────────────────────────────────────────────────────────────────────

def test_retriever_evaluation():
    """
    Entry point recognised by pytest. Builds test cases, runs deepeval's
    evaluate(), and asserts that every test case passes all metrics.
    """
    judge = GroqDeepEvalModel()
    metrics = build_metrics(judge, threshold=0.5)

    # Build test cases synchronously (run the async helper in an event loop)
    test_cases = asyncio.run(build_test_cases())

    print("\n🧪  Running DeepEval retriever metrics …\n")
    results = evaluate(test_cases=test_cases, metrics=metrics)

    print_results(test_cases, metrics)

    # Assert all test cases passed so pytest reports failures correctly
    for tc in results.test_results:
        for metric_result in tc.metrics_data:
            assert metric_result.success, (
                f"Metric '{metric_result.name}' FAILED for input: '{tc.input}'\n"
                f"  Score  : {metric_result.score}\n"
                f"  Reason : {metric_result.reason}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Standalone execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VertexMera — Retriever Evaluation (DeepEval)")
    print("=" * 60)
    test_retriever_evaluation()
