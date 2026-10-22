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
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from langchain_groq import ChatGroq

class GroqDeepEvalModel(DeepEvalBaseLLM):
    """
    Wraps the project's ChatGroq instance so DeepEval can use it as its
    LLM judge without needing an OpenAI key.
    """

    def __init__(self):
        self.model = ChatGroq(
            model="qwen/qwen3.8-27b",
            api_key=os.getenv("GROQ_API_KEY", ""),
            temperature=0,
            max_tokens=512,   # verdicts are short JSON; keep well under OTPM limit
        )

    def get_model_name(self) -> str:
        return "groq/qwen3.8-27b"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        """Sync generate with exponential-backoff retry on 429 rate-limit errors."""
        from groq import RateLimitError
        delay = 30
        for attempt in range(5):
            try:
                return self.model.invoke(prompt).content
            except RateLimitError as exc:
                if attempt == 4:
                    raise
                print(f"[WARN] Rate limit hit (sync), waiting {delay}s … ({exc})")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    async def a_generate(self, prompt: str) -> str:
        """Async generate with exponential-backoff retry on 429 rate-limit errors."""
        from groq import RateLimitError
        delay = 30
        for attempt in range(5):
            try:
                response = await self.model.ainvoke(prompt)
                return response.content
            except RateLimitError as exc:
                if attempt == 4:
                    raise
                print(f"[WARN] Rate limit hit (async), waiting {delay}s … ({exc})")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)


async def retrieve_contexts(query: str, k: int = 4) -> list[str]:
    """
    Uses the project's configured vector store to retrieve the top-k
    document chunks for a given query and returns their page_content strings.
    """
   
    from backend.config import vector_store  

    docs_with_scores = await vector_store.asimilarity_search_with_score(query, k=k)
    return [doc.page_content for doc, _score in docs_with_scores]



# Build test cases from the dataset JSON
DATASET_PATH = Path(__file__).parent / "retriever_dataset.json"


async def build_test_cases() -> list[LLMTestCase]:
    """
    Loads queries & expected outputs from the JSON dataset, runs the
    retriever for each query, and returns a list of DeepEval LLMTestCase
    objects ready for metric evaluation.
    """
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n[INFO]  Loaded {len(dataset)} samples from {DATASET_PATH.name}")
    print("[INFO]  Retrieving contexts from Pinecone ...\n")

    test_cases: list[LLMTestCase] = []

    for i, sample in enumerate(dataset, start=1):
        query: str = sample["input"]
        expected_output: str = sample["expected_output"]
        if sample.get("context"):
            retrieved_contexts: list[str] = sample["context"]
        else:
            retrieved_contexts = await retrieve_contexts(query)

        print(f"  [{i}/{len(dataset)}] Query : {query[:70]}…")
        print(f"         Chunks : {len(retrieved_contexts)} retrieved\n")

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=expected_output,   
                expected_output=expected_output,
                retrieval_context=retrieved_contexts,
            )
        )

    return test_cases



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
            status = "[PASS]" if passed else "[FAIL]"
            print(f"    {name:<25} score={score:.2f}   {status}")
            print(f"    {'':25} reason: {reason}")
            if isinstance(score, float):
                overall_scores[name].append(score)
        print()

    print(f"{'═' * 90}")
    print("  AGGREGATE SCORES")
    print(separator)
    for name, scores in overall_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"    {name:<25} avg = {avg:.3f}   (over {len(scores)} cases)")
    print(f"{'═' * 90}\n")


def test_retriever_evaluation():
    """
    Entry point recognised by pytest. Builds test cases, runs deepeval's
    evaluate(), and asserts that every test case passes all metrics.
    Evaluates one test case at a time to stay within Groq's free-tier
    OTPM (output tokens per minute) rate limit.
    """
    judge = GroqDeepEvalModel()
    metrics = build_metrics(judge, threshold=0.5)

    test_cases = asyncio.run(build_test_cases())
    test_cases = test_cases[:3]  # cap at 3 to stay within Groq free-tier OTPM limits

    print("\n[INFO]  Running DeepEval retriever metrics (sequential, 1 case at a time) ...\n")

    all_test_results = []
    for i, tc in enumerate(test_cases, start=1):
        print(f"[INFO]  Evaluating case {i}/{len(test_cases)}: {tc.input[:60]}...")
        result = evaluate(
            test_cases=[tc],
            metrics=metrics,
            async_config=AsyncConfig(run_async=True, max_concurrent=1),  # no parallel LLM bursts
            cache_config=CacheConfig(write_cache=True, use_cache=True),   # skip on re-runs
            display_config=DisplayConfig(print_results=False, inspect_after_run=False),
        )
        all_test_results.extend(result.test_results)
        if i < len(test_cases):
            print(f"[INFO]  Cooling down 30s before next case...")
            time.sleep(30)

    print_results(test_cases, metrics)

    for tc_result in all_test_results:
        for metric_result in tc_result.metrics_data:
            assert metric_result.success, (
                f"Metric '{metric_result.name}' FAILED for input: '{tc_result.input}'\n"
                f"  Score  : {metric_result.score}\n"
                f"  Reason : {metric_result.reason}"
            )




if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VertexMera — Retriever Evaluation (DeepEval)")
    print("=" * 60)
    test_retriever_evaluation()
