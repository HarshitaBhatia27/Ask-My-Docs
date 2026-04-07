import os
import math
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, List

import ragas

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_core.prompt_values import PromptValue
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import LLMResult

from src.retriever import hybrid_search, rerank, ask
from src.chat_model import get_embeddings


# ── Force n=1 on every RAGAS eval call ───────────────────────────────────────
# RAGAS internally requests n=3 generations for some metrics.
# Groq rejects n>1 with a 400 error, so we override generate_text to cap it.
# agenerate_text is the async version — same fix needed there.

class GroqRagasWrapper(LangchainLLMWrapper):
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        return super().generate_text(
            prompt=prompt,
            n=1,           # force n=1 regardless of what RAGAS asks for
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        return await super().agenerate_text(
            prompt=prompt,
            n=1,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )


# Use llama-3.1-8b-instant for RAGAS scoring — it's faster and uses far fewer
# tokens. Your pipeline still uses 70B for actual answers. This separation
# keeps you well within Groq's free tier daily limit.
eval_groq = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1000,
)
ragas_llm = GroqRagasWrapper(eval_groq)
ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

faithfulness    = Faithfulness(llm=ragas_llm)
answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
context_precision = ContextPrecision(llm=ragas_llm)
context_recall   = ContextRecall(llm=ragas_llm)


# Questions from Sedgewick Algorithms 4th Ed — the PDF used for validation.
# ground_truth is hand-written, not LLM-generated.
TEST_QUESTIONS = [
    {
        "question": "What is dropout regularization?",
        "ground_truth": "Dropout randomly ignores neurons during training with probability p to prevent overfitting and improve generalization."
    },
    {
        "question": "What is the vanishing gradient problem?",
        "ground_truth": "The vanishing gradient problem occurs when gradients shrink as they propagate to lower layers during backpropagation, making those layers hard to train."
    },
    {
        "question": "What is the difference between bagging and boosting?",
        "ground_truth": "Bagging trains predictors in parallel on random subsets of data. Boosting trains them sequentially, each correcting the previous one's errors."
    },
    {
        "question": "What is a convolutional neural network?",
        "ground_truth": "A CNN is a neural network that uses convolutional layers to detect local patterns like edges and textures, commonly used for image tasks."
    },
    {
        "question": "What is the purpose of batch normalization?",
        "ground_truth": "Batch normalization normalizes layer inputs during training to speed up learning and reduce sensitivity to weight initialization."
    },
    {
        "question": "What is gradient descent?",
        "ground_truth": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that reduces the loss function."
    },
    {
        "question": "What is overfitting in machine learning?",
        "ground_truth": "Overfitting is when a model learns the training data too well including noise, and performs poorly on unseen data."
    },
]


def build_eval_dataset(test_cases: list) -> Dataset:
    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"\nRunning pipeline on {len(test_cases)} questions...\n")

    for i, item in enumerate(test_cases):
        q = item["question"]
        gt = item["ground_truth"]

        print(f"[{i+1}/{len(test_cases)}] {q}")

        retrieved_docs = hybrid_search(q, k=10)
        best_docs = rerank(q, retrieved_docs, top_k=4)

        context_strings = [
            f"[Page {doc.metadata.get('page', idx+1)}]: {doc.page_content}"
            for idx, doc in enumerate(best_docs)
        ]

        answer = ask(q)

        questions.append(q)
        answers.append(answer)
        contexts.append(context_strings)
        ground_truths.append(gt)

        print(f"    {answer[:90]}...")

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })


if __name__ == "__main__":

    from src.retriever import vector_store
    if vector_store is None:
        print("\nNo document indexed. Upload a PDF via the Streamlit app first.\n")
        exit(1)

    dataset = build_eval_dataset(TEST_QUESTIONS)

    print("\nRunning RAGAS evaluation...\n")
    results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

    df = pd.DataFrame(results.scores)
    mean_scores = df.mean(numeric_only=True)

    print("\n" + "=" * 52)
    print("  RAGAS RESULTS — AskMyDocs")
    print("=" * 52)

    labels = {
        "faithfulness":      "Faithfulness      (hallucination check)",
        "answer_relevancy":  "Answer Relevancy  (answers the question)",
        "context_precision": "Context Precision (chunks are useful)",
        "context_recall":    "Context Recall    (nothing important missed)",
    }

    for key, label in labels.items():
        if key in mean_scores:
            score = mean_scores[key]
            if score is not None and not math.isnan(score):
                bar = "█" * int(score * 20)
                print(f"  {label}: {score:.4f}  {bar}")
            else:
                print(f"  {label}: N/A (all eval calls failed — check rate limits)")
        else:
            print(f"  {label}: N/A")

    print("=" * 52)

    df.to_csv("ragas_results.csv", index=False)
    print("\nPer-question scores saved to ragas_results.csv")

    f_score = mean_scores.get("faithfulness")
    ar_score = mean_scores.get("answer_relevancy")
    if f_score and ar_score and not math.isnan(f_score) and not math.isnan(ar_score):
        print(
            f'\nResume bullet (copy this):\n'
            f'  "Evaluated pipeline using RAGAS — faithfulness: {f_score:.2f}, '
            f'answer relevancy: {ar_score:.2f};"'
        )