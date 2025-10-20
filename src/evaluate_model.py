import os
import time
import json
import random
import logging
import asyncio
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from ragas.llms import LangchainLLMWrapper  # ✅ to wrap your Groq LLM

from src.grok_chain import get_grok_chain
from langchain_groq import ChatGroq

# -----------------------------
# CONFIG
# -----------------------------
EVAL_FILE = "data/Data_ret.csv"
RESULTS_FILE = "data/eval_ragas_results.json"
MAX_SAMPLES = 50
SLEEP_MIN, SLEEP_MAX = 6, 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# LOAD DATASET
# -----------------------------
def load_eval_dataset():
    df = pd.read_csv(EVAL_FILE)
    df = df.sample(min(MAX_SAMPLES, len(df)), random_state=42)

    questions = []
    ground_truths = []
    contexts = []
    answers = []

    logger.info("🔧 Loading Groq RAG pipeline...")
    chain = get_grok_chain()

    for i, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row.get("Question", ""))
        gt = str(row.get("Value", ""))
        context = str(row.get("Context", ""))

        try:
            pred = chain.invoke({"question": q})
            questions.append(q)
            ground_truths.append(gt)
            contexts.append([context])
            answers.append(pred)

            logger.info(f"[{i}] ✅ Collected QA | Sleeping to avoid rate limit...")
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        except Exception as e:
            logger.error(f"[{i}] ❌ Error: {e}")
            continue

    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts,
        "answer": answers
    })

    return dataset

# -----------------------------
# RAGAS EVALUATION
# -----------------------------
async def run_ragas_evaluation(dataset):
    logger.info(f"🧮 Running RAGAS evaluation on {len(dataset)} samples...")

    # ✅ Use the same Groq model as your RAG pipeline for evaluation
    groq_llm = ChatGroq(
        temperature=0,
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Wrap it for RAGAS
    evaluator_llm = LangchainLLMWrapper(llm=groq_llm)

    results = await evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision],
        llm=evaluator_llm   # ✅ Pass evaluator here instead of custom metric
    )

    logger.info(f"✅ RAGAS Evaluation complete: {results}")
    return results

# -----------------------------
# SAVE RESULTS
# -----------------------------
def save_results(dataset, metrics):
    results_dict = {
        "num_samples": len(dataset),
        "metrics": metrics,
        "sample_predictions": []
    }

    for i in range(min(3, len(dataset))):
        results_dict["sample_predictions"].append({
            "question": dataset[i]["question"],
            "ground_truth": dataset[i]["ground_truth"],
            "answer": dataset[i]["answer"],
            "contexts": dataset[i]["contexts"]
        })

    with open(RESULTS_FILE, "w") as f:
        json.dump(results_dict, f, indent=4)

    logger.info(f"💾 RAGAS results saved to {RESULTS_FILE}")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    dataset = load_eval_dataset()
    metrics = asyncio.run(run_ragas_evaluation(dataset))
    save_results(dataset, metrics)

    print("\n📊 RAGAS Evaluation Summary")
    print(f" - Faithfulness:               {metrics['faithfulness']:.3f}")
    print(f" - Context Precision:          {metrics['context_precision']:.3f}")
