"""Evaluate a previously trained LoRA adapter on a separate test slice.

Adapter weights are provided locally and are not bundled with this project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.pipeline import DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION, normalize_text


def select_held_out(train, test, *, count: int, start: int = 300) -> list[int]:
    """Exclude exact text repeats from training and previously viewed test rows."""
    excluded = {normalize_text(row["text"]) for row in train.select(range(min(2000, len(train))))}
    excluded.update(normalize_text(row["text"]) for row in test.select(range(min(205, len(test)))))
    chosen = []
    for index in range(start, len(test)):
        value = normalize_text(test[index]["text"])
        if not value or value in excluded:
            continue
        chosen.append(index)
        excluded.add(value)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Only {len(chosen)} unique held-out rows are available from index {start}")


def prior_prompt(article: str) -> str:
    """Use the prompt prefix paired with the previously trained adapter."""
    return (
        "<|system|>\n"
        "You are a headline generator. Create a short, catchy headline for the news article.\n"
        "Your headline must be in 1 line. Don't write anything else, only headline, no thanks, no agreement; ONLY HEADLINE.\n"
        "Don't write hash signs or quotanion marks.only headline.\n\n"
        f"<|user|>\nArticle: {article[:300]}\n<|assistant|>\n\nHeadline: "
    )


def metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer
    from sacrebleu import corpus_bleu

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    return {
        "bleu": round(float(corpus_bleu(predictions, [references]).score / 100), 4),
        "rouge1_f1": round(float(np.mean([score["rouge1"].fmeasure for score in scores])), 4),
        "rougeL_f1": round(float(np.mean([score["rougeL"].fmeasure for score in scores])), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("prior_adapter_results.json"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--train-arrow", type=Path)
    parser.add_argument("--test-arrow", type=Path)
    args = parser.parse_args()
    if (args.train_arrow is None) != (args.test_arrow is None):
        parser.error("--train-arrow and --test-arrow must be supplied together")

    import torch
    from datasets import Dataset, load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.train_arrow:
        train = Dataset.from_file(str(args.train_arrow))
        test = Dataset.from_file(str(args.test_arrow))
    else:
        dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)
        train, test = dataset["train"], dataset["test"]
    indices = select_held_out(train, test, count=args.count, start=args.start)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter_dir).to(device).eval()

    references = [test[index]["title"] for index in indices]
    predictions: dict[str, list[str]] = {"base": [], "prior_adapter": []}
    for position, index in enumerate(indices, start=1):
        inputs = tokenizer(prior_prompt(test[index]["text"]), return_tensors="pt").to(device)
        for name in predictions:
            with torch.inference_mode():
                if name == "base":
                    with model.disable_adapter():
                        output = model.generate(
                            **inputs, max_new_tokens=20, do_sample=False,
                            repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                        )
                else:
                    output = model.generate(
                        **inputs, max_new_tokens=20, do_sample=False,
                        repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                    )
            answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predictions[name].append(answer.splitlines()[0].strip().strip('"\''))
        if position % 10 == 0:
            print(f"Evaluated {position}/{len(indices)} rows", flush=True)

    result = {
        "task": "previously trained adapter evaluated on a separate test slice",
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "adapter_sha256": hashlib.sha256(
            (args.adapter_dir / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "test_rows": len(indices),
        "selected_test_indices": indices,
        "selection": "Starting at test index 300; exclude exact normalized-text repeats of train[:2000], test[:205], and earlier selected rows.",
        "decoding": "greedy, max_new_tokens=20, repetition_penalty=1.2",
        "models": {name: metrics(pred, references) for name, pred in predictions.items()},
        "caveat": "This adapter was selected using test[:200]; the evaluated slice was not used for checkpoint selection. Near duplicates were not screened.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
