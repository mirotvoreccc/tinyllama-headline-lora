"""LoRA training and held-out evaluation for TinyLlama headline generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
DATASET_ID = "azminetoushikwasi/news-headlines-CC-FF"
DATASET_REVISION = "0b3d4744cfcfba544baea13733c923fdfd0b5fc3"
SYSTEM = "Write one concise headline for the news article. Return only the headline."


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def unique_indices(rows, count: int, seen: set[str], *, start: int = 0) -> list[int]:
    """Take the earliest non-empty, unseen article texts, updating seen."""
    indices = []
    for index in range(start, len(rows)):
        value = normalize_text(rows[index]["text"])
        if not value or value in seen:
            continue
        seen.add(value)
        indices.append(index)
        if len(indices) == count:
            return indices
    raise ValueError(f"Only {len(indices)} unique rows available; requested {count}")


def prompt_for(tokenizer, article: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Article: {article[:300]}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_example(tokenizer, article: str, headline: str, max_length: int = 512) -> dict:
    """Mask prompt tokens so training loss is calculated on the headline only."""
    prompt = tokenizer.encode(prompt_for(tokenizer, article), add_special_tokens=True)
    answer = tokenizer.encode(headline.strip(), add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(answer) >= max_length:
        answer = answer[: max_length - 1] + [tokenizer.eos_token_id]
    # Preserve the assistant-generation marker at the end of a long prompt.
    prompt = prompt[-(max_length - len(answer)) :] if len(answer) < max_length else []
    ids = prompt + answer
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt) + answer,
    }


class HeadlineCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]):
        import torch

        longest = max(len(x["input_ids"]) for x in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in examples:
            pad = longest - len(example["input_ids"])
            batch["input_ids"].append(example["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(example["attention_mask"] + [0] * pad)
            batch["labels"].append(example["labels"] + [-100] * pad)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def load_splits(train_count: int, validation_count: int, test_count: int,
                train_arrow: Path | None = None, test_arrow: Path | None = None,
                test_start: int = 500):
    """Use separate HF splits and remove exact text repeats across all parts."""
    from datasets import Dataset, load_dataset

    if (train_arrow is None) != (test_arrow is None):
        raise ValueError("train_arrow and test_arrow must be supplied together")
    if train_arrow is not None:
        train = Dataset.from_file(str(train_arrow))
        test = Dataset.from_file(str(test_arrow))
    else:
        dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)
        train, test = dataset["train"], dataset["test"]
    seen: set[str] = set()
    train_indices = unique_indices(train, train_count, seen)
    validation_indices = unique_indices(train, validation_count, seen, start=train_indices[-1] + 1)
    test_indices = unique_indices(test, test_count, seen, start=test_start)
    return (
        train.select(train_indices),
        train.select(validation_indices),
        test.select(test_indices),
    )


def train(args) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    train_data, validation_data, _ = load_splits(
        args.train_count, args.validation_count, args.test_count,
        args.train_arrow, args.test_arrow,
    )
    encode = lambda item: encode_example(tokenizer, item["text"], item["title"])
    tokenized_train = train_data.map(
        encode, remove_columns=train_data.column_names, keep_in_memory=True
    )
    tokenized_validation = validation_data.map(
        encode, remove_columns=validation_data.column_names, keep_in_memory=True
    )

    dtype = torch.float16 if torch.backends.mps.is_available() or torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    arguments = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        eval_strategy="steps",
        eval_steps=max(1, min(50, args.max_steps)),
        save_strategy="steps",
        save_steps=max(1, min(50, args.max_steps)),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=HeadlineCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    adapter_path = args.output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    (args.output_dir / "training_info.json").write_text(
        json.dumps(
            {
                "base_model": MODEL_ID,
                "base_model_revision": MODEL_REVISION,
                "dataset": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "train_rows": len(train_data),
                "validation_rows": len(validation_data),
                "test_rows_reserved": args.test_count,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "gradient_accumulation_steps": 4,
                "maximum_training_example_exposures": args.max_steps * 4,
                "best_validation_loss": trainer.state.best_metric,
            },
            indent=2,
        ) + "\n"
    )


def evaluate(args) -> None:
    import torch
    from peft import PeftModel
    from rouge_score import rouge_scorer
    from sacrebleu import corpus_bleu
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, _, test_data = load_splits(
        args.train_count, args.validation_count, args.test_count,
        args.train_arrow, args.test_arrow, args.test_start,
    )
    adapter_path = args.output_dir / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = PeftModel.from_pretrained(base, adapter_path)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    generated: dict[str, list[str]] = {"base": [], "trained_adapter": []}
    references = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    for index, item in enumerate(test_data, start=1):
        prompt_ids = tokenizer(prompt_for(tokenizer, item["text"]), return_tensors="pt").to(device)
        for name in generated:
            with torch.inference_mode():
                if name == "base":
                    with model.disable_adapter():
                        output = model.generate(**prompt_ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                else:
                    output = model.generate(**prompt_ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(output[0, prompt_ids.input_ids.shape[1] :], skip_special_tokens=True)
            generated[name].append(answer.splitlines()[0].strip())
        references.append(item["title"])
        if index % 10 == 0 or index == len(test_data):
            print(f"Evaluated {index}/{len(test_data)} test articles", flush=True)
    def scores(predictions: list[str]) -> dict[str, float]:
        rouge = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
        return {
            "bleu": round(float(corpus_bleu(predictions, [references]).score / 100), 4),
            "rouge1_f1": round(float(np.mean([score["rouge1"].fmeasure for score in rouge])), 4),
            "rougeL_f1": round(float(np.mean([score["rougeL"].fmeasure for score in rouge])), 4),
        }
    result = {
        "test_rows": len(test_data),
        "test_start_index": args.test_start,
        "base_model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "models": {name: scores(predictions) for name, predictions in generated.items()},
        "decoding": f"greedy, max_new_tokens={args.max_new_tokens}",
    }
    (args.output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--test-start", type=int, default=500,
                        help="First original test index eligible for evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-arrow", type=Path)
    parser.add_argument("--test-arrow", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
