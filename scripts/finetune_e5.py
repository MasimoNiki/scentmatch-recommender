"""Fine-tune multilingual-e5-small on ScentMatch hard-negative n-tuples."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--output-dir", type=Path, default=Path("models/e5_finetuned"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import torch
        from datasets import load_dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.training_args import BatchSamplers
    except ImportError as error:
        raise RuntimeError(
            'Training dependencies are missing. Run: pip install -e ".[ml]"'
        ) from error

    project_dir = args.project_dir.resolve()
    data_dir = project_dir / "data" / "training"
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train_triplets_v1.jsonl"),
            "validation": str(data_dir / "validation_triplets_v1.jsonl"),
        },
    )
    model = SentenceTransformer(args.model)
    loss = losses.MultipleNegativesRankingLoss(model)
    output_dir = args.output_dir.resolve()
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        run_name="scentmatch-e5-hard-negatives-v1",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(str(output_dir / "final"))
    print(f"Fine-tuned model: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
