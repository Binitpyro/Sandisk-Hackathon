"""
Embedding model fine-tuning utility for domain-specific corpora.

Generates (anchor, positive) training pairs from the indexed chunk data
and fine-tunes the bi-encoder embedding model using contrastive learning.
This improves retrieval accuracy on your specific document collection by
teaching the model which chunks are semantically related.

Usage (CLI)::

    python -m app.embeddings.finetune            # uses defaults
    python -m app.embeddings.finetune --epochs 3 --output models/finetuned

The script is designed to be run *offline* after a corpus has been indexed.
The fine-tuned model can then replace the default model by setting the
``EMBEDDING_MODEL`` environment variable or passing the path to
``EmbeddingService``.
"""

import argparse
import asyncio
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


async def _load_training_pairs(db_path: str = "pma_metadata.db") -> List[Tuple[str, str]]:
    """Build (anchor, positive) pairs from co-located chunks.

    Strategy: for every file with ≥ 2 chunks, pair each chunk with the
    *next* chunk in the same file (they share topical context).  This is a
    simple label-free approach from the LlamaIndex fine-tuning guide.
    """
    import aiosqlite

    pairs: List[Tuple[str, str]] = []
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(
            "SELECT DISTINCT file_id FROM chunks ORDER BY file_id"
        ) as cursor:
            file_ids = [row[0] for row in await cursor.fetchall()]

        for fid in file_ids:
            async with conn.execute(
                "SELECT text_preview FROM chunks WHERE file_id = ? "
                "ORDER BY start_offset",
                (fid,),
            ) as cursor:
                texts = [row[0] for row in await cursor.fetchall()]

            # Create adjacent-chunk pairs
            for i in range(len(texts) - 1):
                pairs.append((texts[i], texts[i + 1]))

    random.shuffle(pairs)
    return pairs


def finetune(
    base_model: str = "all-MiniLM-L6-v2",
    output_dir: str = "models/finetuned-embedding",
    db_path: str = "pma_metadata.db",
    epochs: int = 2,
    batch_size: int = 16,
    warmup_fraction: float = 0.1,
) -> str:
    """Fine-tune the embedding model on the indexed corpus.

    Parameters
    ----------
    base_model:
        HuggingFace model name or local path of the base bi-encoder.
    output_dir:
        Where to save the fine-tuned model.
    db_path:
        Path to the SQLite metadata database.
    epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    warmup_fraction:
        Fraction of total training steps used for learning-rate warmup.

    Returns
    -------
    str:
        Path to the saved fine-tuned model.
    """
    logger.info("Loading training pairs from %s...", db_path)
    pairs = asyncio.run(_load_training_pairs(db_path))

    if len(pairs) < 10:
        logger.warning(
            "Only %d training pairs found — need at least 10.  "
            "Index more documents before fine-tuning.",
            len(pairs),
        )
        return ""

    logger.info("Loaded %d training pairs.", len(pairs))

    # Convert to InputExample objects
    examples = [InputExample(texts=[a, b]) for a, b in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)  # type: ignore[arg-type]

    # Load the base model
    model = SentenceTransformer(base_model)

    # Use Multiple Negatives Ranking Loss (contrastive, no negatives needed)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(loader) * epochs * warmup_fraction)

    logger.info(
        "Fine-tuning '%s' for %d epochs (batch=%d, warmup=%d steps)…",
        base_model,
        epochs,
        batch_size,
        warmup_steps,
    )

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
    )

    logger.info("Fine-tuned model saved to: %s", output_dir)
    return output_dir


# ── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Fine-tune the embedding model on indexed documents."
    )
    parser.add_argument(
        "--base-model",
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="Base model name or path (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output",
        default="models/finetuned-embedding",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--db", default="pma_metadata.db", help="Path to metadata database"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    args = parser.parse_args()

    result = finetune(
        base_model=args.base_model,
        output_dir=args.output,
        db_path=args.db,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    if result:
        print(f"\nDone! Fine-tuned model saved to: {result}")
        print(
            "To use it, set:  EMBEDDING_MODEL={} or pass the path to "
            "EmbeddingService().".format(result)
        )
    else:
        print("\nFine-tuning skipped — not enough training data.")


if __name__ == "__main__":
    main()
