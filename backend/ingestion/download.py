"""Dataset download and caching.

Downloads the specified dataset (e.g., ai4bharat/MSMARCO-XI)
from Hugging Face and saves it locally for processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset, Dataset

from backend.utils.logging import get_logger

logger = get_logger(__name__)


def download_dataset(
    dataset_name: str,
    split: str,
    language: str,
    output_dir: Path,
    max_samples: int | None = None,
) -> Path:
    """Download a dataset from Hugging Face and save to disk.

    Args:
        dataset_name: Name of the dataset on Hugging Face.
        split: Dataset split to download (e.g., 'train').
        language: Language subset to download (e.g., 'en').
        output_dir: Directory to save the downloaded data.
        max_samples: Optional limit on number of samples.

    Returns:
        Path to the saved JSONL file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a deterministic filename based on parameters
    sample_suffix = f"_{max_samples}" if max_samples else ""
    safe_name = dataset_name.replace("/", "_")
    output_file = output_dir / f"{safe_name}_{language}_{split}{sample_suffix}.jsonl"
    
    if output_file.exists():
        logger.info("dataset_already_downloaded", path=str(output_file))
        return output_file

    logger.info(
        "downloading_dataset",
        dataset=dataset_name,
        language=language,
        split=split,
        max_samples=max_samples,
    )

    try:
        # Load dataset
        try:
            ds = load_dataset(dataset_name, language, split=split, streaming=True)
        except ValueError:
            logger.info("language_subset_not_found_falling_back_to_default", language=language)
            ds = load_dataset(dataset_name, split=split, streaming=True)
            
        from backend.config.settings import get_settings
        settings = get_settings()
        
        # In this specific test env, HF Datasets Server throws 500 TooBigRowGroupsError for MSMARCO-XI.
        # If demo mode is active, we can allow synthetic data if dataset is unreachable.
        # Otherwise, we STRICTLY fail unless we use the pre-downloaded duckdb subset.
        
        if not isinstance(ds, Dataset):
            # If it's a DatasetDict, try to get the requested split
            if hasattr(ds, '__getitem__') and split in ds: # type: ignore
                ds = ds[split] # type: ignore
            else:
                raise ValueError(f"Could not extract split {split} from dataset")
                
        # Subsample if requested (using streaming take)
        if max_samples:
            logger.info("subsampling_dataset", target=max_samples)
            # Use fixed seed for reproducibility
            ds = ds.shuffle(seed=42, buffer_size=1000).take(max_samples)
            
        # Convert to list of dicts for JSONL writing
        records = []
        for item in ds:
            # We want to extract 'query', 'passage', 'id' depending on dataset structure.
            # MSMARCO-XI typically has query_id, query, passage_id, passage.
            # Normalizing structure to have id, text, and metadata
            
            # Simple heuristic mapping for typical HF datasets
            record: dict[str, Any] = {"metadata": {}}
            
            # We strictly map MSMARCO-XI data
            if "query" in item and "passages" in item:
                # MSMARCO-XI real schema
                # passages is a dict of lists
                passages = item["passages"].get("Translated_passages", [])
                if not passages:
                    continue
                    
                record["id"] = str(item.get("query_id", len(records)))
                record["text"] = str(passages[0])
                record["metadata"]["query"] = str(item["query"])
                record["metadata"]["query_id"] = str(item.get("query_id", ""))
            elif "query" in item and "passage" in item:
                # Other MSMARCO-style datasets
                record["id"] = str(item.get("passage_id", item.get("id", len(records))))
                record["text"] = str(item["passage"])
                record["metadata"]["query"] = str(item["query"])
                record["metadata"]["query_id"] = str(item.get("query_id", ""))
            else:
                if not settings.demo_mode:
                    raise RuntimeError("Dataset does not match MSMARCO schema. Cannot synthesize data when DEMO_MODE is False.")
                
                # Fallback, just dump everything
                record["id"] = str(item.get("id", len(records)))
                # Try to find a reasonable text field
                for k, v in item.items():
                    if isinstance(v, str) and len(v) > 50:
                        record["text"] = v
                        break
                else:
                    record["text"] = json.dumps(item)
                    
            record["metadata"]["language"] = language
            
            # Keep original fields in metadata
            for k, v in item.items():
                if k not in ["text", "passage", "passages"] and k not in record["metadata"]:
                    record["metadata"][k] = v
                    
            records.append(record)
            
        # Write to JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
        logger.info("dataset_download_complete", path=str(output_file), count=len(records))
        return output_file
        
    except Exception as e:
        logger.error("dataset_download_failed", error=str(e))
        raise
