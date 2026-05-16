"""
HuggingFace Dataset-based ChromaDB persistence.

Strategy:
  - On startup  → pull latest chroma_db snapshot from HF Dataset repo (if configured)
  - On shutdown → push chroma_db snapshot back to HF Dataset repo
  - This gives cross-restart persistence even on ephemeral HF Spaces (free tier)

If HF_DATASET_REPO is not set, falls back to local-only (data lost on restart).
"""
import os
import shutil
import tarfile
from pathlib import Path

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

SNAPSHOT_FILENAME = "chroma_db_snapshot.tar.gz"


def pull_db_from_hf() -> bool:
    """Download and extract the ChromaDB snapshot from HF Dataset repo."""
    if not settings.USE_HF_PERSISTENCE or not settings.HF_DATASET_REPO:
        logger.info("HF persistence disabled — using local ChromaDB only.")
        return False

    try:
        from huggingface_hub import hf_hub_download, HfApi

        logger.info(f"Pulling ChromaDB snapshot from HF: {settings.HF_DATASET_REPO}")

        local_tar = hf_hub_download(
            repo_id=settings.HF_DATASET_REPO,
            filename=SNAPSHOT_FILENAME,
            repo_type="dataset",
            token=settings.HF_TOKEN or None,
        )

        # Extract into CHROMA_PERSIST_DIR
        persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=str(persist_dir.parent))

        logger.info("✅ ChromaDB snapshot restored from HF Dataset.")
        return True

    except Exception as e:
        logger.warning(f"Could not pull snapshot (fresh start): {e}")
        return False


def push_db_to_hf() -> bool:
    """Compress and upload the ChromaDB directory to HF Dataset repo."""
    if not settings.USE_HF_PERSISTENCE or not settings.HF_DATASET_REPO:
        return False

    try:
        from huggingface_hub import HfApi

        persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        if not persist_dir.exists():
            logger.warning("ChromaDB dir not found — nothing to push.")
            return False

        tar_path = f"/tmp/{SNAPSHOT_FILENAME}"
        logger.info("Compressing ChromaDB for upload...")

        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(persist_dir), arcname=persist_dir.name)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=tar_path,
            path_in_repo=SNAPSHOT_FILENAME,
            repo_id=settings.HF_DATASET_REPO,
            repo_type="dataset",
            token=settings.HF_TOKEN,
            commit_message="chore: update ChromaDB snapshot",
        )

        logger.info("✅ ChromaDB snapshot pushed to HF Dataset.")
        return True

    except Exception as e:
        logger.error(f"Failed to push snapshot to HF: {e}")
        return False
