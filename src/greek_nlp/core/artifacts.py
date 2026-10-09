import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STORE = Path(os.getenv("MODELS_DIR", ROOT / "models_store"))


def ensure_artifact(name: str, repo_env: str) -> Path:
    """Επιστρέφει τον τοπικό φάκελο. Αν λείπει, τον κατεβάζει από το Hugging Face Hub."""
    local = STORE / name
    if local.exists():
        return local

    repo_id = os.getenv(repo_env)
    if not repo_id:
        raise FileNotFoundError(f"Λείπει ο φάκελος {local} και δεν έχει οριστεί η {repo_env}")

    from huggingface_hub import snapshot_download

    repo_type = "dataset" if name == "rag_index" else "model"
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local,
        token=os.getenv("HF_TOKEN"),
    )
