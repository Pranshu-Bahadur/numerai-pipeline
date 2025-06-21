import numerapi
from pathlib import Path
from .config import DATA_DIR, DATA_VERSION

def download_data(cache_dir: Path | str = DATA_DIR, file_name: str = 'train') -> Path:
    """Download Numerai parquet once, then reuse from cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    dest = cache_dir / file_name

    if dest.exists():
        return dest

    api = numerapi.NumerAPI()
    print(f"[data] downloading → {dest}")
    api.download_dataset(f"{DATA_VERSION}/{file_name}.parquet", str(dest))
    return dest
