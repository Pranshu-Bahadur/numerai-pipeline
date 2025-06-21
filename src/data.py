import numerapi
from .config import DATA_DIR, DATA_VERSION

def download_data(cache_dir: Path | str = DATA_DIR, file_name: str = 'train') -> Path:
    """Download Numerai parquet once, then reuse from cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    dest = cache_dir / FILE_NAME

    if dest.exists():
        return dest

    api = numerapi.NumerAPI()
    print(f"[data] downloading → {dest}")
    api.download_dataset(f"{NUMERAI_VERSION}/{file_name}.parquet", dest)
    return dest
