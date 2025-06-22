import numerapi, json
from pathlib import Path

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
META_DIR = Path("meta"); META_DIR.mkdir(exist_ok=True)

def download_data(cache_dir: Path | str = DATA_DIR, version: str = 'v5.0', file_name: str = 'train') -> Path:
    """Download Numerai parquet once, then reuse from cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    dest = cache_dir / file_name

    if dest.exists():
        return dest

    napi = numerapi.NumerAPI()
    print(f"[data] downloading → {dest}")
    napi.download_dataset(f"{version}/{file_name}.parquet", str(dest))
    return dest

def get_feature_names(version: str = 'v5.0', set_name: str) -> list[str]:
    """Return list of columns for feature-set (small|medium|large)."""
    meta = META_DIR / f"{version}_features.json"
    if not meta.exists():
        napi = numerapi.NumerAPI()
        print(f"[feature set] downloading → {meta}")
        napi.download_dataset(f"{version}/features.json", str(meta))
    return json.load(open(meta))["feature_sets"][set_name]
