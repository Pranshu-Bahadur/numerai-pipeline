from src import data

def test_download(tmp_path):
    p = data.download_data(tmp_path)
    # file exists and is non-empty
    assert p.exists() and p.stat().st_size > 0

def test_feature_list_small():
    feats = data.get_feature_names("v5.0", "small")
    # basic sanity: non-empty and column names are strings
    assert isinstance(feats, list) and len(feats) > 0
    assert all(isinstance(f, str) for f in feats)

