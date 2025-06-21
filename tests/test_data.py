from src import data


def test_download(tmp_path):
    p = data.download_data(tmp_path)
    # file exists and is non-empty
    assert p.exists() and p.stat().st_size > 0

