from pathlib import Path

from src import config
from src.generate_data import generate_scientific_data


def test_generate_writes_file(tmp_path):
    # Run generator with small sample and ensure file is created
    generate_scientific_data(n_points=10)
    out = Path(config.raw_data_path())
    assert out.exists()
    # Clean up so test can be re-run cleanly
    try:
        out.unlink()
    except OSError:
        pass
