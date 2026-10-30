import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
