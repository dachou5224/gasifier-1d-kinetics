import importlib.util
import os


def _load_module():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../diagnostics/diagnose_cell0.py")
    )
    spec = importlib.util.spec_from_file_location("diagnose_cell0", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diagnose_cell0_loader_supports_canonical_case_name():
    module = _load_module()
    case_data, coal_db = module.load_case_data("Texaco_I-1")

    assert case_data["inputs"]["coal"] == "texaco i-1_Coal"
    assert case_data["inputs"]["FeedRate"] == 76.66
    assert "texaco i-1_Coal" in coal_db
