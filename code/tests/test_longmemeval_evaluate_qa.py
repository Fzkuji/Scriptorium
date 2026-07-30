import hashlib
import importlib.util
from pathlib import Path


def test_gateway_launcher_preserves_and_configures_official_evaluator():
    official = Path("benchmarks/longmemeval/src/evaluation/evaluate_qa.py")
    launcher = Path("scripts/run_longmemeval_gateway_eval.py")

    assert hashlib.sha256(official.read_bytes()).hexdigest() == (
        "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
    )
    assert launcher.is_file()
    spec = importlib.util.spec_from_file_location("gateway_eval", launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.patched_source(official.read_text())
    assert "openai_api_base = os.getenv('OPENAI_BASE_URL')" in source
    assert "metric_model = os.getenv('OPENAI_MODEL', metric_model)" in source
