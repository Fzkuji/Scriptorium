import importlib.util
import inspect
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "experiment_llm_self_review_longmemeval.py"


def test_pilot_selection_and_variants_are_generic():
    assert SCRIPT.exists()
    spec = importlib.util.spec_from_file_location(
        "experiment_llm_self_review_longmemeval", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.RETRIEVAL_ERROR_INDICES == (
        83, 109, 141, 153, 166, 299, 326, 389, 460,
    )
    assert module.CONTROL_INDICES == (7, 159, 193, 246, 373, 449)
    assert set(module.VARIANTS) == {
        "reflective-prompt",
        "same-model-review",
        "countercheck-prompt",
        "conservative-review",
    }
    assert module.VARIANTS["reflective-prompt"]["review_prompt"] is None
    assert "missing or conflicting memory" in module.REFLECTIVE_PROMPT
    assert "missing or conflicting memory" in module.VARIANTS[
        "same-model-review"
    ]["review_prompt"]
    for forbidden in (
        "multi-session", "temporal-reasoning", "knowledge-update",
        "single-session-preference",
    ):
        assert forbidden not in module.REFLECTIVE_PROMPT
        assert forbidden not in module.VARIANTS["same-model-review"][
            "review_prompt"
        ]
        assert forbidden not in module.COUNTERCHECK_ADDITION
        assert forbidden not in module.CONSERVATIVE_REVIEW_PROMPT
    assert module.VARIANTS["countercheck-prompt"] == {
        "prompt_template": module.lme.LME_SINGLE_PROMPT
        + module.COUNTERCHECK_ADDITION,
        "review_prompt": None,
    }
    assert module.VARIANTS["conservative-review"] == {
        "prompt_template": module.lme.LME_SINGLE_PROMPT,
        "review_prompt": module.CONSERVATIVE_REVIEW_PROMPT,
    }
    assert inspect.signature(module.install_variant).parameters[
        "pilot_only"
    ].default is True
