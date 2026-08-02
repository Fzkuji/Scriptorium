from pathlib import Path


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "docs" / "Model-Aligned-Wiki.html").is_file()
)


def test_portable_setup_files_exist_without_credentials():
    for name in ("requirements.txt", "requirements-dev.txt", "setup.sh"):
        assert (ROOT / name).is_file()
    text = (ROOT / "requirements.txt").read_text()
    assert "sk-" not in text
    assert "api_key" not in text.lower()


def test_canonical_layout_has_relative_compatibility_links():
    for name in (
        "src", "scripts", "tests", "benchmarks", "experiments", "figures",
        "gold_memory", "results", "third_party",
    ):
        link = ROOT / name
        assert link.is_symlink()
        assert link.resolve() == (ROOT / "code" / name).resolve()
    assert (ROOT / "docs" / "Model-Aligned-Wiki.html").is_file()
    assert (ROOT / "docs" / "experiments" / "experiment.html").is_file()
    assert not (ROOT / "Model-Aligned-Wiki.html").exists()
    assert not (ROOT / "experiment-plan.html").exists()
