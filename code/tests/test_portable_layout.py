from pathlib import Path


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "docs" / "Model-Aligned-Wiki.html").is_file()
)

CODE_DIRECTORIES = (
    "src", "scripts", "tests", "benchmarks", "figures",
    "gold_memory", "results", "baselines", "scriptorium",
)


def test_portable_setup_files_exist_without_credentials():
    for name in ("requirements.txt", "requirements-dev.txt", "setup.sh"):
        assert (ROOT / name).is_file()
    text = (ROOT / "requirements.txt").read_text()
    assert "sk-" not in text
    assert "api_key" not in text.lower()


def test_the_repository_root_holds_no_second_copy_of_the_code_tree():
    """Every working directory lives under code/, and only there.

    The root once carried a symlink per directory, left from a layout where
    they were top level. Two ways to name one directory is one too many: a
    reader cannot tell which tree is real, and a path in a config or a
    document can mean either.
    """
    for name in CODE_DIRECTORIES:
        assert (ROOT / "code" / name).is_dir()
        assert not (ROOT / name).exists()
    assert not any(child.is_symlink() for child in ROOT.iterdir())


def test_third_party_checkouts_sit_beside_the_adapters_that_drive_them():
    assert (ROOT / "code" / "baselines" / "third_party").is_dir()
    assert not (ROOT / "third_party").exists()


def test_documents_are_where_the_entry_page_says():
    assert not (ROOT / "experiments").exists()
    assert not (ROOT / "code" / "experiments").exists()
    assert (ROOT / "docs" / "Model-Aligned-Wiki.html").is_file()
    assert (ROOT / "docs" / "experiments" / "experiment.html").is_file()
    assert not (ROOT / "Model-Aligned-Wiki.html").exists()
    assert not (ROOT / "experiment-plan.html").exists()
