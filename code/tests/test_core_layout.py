from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


def test_current_core_is_imported_without_a_version_namespace():
    from src import (
        BuildConfig,
        MemoryConfig,
        MemoryWorkspace,
        QueryConfig,
        build_memory,
        collect_answer,
    )

    assert BuildConfig.__module__ == "src.build"
    assert MemoryConfig.__module__.startswith("src.management")
    assert MemoryWorkspace.__module__.startswith("src.management")
    assert QueryConfig.__module__ == "src.retrieval.config"
    assert callable(build_memory)
    assert callable(collect_answer)


def test_src_contains_only_core_packages():
    source = CODE_ROOT / "src"
    assert not (source / "adapters").exists()
    for filename in (
        "chatgpt_proxy.py",
        "openai_gpt55_flex_gateway.py",
        "openrouter_gpt4o_mini_gateway.py",
    ):
        assert not (source / filename).exists()


def test_locked_evaluator_uses_the_scripts_evaluation_compatibility_link():
    compatibility = CODE_ROOT / "src" / "evaluation"
    assert compatibility.is_symlink()
    assert compatibility.resolve() == (CODE_ROOT / "scripts" / "evaluation").resolve()

    from src.evaluation.answerer import generate_answer

    assert callable(generate_answer)


def test_no_active_version_router_or_old_native_memory_modules():
    for relative in (
        "src/nativemem.py",
        "src/v8_memory.py",
        "src/v10_memory.py",
        "src/nativemem_versions",
        "src/legacy",
        "scripts/adapters/run_nativemem.py",
    ):
        assert not (CODE_ROOT / relative).exists()


def test_experiments_directory_is_not_part_of_the_code_layout():
    repository = CODE_ROOT.parent
    assert not (CODE_ROOT / "experiments").exists()
    assert not (repository / "experiments").exists()
    assert (CODE_ROOT / "scripts" / "configs").is_dir()
    assert (CODE_ROOT / "results" / "analysis").is_dir()
