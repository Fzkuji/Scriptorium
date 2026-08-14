from pathlib import Path

import pytest

from src.execution_platform import detect_platform, resolve_execution_profile


def test_detects_wsl_from_linux_proc_version(tmp_path):
    version = tmp_path / "version"
    version.write_text("Linux version 6.6.87.2-microsoft-standard-WSL2\n")

    assert detect_platform(
        os_name="posix", sys_platform="linux", proc_version=version
    ) == "wsl"


def test_detects_macos_without_proc_probe(tmp_path):
    assert detect_platform(
        os_name="posix",
        sys_platform="darwin",
        proc_version=tmp_path / "missing",
    ) == "macos"


@pytest.mark.parametrize("detected", ["wsl", "macos"])
def test_posix_profiles_resolve_auto_shell_to_bash(tmp_path, detected):
    profile = resolve_execution_profile(
        "auto",
        shell_backend="auto",
        output_dir=tmp_path,
        detected=detected,
    )

    assert profile.detected == detected
    assert profile.shell_backend == "posix-bash"


def test_explicit_profile_rejects_wrong_host(tmp_path):
    with pytest.raises(ValueError, match="does not match host"):
        resolve_execution_profile(
            "macos",
            shell_backend="auto",
            output_dir=tmp_path,
            detected="wsl",
        )


def test_explicit_shell_backend_is_preserved(tmp_path):
    profile = resolve_execution_profile(
        "wsl",
        shell_backend="native",
        output_dir=Path(tmp_path),
        detected="wsl",
    )

    assert profile.shell_backend == "native"
