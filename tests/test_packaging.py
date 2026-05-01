import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def test_triton_dependencies_are_platform_specific():
    dependencies = _project_metadata()["dependencies"]

    assert "triton-windows>=3.6,<3.7; platform_system == 'Windows'" in dependencies
    assert "triton>=3.6,<3.7; platform_system == 'Linux'" in dependencies
    assert not any("triton; platform_system != 'Windows'" in dep for dep in dependencies)


def test_supported_os_classifiers_match_triton_markers():
    classifiers = set(_project_metadata()["classifiers"])

    assert "Operating System :: Microsoft :: Windows" in classifiers
    assert "Operating System :: POSIX :: Linux" in classifiers
