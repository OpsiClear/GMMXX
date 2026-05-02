import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def test_triton_dependencies_are_platform_specific():
    # triton is now an optional extra, not a hard dependency
    project = _project_metadata()
    optional_deps = project.get("optional-dependencies", {})
    triton_deps = optional_deps.get("triton", [])

    assert "triton-windows>=3.6,<3.7; platform_system == 'Windows'" in triton_deps
    assert "triton>=3.6,<3.7; platform_system == 'Linux'" in triton_deps
    assert not any("triton; platform_system != 'Windows'" in dep for dep in triton_deps)

    # triton must NOT be in hard dependencies
    hard_deps = project.get("dependencies", [])
    assert not any("triton" in dep for dep in hard_deps)


def test_supported_os_classifiers_match_triton_markers():
    classifiers = set(_project_metadata()["classifiers"])

    assert "Operating System :: Microsoft :: Windows" in classifiers
    assert "Operating System :: POSIX :: Linux" in classifiers
