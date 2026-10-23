"""Check release metadata, version agreement, and distribution file contents."""

import argparse
import ast
import os
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    code = ast.parse((root / "src/pybdiff/__init__.py").read_text())
    version = next(ast.literal_eval(node.value) for node in code.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "__version__"
                           for t in node.targets))
    tag = os.environ.get("RELEASE_TAG")
    if tag and tag != f"v{version}":
        raise ValueError(f"Release tag {tag} does not match package version {version}")
    wheels, sources = list(args.dist.glob("*.whl")), list(args.dist.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Expected exactly one wheel and one sdist"
    expected = {f"pybdiff/{p.name}" for p in (root / "src/pybdiff").glob("*.py")}
    with zipfile.ZipFile(wheels[0]) as wheel:
        files = {n for n in wheel.namelist() if not n.endswith("/")}
        runtime = {n for n in files if n.startswith("pybdiff/")}
        assert runtime == expected, f"Unexpected runtime files: {runtime ^ expected}"
        assert all(n.startswith(("pybdiff/", f"pybdiff-{version}.dist-info/")) for n in files)
        info = BytesParser().parsebytes(wheel.read(f"pybdiff-{version}.dist-info/METADATA"))
        assert info["Name"] == "pybdiff" and info["Version"] == version
        assert info["License-Expression"] == "MIT"
        wheel_code = {n: wheel.read(n) for n in runtime}
    with tarfile.open(sources[0]) as source:
        members = {m.name.split("/", 1)[1]: m for m in source if m.isfile()}
        allowed_roots = {"src", "tests", "examples", "pyproject.toml", "README.md",
                         "CHANGELOG.md", "LICENSE", "PKG-INFO", ".gitignore"}
        assert all(Path(n).parts[0] in allowed_roots for n in members), members.keys()
        assert all(".." not in Path(n).parts and not Path(n).is_absolute() for n in members)
        for name in members:
            assert Path(name).suffix not in {".dta", ".log", ".png", ".pdf", ".pyc"}, name
        assert {"pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE"} <= members.keys()
        assert "examples/quickstart.py" in members and "tests/conftest.py" in members
        for name, content in wheel_code.items():
            assert source.extractfile(members[f"src/{name}"]).read() == content, name
        info = BytesParser().parsebytes(source.extractfile(members["PKG-INFO"]).read())
        assert info["Name"] == "pybdiff" and info["Version"] == version
    for path in (wheels[0], sources[0]):
        print(f"Validated {path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
