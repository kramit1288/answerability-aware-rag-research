"""Resolve and download a CPython-compatible Phase 2 wheelhouse via PyPI JSON.

This exists because pip's simple-index transport is unreliable in the managed
Windows environment. Resolution uses wheel metadata and pip's vendored
``packaging`` implementation; installation is still performed by pip offline.
"""

from __future__ import annotations

import email
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict, deque
from pathlib import Path

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
from pip._vendor.packaging.version import InvalidVersion, Version


ROOT_REQUIREMENTS = {
    "numpy": "2.5.2",
    "scipy": "1.18.0",
    "pandas": "3.0.5",
    "torch": "2.13.0",
    "transformers": "5.15.0",
    "sentence-transformers": "5.7.0",
    "rank-bm25": "0.2.2",
    "pyarrow": "25.0.1",
    "scikit-learn": "1.9.0",
    "pytest": "9.1.1",
}


def pypi_json(name: str, version: str | None = None) -> dict:
    suffix = f"/{version}" if version else ""
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{name}{suffix}/json",
        headers={"User-Agent": "answerability-rag-phase02-resolver/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def compatible_wheel(files: list[dict], supported: set) -> dict | None:
    candidates = []
    for item in files:
        filename = item["filename"]
        if not filename.endswith(".whl"):
            continue
        try:
            _, _, _, tags = parse_wheel_filename(filename)
        except ValueError:
            continue
        matching = supported.intersection(tags)
        if matching:
            candidates.append((min(supported_order[tag] for tag in matching), filename, item))
    return min(candidates, default=(None, None, None))[2]


def select_release(name: str, constraints: list, supported: set) -> tuple[str, dict]:
    data = pypi_json(name)
    versions = []
    for raw in data["releases"]:
        try:
            parsed = Version(raw)
        except InvalidVersion:
            continue
        if not parsed.is_prerelease:
            versions.append((parsed, raw))
    versions.sort()
    for parsed, raw in reversed(versions):
        if all(parsed in specifier for specifier in constraints):
            wheel = compatible_wheel(data["releases"][raw], supported)
            if wheel:
                return raw, wheel
    raise RuntimeError(f"No compatible wheel for {name} satisfying {constraints}")


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "answerability-rag-phase02/1"})
    with urllib.request.urlopen(request, timeout=300) as response, path.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def wheel_requirements(path: Path) -> list[Requirement]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))
    requirements = []
    for value in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        requirements.append(requirement)
    return requirements


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    wheelhouse = root / ".cache" / "phase02_wheels"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    supported_list = list(sys_tags())
    supported = set(supported_list)
    supported_order = {tag: index for index, tag in enumerate(supported_list)}

    constraints = defaultdict(list)
    fixed = {canonicalize_name(name): version for name, version in ROOT_REQUIREMENTS.items()}
    for name, version in fixed.items():
        constraints[name].append(Requirement(f"{name}=={version}").specifier)

    selected: dict[str, tuple[str, dict, Path]] = {}
    processed: set[tuple[str, str]] = set()
    queue = deque(fixed)
    while queue:
        name = canonicalize_name(queue.popleft())
        version, wheel = select_release(name, constraints[name], supported)
        current = selected.get(name)
        if current and current[0] == version:
            continue
        path = wheelhouse / wheel["filename"]
        download(wheel["url"], path)
        selected[name] = (version, wheel, path)
        if (name, version) in processed:
            continue
        processed.add((name, version))
        print(f"resolved {name}=={version} -> {path.name}", flush=True)
        for requirement in wheel_requirements(path):
            dependency = canonicalize_name(requirement.name)
            constraints[dependency].append(requirement.specifier)
            queue.append(dependency)

    lock = {
        name: {"version": version, "filename": path.name, "sha256": wheel["digests"]["sha256"]}
        for name, (version, wheel, path) in sorted(selected.items())
    }
    lock_path = wheelhouse / "resolved.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {lock_path} with {len(lock)} packages")
