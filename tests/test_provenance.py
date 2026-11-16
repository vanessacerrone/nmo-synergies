"""The checksums of data/PROVENANCE.md agree with those that scripts/download_data.py verifies, and with the
tables derived from the releases that the repository stores."""
import hashlib
import importlib.util
import re

from nmosyn.paths import DATA, ROOT


def _sources() -> dict:
    spec = importlib.util.spec_from_file_location('download_data', ROOT / 'scripts' / 'download_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SOURCES


def test_provenance_checksums():
    sections = (DATA / 'PROVENANCE.md').read_text().split('\n### ')[1:]
    for key, release in _sources().items():
        [text] = [s for s in sections if f'(`{key}/`)' in s.splitlines()[0]]
        archive = getattr(release, 'archive_sha256', None)
        rows = re.findall(r'^\| `([^`]+)` \|.*\| `([0-9a-f]{64})` \|$', text, re.M)
        assert all(release.sha256.get(name) == digest for name, digest in rows), key
        assert set(re.findall(r'[0-9a-f]{64}', text)) <= {*release.sha256.values(), archive}, key
        assert rows or archive, key
        assert archive is None or archive in text, key


def test_derived_tables_checksums():
    sections = (DATA / 'PROVENANCE.md').read_text().split('\n## Derived tables\n')[1].split('\n### ')[1:]
    for folder in ('genie_ch2_xsec', 'genie_water_xsec'):
        [text] = [s for s in sections if f'(`data/{folder}/`)' in s.splitlines()[0]]
        rows = re.findall(r'^\| `([^`]+)` \|.*\| `([0-9a-f]{64})` \|$', text, re.M)
        assert len(rows) == 2, folder
        for name, digest in rows:
            assert hashlib.sha256((DATA / folder / name).read_bytes()).hexdigest() == digest, (folder, name)
