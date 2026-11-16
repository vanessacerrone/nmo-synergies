"""The example notebook runs from start to end on the current library (about 35 s)."""
import pytest

from nmosyn.paths import EXTERNAL, ROOT

nbformat = pytest.importorskip('nbformat')
nbclient = pytest.importorskip('nbclient')


@pytest.mark.skipif(not (EXTERNAL / 'icecube_b4ritm').exists() or not (EXTERNAL / 'km3net_orca6').exists(),
                    reason='the DeepCore and ORCA6 releases are not downloaded (make data)')
def test_examples_notebook_runs():
    path = ROOT / 'examples' / 'examples.ipynb'
    nb = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, resources={'metadata': {'path': str(path.parent)}}).execute()
