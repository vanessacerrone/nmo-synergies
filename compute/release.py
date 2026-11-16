"""Numbers read from the IceCube DeepCore public release (App. A.4, App. E, fig_deepcore_resolution and
fig_deepcore_energy_window).

The event totals of the sample: observed events in the analysis bins (data.tab), expected atmospheric muons (mc_mu.tab),
and their difference, the neutrino events to which the tabulated response is normalized. The energy and zenith
resolution derived from the nu_mu CC events of the release (nmosyn.experiments.deepcore.resolution), with the bin
variance removed (kernel_*) for the Gaussian response of fig_deepcore_resolution and fig_deepcore_energy_window. Output:
results/release.npz. Run from the repository root: python -m compute.release
"""
import numpy as np

from nmosyn.experiments import icfit
from nmosyn.experiments.deepcore import DeepCore
from nmosyn.paths import RESULTS


def main():
    fit = icfit.DeepCoreFit()
    resolution = DeepCore('numu').resolution
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'release.npz', observed=fit.observed.sum(), muons=fit.muons.sum(),
                        neutrinos=fit.observed.sum() - fit.muons.sum(), resolution_e=resolution.e,
                        resolution_lne=resolution.lne, resolution_cosz=resolution.cosz,
                        kernel_lne=resolution.sigma_lne(resolution.e, binned=False),
                        kernel_cosz=resolution.sigma_cosz(resolution.e, binned=False))


if __name__ == '__main__':
    main()
