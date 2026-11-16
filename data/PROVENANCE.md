# Data provenance

This file documents the public releases that the computation reads: what each one is, where it comes from,
its terms of use, the files used with their SHA-256 checksums, and where the paper uses it. Figure, table, and
appendix numbers refer to the paper.

## Public releases

The public releases are not stored in the repository.

    python scripts/download_data.py            # fetch into data/external/ (about 84 MB)
    python scripts/download_data.py --check    # verify the files on disk, download nothing
    python scripts/download_data.py genie_xsec_2_12_10    # opt-in, 498 MiB: the input of the CH2 and water tables

The script fetches each release into its folder under `data/external/` and checks the SHA-256 of every file
as stored, after decompression or extraction. Files already present with the right checksum are not
downloaded again, so a second run resumes an interrupted one. The tables below give the checksums of the files
that the computation reads and of the archives; the script holds the checksum of every file it fetches, and
`tests/test_provenance.py` checks that the two agree. All URLs were checked on 2026-10-19.

| folder | release | license or terms | used in |
|---|---|---|---|
| `icecube_b4ritm/` | IceCube DeepCore golden event sample, doi:10.7910/DVN/B4RITM | CC0 1.0 | App. A.4, App. E; fig_deepcore_bin_weights, fig_deepcore_dcp, fig_deepcore_resolution, fig_deepcore_energy_window, fig_deepcore_vs_orca6, fig_full_fit_validation, fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, fig_published_best_fits, fig_synergy_vs_precision, fig_synergy_rings, fig_deepcore_analytic, and fig_deepcore_reproduction |
| `icecube_u20mmb/` | IceCube DeepCore 9.3-year likelihood maps, doi:10.7910/DVN/U20MMB | CC0 1.0 | fig_published_best_fits |
| `km3net_orca6/` | KM3NeT/ORCA6 433 kt-yr open data v0.5 | CC BY 4.0 | App. A.5, fig_deepcore_vs_orca6 |
| `honda/` | Honda et al. 2014 flux tables, South Pole and Fréjus | none stated | App. A.4, A.5 and every DeepCore and ORCA6 result |
| `nova_2024/` | NOvA 2024 data release, doi:10.5281/zenodo.21284589 | CC BY 4.0 | App. A.3; fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, fig_published_best_fits, fig_synergy_vs_precision, and fig_synergy_rings |
| `nova_flux_2017/` | NOvA near-detector flux 2017, doi:10.15484/1959359 | none stated | App. A.3; fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, fig_published_best_fits, fig_synergy_vs_precision, and fig_synergy_rings |
| `t2k_flux_2020/` | T2K neutrino beam flux prediction 2020 | none stated (© T2K) | App. A.3; fig_solutions_plane and fig_intercept_ranges |
| `dune_globes_xsec/` | GENIE cross sections of the DUNE TDR GLoBES configuration, arXiv:2103.04797 | arXiv non-exclusive distribution license | energy grid of `data/genie_ch2_xsec/` and `data/genie_water_xsec/` |
| `genie_xsec_2_12_10/` | GENIE 2.12.10 pre-computed cross-section splines, DefaultPlusValenciaMEC (opt-in) | none stated (GENIE code: GPLv3) | input of `data/genie_ch2_xsec/` (App. A.3, NOvA) and `data/genie_water_xsec/` (App. A.3, T2K) |
| `icecube_prd108/` | IceCube PRD 108, 012014, arXiv:2304.12236v2 (the paper, for its Fig. 26) | CC BY 4.0 | fig_deepcore_reproduction |

### IceCube DeepCore golden event sample (`icecube_b4ritm/`)

**What.** The public release of the IceCube DeepCore measurement of atmospheric neutrino mixing with the
high-purity "golden" event sample. The four `mc_nu*` files list the simulated neutrino events one by one:
flavor and charge, interaction, true and reconstructed energy and direction, particle identification, and a
weight in GeV cm² sr that gives the expected number of events when multiplied by the flux and the livetime.
The reconstructed variables are given at the centers of the analysis bins. `data.tab`
holds the observed counts per analysis bin, `mc_mu.tab` the expected atmospheric-muon background per bin, and
the three `hs_*` files the hypersurfaces that correct the expected counts per bin for the detector
systematic parameters.

**Citation.** IceCube Collaboration, *Replication Data for: Measurement of atmospheric neutrino mixing with
improved IceCube DeepCore calibration and data processing*, Harvard Dataverse, version 1.0 (2025),
[doi:10.7910/DVN/B4RITM](https://doi.org/10.7910/DVN/B4RITM). Analysis: IceCube Collaboration,
Phys. Rev. D 108, 012014 (2023), arXiv:2304.12236, doi:10.1103/PhysRevD.108.012014.

**License.** CC0 1.0 (Dataverse dataset terms).

**Retrieval.** Dataverse access API, `https://dataverse.harvard.edu/api/access/datafile/<id>`, file ids in
`scripts/download_data.py`. Dataverse ingests the uploaded `.csv` tables and serves them by default as
tab-separated `.tab` files, which are the files used; the checksums refer to this format. Size 63.8 MB, of
which `mc_numu_cc.tab` is 44.6 MB. Also fetched: `readme.md`, the description of the release. The notebook
`example.ipynb` of the release is not fetched.

| file | content | SHA-256 |
|---|---|---|
| `mc_numu_cc.tab` | simulated $\nu_\mu$ CC events | `0354af4398b5d02004ea0b6631ca81a348bb5d08b406b2670b0abeadaac2bb53` |
| `mc_nue_cc.tab` | simulated $\nu_e$ CC events | `e7040158d7d4194e0d080c653651f4cbe7c7ba78d8a03731032c4f9751d58fc3` |
| `mc_nutau_cc.tab` | simulated $\nu_\tau$ CC events | `5558c64dc43acd9c9ad0c0751a1c5d28fef3f3b4d3cc4c632d47c176022627e5` |
| `mc_nu_nc.tab` | simulated NC events | `4ea029d862cb0c67bd0d704af437ebcbda80ed6ac8c57e5db6364c595004e89e` |
| `data.tab` | observed counts per analysis bin | `c1b9baa1bc53f41bb43a4daf7cbeb953c2c6b74c8e13e59afb5764274413671a` |
| `mc_mu.tab` | expected atmospheric muons per bin | `af11edbfe0b14f76d5fcb7912f857dd331956deeef3efcba388b17ec3f67a5fd` |
| `hs_nu_nc_nue_cc.tab` | hypersurfaces, NC and $\nu_e$ CC | `b60aad105805818191ecc8bdacef920fe662a39a7b64f1b17c5d18ccbbf96dd6` |
| `hs_numu_cc.tab` | hypersurfaces, $\nu_\mu$ CC | `e392b277091b98b47ece769fe50976130da5e0e3a0ca1be5ec1d53bec2568ab9` |
| `hs_nutau_cc.tab` | hypersurfaces, $\nu_\tau$ CC | `55e1608d05cd5b27891ec052762431b3b0b764b61062de9414712f582e920067` |

**Use in the paper.**
- The four `mc_nu*` files give the tabulated detector response of App. A.4 (Eq. A.3): the acceptance and
  the energy and zenith migration tables. This response enters every DeepCore result: fig_deepcore_bin_weights, fig_deepcore_dcp, fig_deepcore_resolution, fig_deepcore_energy_window, fig_deepcore_vs_orca6, fig_full_fit_validation, fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, fig_published_best_fits, fig_synergy_vs_precision, fig_synergy_rings, and fig_deepcore_analytic,
  and the DeepCore numbers of Secs. 2, 4, 5, 6 and App. D. The energy resolution of fig_deepcore_resolution and fig_deepcore_energy_window is extracted from
  the same events, with the variance of the analysis bins removed.
- The normalization to 21 402 neutrino events (App. A.4) is the 21 914 observed events of `data.tab`
  minus the 512 expected muons of `mc_mu.tab`.
- The nine data files enter the full fit of App. E: fig_full_fit_validation (full fit and counts of the full fitter), fig_published_best_fits
  (fit to the data, open circle), fig_deepcore_reproduction, and the fits to the data of App. E.3 and E.4.

### IceCube DeepCore 9.3-year likelihood maps (`icecube_u20mmb/`)

**What.** The $-2\Delta\ln L$ maps of the IceCube DeepCore measurement with convolutional neural networks
and 9.3 years of data, on a grid of 39 values of $\sin^2\theta_{23}$ times 29 (normal ordering) or 31
(inverted ordering) values of $\Delta m^2_{32}$, with the best fit of each ordering. Each `.npy` file holds a
pickled Python dictionary (read with `numpy.load(path, allow_pickle=True).item()`).

**Citation.** IceCube Collaboration, *Data release for neutrino oscillation parameters using convolutional
neural networks with 9.3 years of data in IceCube DeepCore*, Harvard Dataverse, version 2.0 (2024),
[doi:10.7910/DVN/U20MMB](https://doi.org/10.7910/DVN/U20MMB). Analysis: IceCube Collaboration, Phys. Rev.
Lett. 134, 091801 (2025), arXiv:2405.02163, doi:10.1103/PhysRevLett.134.091801.

**License.** CC0 1.0 (Dataverse dataset terms).

**Retrieval.** Dataverse access API, file ids in `scripts/download_data.py`. Size 60 kB. Also fetched:
`saved_no_delta_LLH_map.npy` (the normal-ordering map), the release's `README.txt`, stored as
`README_icecube_9.3y.txt`, and `script.py`, the example reader of the release.

| file | content | SHA-256 |
|---|---|---|
| `saved_io_delta_LLH_map.npy` | inverted-ordering map and best fit | `103bf82f163c5f3c4fb9278b762aebd23bca083481fa703e273dee890664d48b` |

**Use in the paper.** fig_published_best_fits, pentagon: the inverted-ordering best fit of $\Delta m^2_{32}$ and its
$1\sigma$ interval ($-2\Delta\ln L = 1$), profiled over $\sin^2\theta_{23}$ on the inverted-ordering map and
converted to $\Delta m^2_{31}$ as stated in the caption.

### KM3NeT/ORCA6 open data (`km3net_orca6/`)

**What.** The binned response of KM3NeT/ORCA with six detection units, 433 kt-yr. The tree
`binned_nu_response` gives, for every neutrino flavor, charge, and interaction, the expected weight per unit
flux of each true cell $(E, \cos\theta_z)$ in each reconstructed bin $(E, \cos\theta_z, \text{class})$, with
three classes: high-purity tracks, showers, and low-purity tracks. The trees `binned_muon` and `binned_data`
hold the atmospheric-muon background and the observed counts, and the axes `E_true_axis`, `Ct_true_axis`,
`E_reco_axis`, and `Ct_reco_axis` the true and the reconstructed binning. The release prescribes the
azimuth-averaged Honda flux at the Fréjus site, without mountain, at solar minimum (`honda/` below).

**Citation.** KM3NeT Collaboration, *Oscillation Analysis ORCA433kt-y / ORCA 6* (ORCA6 433 kt-y neutrino
dataset v0.5), KM3NeT Open Data Centre, dataset version 2.3 (2025),
<https://opendata.km3net.de/dataset.xhtml?persistentId=doi:10.5072/FK2/Y0UXVW>. Analysis: KM3NeT
Collaboration, JHEP 10 (2024) 206 [Erratum: JHEP 10 (2025) 041], arXiv:2408.07015,
doi:10.1007/JHEP10(2024)206. The persistent identifier of the dataset uses the DataCite test prefix 10.5072
and does not resolve at doi.org, so the dataset is cited by its URL.

**License.** CC BY 4.0 (dataset terms and README of the release).

**Retrieval.** Dataverse access API of the KM3NeT Open Data Centre,
`https://opendata.km3net.de/api/access/datafile/<id>`, file ids in `scripts/download_data.py`. Size 14.4 MB.
Also fetched, as documentation of the release: `README.md`, `bins_433kt-y_v0.5.json` (the binning in plain
text), and `chi2_landscape_NO.csv` and `chi2_landscape_IO.csv` (the published $\chi^2$ landscapes).

| file | content | SHA-256 |
|---|---|---|
| `ORCA6_433kt-y_opendata_v0.5.root` | response, muon background, observed counts, axes | `79453657367c8b8582f6ec384837e6d0ac3ce62fc45378984d4d6d33c58441c3` |

**Use in the paper.** The ROOT file gives the ORCA6 counts of App. A.5: fig_deepcore_vs_orca6 (all classes and high-purity
tracks), the weights of the three event classes and their effect on the intercept, and the 0.7% agreement of the
predicted and observed totals.

### Honda atmospheric neutrino flux (`honda/`)

**What.** The HKKM 2014 one-year-average flux tables of $\nu_\mu$, $\bar\nu_\mu$, $\nu_e$, $\bar\nu_e$,
averaged over azimuth (20 bins of $\cos\theta_z$, one azimuth bin: `20-01`), at solar minimum, without
mountain over the detector, in (m² s sr GeV)⁻¹. `spl` is the South Pole, `frj` the Fréjus site (sections 2.6
and 2.4.a of the page).

**Citation.** M. Honda, M. Sajjad Athar, T. Kajita, K. Kasahara, and S. Midorikawa, Phys. Rev. D 92, 023004
(2015), arXiv:1502.03916, doi:10.1103/PhysRevD.92.023004. Tables:
<http://www-rccn.icrr.u-tokyo.ac.jp/mhonda/public/nflx2014/index.html>.

**License.** None stated on the page.

**Retrieval.** `<name>.d.gz` from the page above, decompressed on download. The host serves the tables over
plain http only; the checksums refer to the decompressed files.

| file | content | SHA-256 |
|---|---|---|
| `spl-ally-20-01-solmin.d` | South Pole | `fb26dccb15317d2ca1d7ba54af1cfa17a188127c8f74f7f7c23c5beb96b3fc83` |
| `frj-ally-20-01-solmin.d` | Fréjus | `e9397af19d58514600e646b14c6449ac2bdea972476dda6d288db375b52131fa` |

**Use in the paper.** The South Pole table is the flux of every DeepCore result (App. A.4, Eq. A.3, the
migration tables, and the full fit of App. E). The Fréjus table is the flux of ORCA6 (App. A.5, fig_deepcore_vs_orca6).

### NOvA 2024 data release (`nova_2024/`)

**What.** The data release of the NOvA analysis of $\nu_e$ appearance and $\nu_\mu$ disappearance with
$26.61\times10^{20}$ protons on target in neutrino mode and $12.5\times10^{20}$ in antineutrino mode: data
histograms of all far-detector samples, the predictions of every signal and background component at the
NOvA best fit (the $\nu_\mu$ samples also without oscillations), and credible-interval contours.

**Citation.** NOvA Collaboration, *NOvA 2024 official data release (26.61E20 neutrino + 12.5E20
antineutrino)*, FERMILAB-DATA-2025-05, Zenodo, version of 2026-07-09,
[doi:10.5281/zenodo.21284589](https://doi.org/10.5281/zenodo.21284589) (all versions:
doi:10.5281/zenodo.17822358). The NOvA data-release page, <https://novaexperiment.fnal.gov/data-releases/>,
also links it as Fermilab public document 594. Analysis: NOvA Collaboration, Phys. Rev. Lett. 136, 011802
(2026), arXiv:2509.04361, doi:10.1103/x53y-2b86.

**License.** CC BY 4.0 (Zenodo record).

**Retrieval.** `data_release_2024.zip` (1.3 MB) from the Zenodo record, unpacked without its folder
`data_release_2024/`. Archive SHA-256 `9ef7b65dda776590c07209f4c5a9879c41a9cc08677b3016189d87b57b365858`.
Also unpacked: `README.md`, `NOvA_2024_data_histograms.root` (the observed far-detector samples), and the
six `contours_*.root` files (credible intervals).

| file | content | SHA-256 |
|---|---|---|
| `NOvA_2024_prediction_with_systs_histograms.root` | predictions by component, no-oscillation $\nu_\mu$ samples | `35be5a68210fec77072ce864b0c6bdd3655c2829526b1ce327614f24c70233f9` |

**Use in the paper.** `NOvA_2024_prediction_with_systs_histograms.root` gives the no-oscillation
$\nu_\mu$-like spectra in reconstructed energy and the $\nu_\mu$-like backgrounds of the NOvA far-detector
spectra (App. A.3), which enter the NOvA bands and lines of fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, fig_published_best_fits, fig_synergy_vs_precision, and fig_synergy_rings.

### NOvA near-detector flux 2017 (`nova_flux_2017/`)

**What.** The NuMI flux at the NOvA near detector for the 2017 analysis, per flavor ($\nu_\mu$,
$\bar\nu_\mu$, $\nu_e$, $\bar\nu_e$) and horn current (FHC, RHC), with its fractional uncertainty from hadron
production and beam focusing. The document states that the fluxes are integrated over the fiducial volume of
the near detector. The text tables give neutrinos per m² per $10^6$ protons on target in each energy bin;
the ROOT files hold the same fluxes as histograms, also divided by the bin width.

**Citation.** NOvA Collaboration, *Flux Histograms for 2017 Analysis*, FERMILAB-DATA-2017-01,
[doi:10.15484/1959359](https://doi.org/10.15484/1959359), Fermilab public document 8, version 1,
<https://publicdocs.fnal.gov/cgi-bin/ShowDocument?docid=8>.

**License.** None stated.

**Retrieval.** `https://publicdocs.fnal.gov/cgi-bin/RetrieveFile?docid=8&version=1&filename=<file>`.
Size 28 kB. The server sits behind a bot challenge that sometimes answers HTTP 403; the script retries after a
wait, and a second run fetches whatever is still missing. Also fetched: `FHC_Flux_NOvA_ND_2017.root` and
`RHC_Flux_NOvA_ND_2017.root`.

| file | content | SHA-256 |
|---|---|---|
| `FHC_Flux_numu_NOvA_ND_2017.txt` | FHC $\nu_\mu$ | `c9477c7517e5269361994b8a39eccb2c54732920d1917e538339b1758a19c505` |
| `FHC_Flux_numubar_NOvA_ND_2017.txt` | FHC $\bar\nu_\mu$ | `d429ccc573469a924815c248b521bd2aea396f8e46770b5884d3c5292c5be959` |
| `FHC_Flux_nue_NOvA_ND_2017.txt` | FHC $\nu_e$ | `0e80cd4362f5e391039c43a31c18a124a0d7c1f97685c2ea50731ee29f59a56b` |
| `FHC_Flux_nuebar_NOvA_ND_2017.txt` | FHC $\bar\nu_e$ | `d483e1a5c90e78066197c1b405cb2627c18f3aff13e4b291782faf3ed65d5e55` |
| `RHC_Flux_numu_NOvA_ND_2017.txt` | RHC $\nu_\mu$ | `edd8b876ed37d1f6f535d00f4434b7b93a8156212f9f151729aabf0bc1e59ed5` |
| `RHC_Flux_numubar_NOvA_ND_2017.txt` | RHC $\bar\nu_\mu$ | `7ba49a5a3a2fdadd7fe095be6cb260c48c45328a9e4a9220d18cbe07942c664b` |
| `RHC_Flux_nue_NOvA_ND_2017.txt` | RHC $\nu_e$ | `0e3085fd495294dbf3e17dbdb32b9b29b0b5aab846cff08119b4443a913f81af` |
| `RHC_Flux_nuebar_NOvA_ND_2017.txt` | RHC $\bar\nu_e$ | `1622dce9549da4bec50f63a018a19cac6450a01599b4f326511173382c8a7590` |

**Use in the paper.** The eight text tables, with the CH2 cross sections of `data/genie_ch2_xsec/`, split the NOvA
no-oscillation $\nu_\mu$-like spectra into $\nu_\mu$ and $\bar\nu_\mu$ and give the shape of the beam $\nu_e$
(App. A.3), in the NOvA bands and lines of fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, fig_published_best_fits, fig_synergy_vs_precision, and fig_synergy_rings.

### T2K neutrino beam flux prediction 2020 (`t2k_flux_2020/`)

**What.** The T2K 2020 flux prediction without oscillations at the near detector ND280 and at
Super-Kamiokande, for +250 kA (FHC) and −250 kA (RHC) horn current, for the nominal beam and for the actual
run conditions, as text tables and ROOT histograms, with the covariance of the energy bins.

**Citation.** T2K Collaboration, *Neutrino beam flux prediction 2020* (2021),
<https://t2k-experiment.org/results/neutrino-beam-flux-prediction-2020/>. The page states that this is the
flux of the T2K 2020 oscillation analysis, "to be submitted to Eur. Phys. J. C"; that analysis later appeared
as T2K Collaboration, Eur. Phys. J. C 83, 782 (2023), arXiv:2303.03222, doi:10.1140/epjc/s10052-023-11819-x.

**Terms.** No license stated; the page states © The T2K Collaboration 2021, all rights reserved.

**Retrieval.** `https://t2k-experiment.org/wp-content/uploads/T2Kflux2020.gz`, a gzip-compressed tar file
(277 kB), unpacked without its folder `t2kflux_2020_public_release/`. Archive SHA-256
`0c8e921ede08508d192cb203ebfb6386d15fbc7ab6ad099ebfd503cae12a8569`. The macOS metadata entry `._README.pdf`
of the archive is skipped. Also unpacked: `README.pdf`, the covariance `t2kflux_2020_covariance.root`, the
eight ROOT histogram files and ND280 tables, and the two Super-Kamiokande tables for the run conditions.

| file | content | SHA-256 |
|---|---|---|
| `t2kflux_2020_plus250kA_nominal_sk.txt` | Super-Kamiokande, FHC, nominal | `cf70149d8f03ae9dc9a629a2808686dd1dfcdca63b3ee0c9ea7557143c2d0b58` |
| `t2kflux_2020_minus250kA_nominal_sk.txt` | Super-Kamiokande, RHC, nominal | `2379f71da48ebffee6f73c984e1de298173e51d5d9495f299279d2ef299f8b73` |

**Use in the paper.** The two nominal Super-Kamiokande tables are the flux of the T2K far-detector spectra
(App. A.3), in the T2K bands of fig_solutions_plane and fig_intercept_ranges.

### DUNE GLoBES cross sections (`dune_globes_xsec/`)

**What.** The GENIE 2.12 charged-current and neutral-current cross sections of the GLoBES configuration
that approximates the DUNE Technical Design Report, in the GLoBES format: $\log_{10}(E/\text{GeV})$ and
$\sigma/E$ for $\nu_e$, $\nu_\mu$, $\nu_\tau$, $\bar\nu_e$, $\bar\nu_\mu$, $\bar\nu_\tau$.

**Citation.** DUNE Collaboration, *Experiment Simulation Configurations Approximating DUNE TDR*,
arXiv:2103.04797, ancillary files `dune_globes/xsec/` of version 2.

**License.** arXiv non-exclusive license to distribute; no separate license for the ancillary files.

**Retrieval.** `https://arxiv.org/src/2103.04797v2/anc/dune_globes/xsec/<file>`. Size 7 kB.

| file | content | SHA-256 |
|---|---|---|
| `xsec_cc.dat` | charged current | `bb950e7fb645bc5cdb052d17e2076534be2cbfd24e83ddc851f6d906ece93efd` |
| `xsec_nc.dat` | neutral current | `613cb9005dae60e028e429f125b537b11bce3f7aaa389d6dc329f4a9ce7f28d9` |

**Use in the paper.** None directly: `scripts/make_ch2_xsec.py` and `scripts/make_water_xsec.py` take from
`xsec_cc.dat` the 75 values of $\log_{10}(E/\text{GeV})$ of the derived tables `data/genie_ch2_xsec/` (NOvA) and
`data/genie_water_xsec/` (T2K), and print the ratio to these cross sections as a check.

### GENIE 2.12.10 cross-section splines (`genie_xsec_2_12_10/`)

**What.** The pre-computed cross-section splines of the neutrino event generator GENIE 2.12.10 with the
"Default+CCMEC" list of processes (Valencia model of meson-exchange currents): per neutrino flavor, target
isotope, and process, the total cross section per nucleus on 500 knots from 0.01 to 400 GeV, logarithmically
spaced, in natural units (GeV$^{-2}$). `gxspl-FNALsmall.xml` (856 MB) holds the six flavors on 44 isotopes,
among them H1 (`tgt:1000010010`), C12 (`tgt:1000060120`), and O16 (`tgt:1000080160`); `README` lists the files of the archive.

**Citation.** C. Andreopoulos et al., *The GENIE Neutrino Monte Carlo Generator*, Nucl. Instrum. Meth. A 614, 87
(2010), arXiv:0905.2517, doi:10.1016/j.nima.2009.12.009. The meson-exchange-current splines
(`NievesSimoVacasMECPXSec2016`) are the Valencia model, J. Nieves, I. Ruiz Simo, and M. J. Vicente Vacas, Phys.
Rev. C 83, 045501 (2011), arXiv:1102.2777, doi:10.1103/PhysRevC.83.045501, as implemented in GENIE by J. Schwehr,
D. Cherdack, and R. Gran, arXiv:1601.02038. Splines: <https://scisoft.fnal.gov/scisoft/packages/genie_xsec/v2_12_10/>.

**Terms.** Neither the download page nor the README of the archive states a license. GENIE 2.12.10 itself is
distributed under GPLv3 (file `LICENSE` at tag `R-2_12_10` of <https://github.com/GENIE-MC/Generator>), which covers
the output of a program only when the output is itself a covered work. The file `CITATION` at the same tag endorses
the MCnet guidelines for fair academic use and asks every derivative work to cite Andreopoulos et al. (2010). The
splines are not stored in the repository.

**Retrieval.** Opt-in: `python scripts/download_data.py genie_xsec_2_12_10` fetches
`genie_xsec-2.12.10-noarch-DefaultPlusValenciaMEC.tar.bz2` (522 012 367 bytes, 498 MiB) and unpacks
`gxspl-FNALsmall.xml` and `README` without the folder `genie_xsec/v2_12_10/NULL/DefaultPlusValenciaMEC/data/`.
Archive SHA-256 `fe1b584e7014bba6c4cba5646e1031f344e9efbf799a2aa26b706e28c40a4481`. The other files of the archive
(`gxspl-FNALbig.xml.gz` with 113 isotopes, `gxspl-freenuc.xml.gz`, `xsec_graphs.root`, configuration files) are
not unpacked. The plain run of the script skips this release: only `scripts/make_ch2_xsec.py` and
`scripts/make_water_xsec.py` read it, and their output is in the repository.

| file | content | SHA-256 |
|---|---|---|
| `gxspl-FNALsmall.xml` | splines, 6 flavors, 44 isotopes | `c28d6e6596d0b67b900b32e8dfdaf48ebf8fc938b942c1db0374f2f27b26539a` |
| `README` | content of the archive | `fc41d8b89f553279d7bc78a907888598eed5d5c6b3a08ccef42bf9da50bd347e` |

**Use in the paper.** None directly: the input of the CH2 and water cross sections of the NOvA and T2K spectra (below).

### NuFast-Earth source code (`nufast_earth/`)

**What.** The C++ sources of NuFast-Earth at commit `327586dddcb8c01282feb7560d029b63ea1257f4` (22 September
2026): the engine that `nmosyn/osc.py` translates. Not used to compute any result; `tests/test_nufast_earth.py`
compiles it and checks `nmosyn/osc.py` against it.

**Citation.** P. B. Denton and S. J. Parke, *NuFast-Earth: Efficient Atmospheric, Solar, and Supernova Neutrino
Propagation Through the Earth*, Phys. Rev. D, arXiv:2511.04735; https://github.com/PeterDenton/NuFast-Earth.

**License.** MIT, copyright (c) 2025 Peter B. Denton (file `LICENSE`, also reproduced in `THIRD_PARTY_LICENSES`).

**Retrieval.** `https://raw.githubusercontent.com/PeterDenton/NuFast-Earth/<commit>/<path>`. Size 60 kB.

| file | content | SHA-256 |
|---|---|---|
| `NuFastEarth.cpp` | probability engine | `a7ff0f58beec38af22b418653be267f7c999c570d651b7da5541ea340bdf088e` |
| `Oscillation.cpp` | eigenvalues, layer amplitudes, rotation | `03ed2663f22eea78c27968c6aa781aba457cb7974e8de957e618cb6947d1e9d9` |
| `Earth.cpp` | density profiles | `fc0e6c2e9d6b3e6006740ce4f10b3c5dac131fdc5e39f690e1edc5cad330a368` |
| `Geometry.cpp` | paths through the Earth | `799fb9b91a5508260e3e6fcedeed602c1ffbbbd7736190eb75eb35b5fe94362a` |
| `Solar.cpp` | solar density | `757717f07a0963b2854cdc89c4855d7da65bfc3a4b21a60687f4c63832971315` |
| `NuFastEarth.h` | header | `6c47861b038e7b2408868bec1e3ec83d9bc54572b7b58cbba651b9ebe51d9dfb` |
| `Earth.h` | header | `7e1030e911cca1c042a1420af81b4b5dae15e88193d954da26359e317a9cb4cc` |
| `Matrix.h` | header | `77eca8a4bb92f729ab6f2f1066c45fb57e573c3245e09223b1ede6b8d0bf1447` |
| `Geometry.h` | header | `d5bd161698b254ddebce4f4b5b410fdf1b89b33a7de4526bd146d131f96678f7` |
| `Solar.h` | header | `e56abc55c2993eb400658afde4f0a1405f6ff7794f715b4f8c294d11ac6a717e` |
| `LICENSE` | MIT license | `9f6bef5fefac5e99e335706bbd6fb3a8b5f32f87f9440b9ae9c251977c81e095` |
| `Readme.md` | usage and citation request | `02351cf8bcb063eb344586a1b0ea335a34b187fc99974ac940ad1fc7b16a33cb` |

**Use in the paper.** None directly: the probabilities of Secs. 3-5 come from `nmosyn/osc.py`, which the test
checks against this code to 1e-9.

### IceCube DeepCore golden-sample paper (`icecube_prd108/`)

**What.** The paper of the IceCube DeepCore golden event sample, arXiv version 2 (8 August 2023), whose Fig. 26
(page 27) shows the published 90% C.L. contour of $\sin^2\theta_{23}$ and $\Delta m^2_{32}$ in the normal ordering.
The release of the sample (`icecube_b4ritm/`) does not contain the contour, so it is read from the figure.

**Citation.** IceCube Collaboration, R. Abbasi et al., *Measurement of atmospheric neutrino mixing with improved
IceCube DeepCore calibration and data processing*, Phys. Rev. D 108, 012014 (2023), arXiv:2304.12236,
doi:10.1103/PhysRevD.108.012014.

**License.** CC BY 4.0 (arXiv listing). The PDF is not stored in the repository.

**Retrieval.** `https://arxiv.org/pdf/2304.12236v2`. Size 2.5 MB.

| file | content | SHA-256 |
|---|---|---|
| `2304.12236v2.pdf` | the paper, arXiv version 2 | `b0c5ab838d73a7b1bd34f43bddbf5f8a03792bb823a47f49de5a6ebcb41c74b7` |

**Derived file.** `data/icecube_prd108_fig26_contour90.json` holds the 103 points of the blue contour
("DeepCore 8 years - golden event sample") and the center of its blue best-fit cross, with $\Delta m^2_{32}$ in
$10^{-3}$ eV$^2$. `scripts/digitize_icecube_fig26.py` reads them from the vector paths of the figure: the page is
converted to SVG with poppler, both axes are calibrated on their 9 and 7 tick marks and the printed tick values
(largest residual 2e-5), and every vertex of the path is converted. The coordinates are those of the published
curve to the precision of the PDF, not a manual trace. Created on 2026-11-03.

**Use in the paper.** The published contour and best fit in fig_deepcore_reproduction, against our fit of the public release
(App. E.3).

## Derived tables

### GENIE cross sections on CH2 (`data/genie_ch2_xsec/`)

**What.** `xsec_cc.dat` and `xsec_nc.dat`, in the layout and on the grid of `data/genie_water_xsec/` (below):
$\sigma/E$ in $10^{-38}$ cm²/GeV per nucleon of CH2, the hydrocarbon of the NOvA detectors. Stored in the repository.

**Made by** `python -I scripts/make_ch2_xsec.py data/external/genie_xsec_2_12_10/gxspl-FNALsmall.xml
data/genie_ch2_xsec`, on 2026-11-14: the splines and processes of the water tables below, on C12 and H1, with CH2 per
nucleon $(\sigma_{\rm C12} + 2\sigma_{\rm H1})/14$. Check: at 50 GeV the charged-current $\sigma/E$ is 0.652
($\nu_\mu$) and 0.336 ($\bar\nu_\mu$).

**Terms and citation.** As for the water tables below.

| file | content | SHA-256 |
|---|---|---|
| `xsec_cc.dat` | charged current, CH2 | `0dbb866da4ab251b9924927ab9038cf319039e793e7771c9915631f6fc7cc23f` |
| `xsec_nc.dat` | neutral current, CH2 | `7931942858e90ad29c8d31d911aadbc5635003d9b59769c3d28e375522f1668a` |

**Use in the paper.** The cross sections of the NOvA far-detector spectra of App. A.3 (read without smoothing): the
split of the NOvA $\nu_\mu$-like spectra into $\nu_\mu$ and $\bar\nu_\mu$ and the shape of the beam $\nu_e$, in the
NOvA bands and lines of fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, fig_published_best_fits, fig_synergy_vs_precision, and
fig_synergy_rings.

### GENIE cross sections on water (`data/genie_water_xsec/`)

**What.** `xsec_cc.dat` and `xsec_nc.dat`, in the GLoBES layout of `dune_globes_xsec/` and on its 75 values of
$\log_{10}(E/\text{GeV})$: $\sigma/E$ in $10^{-38}$ cm²/GeV per nucleon of water for $\nu_e$, $\nu_\mu$,
$\nu_\tau$ (zero), $\bar\nu_e$, $\bar\nu_\mu$, $\bar\nu_\tau$ (zero). Stored in the repository.

**Made by** `python -I scripts/make_water_xsec.py data/external/genie_xsec_2_12_10/gxspl-FNALsmall.xml
data/genie_water_xsec`, on 2026-11-13. The script sums, per flavor, every charged-current (`Weak[CC]`: QES, RES,
DIS with charm, COH, MEC) and every neutral-current spline (`Weak[NC]`: QES, RES, DIS, COH) of O16 and H1, without
neutrino-electron scattering (NuEEL) and inverse muon decay (IMD); each spline is interpolated linearly in
$\log_{10}E$ onto the grid, and water per nucleon is $(\sigma_{\rm O16} + 2\sigma_{\rm H1})/18$, with
1 GeV$^{-2}$ = $0.389379\times10^{-27}$ cm². Check: at 50 GeV the charged-current $\sigma/E$ is 0.659 ($\nu_\mu$)
and 0.333 ($\bar\nu_\mu$).

**Terms and citation.** Numerical output derived from the GENIE splines above (terms there). Cite Andreopoulos
et al. (2010) and, for the meson-exchange currents, Nieves et al. (2011) and Schwehr et al. (2016).

| file | content | SHA-256 |
|---|---|---|
| `xsec_cc.dat` | charged current, water | `5174221658603b8adfee2ebe41a16d1620a18beda4f8d200a261f8474dbedf79` |
| `xsec_nc.dat` | neutral current, water | `16d76cbbea1276db8d88a0a07fd8db70052c16c7c4602514dd056c683d7630b1` |

**Use in the paper.** The cross sections of the T2K far-detector spectra of App. A.3 (flux times cross section,
read without smoothing), in the T2K bands of fig_solutions_plane and fig_intercept_ranges.
