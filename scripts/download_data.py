"""Fetch the public releases used by the paper into data/external/ and verify their SHA-256 checksums.

    python scripts/download_data.py                   fetch every file that is missing or has a wrong checksum,
                                                      except the opt-in releases of OPT_IN, fetched by name only
    python scripts/download_data.py --check           verify the files on disk, download nothing
    python scripts/download_data.py honda nova_2024   restrict to some releases

Each checksum refers to the file as it is stored, after decompression or extraction. Files already on disk
with the right checksum are not downloaded again, so a run that stops halfway resumes where it stopped.
A release that cannot be fetched is reported and skipped; the script then exits with status 1.
Sources, citations, and terms of use: data/PROVENANCE.md. Python standard library only.
"""
import argparse
import gzip
import hashlib
import http.client
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

EXTERNAL = Path(__file__).resolve().parents[1] / 'data' / 'external'
# Some servers reject requests without a user agent or an Accept header.
HEADERS = {'User-Agent': 'nmosyn-download/1.0', 'Accept': '*/*'}
HARVARD = 'https://dataverse.harvard.edu/api/access/datafile/{}'   # Dataverse file id
KM3NET = 'https://opendata.km3net.de/api/access/datafile/{}'       # Dataverse file id
RETRY_WAIT = (5, 20, 60)        # seconds before the second, third, and fourth attempt
# Besides 5xx, the HTTP errors that clear after a wait: bot challenges (publicdocs.fnal.gov) and rate limits.
TRANSIENT = {403, 408, 429}


class Failure(Exception):
    """A download or a checksum that failed."""


@dataclass(frozen=True)
class Files:
    """Files fetched one by one from url.format(key), where key is the server's file id or the file name."""
    landing: str
    url: str
    sha256: dict[str, str]                                  # stored name -> checksum of the stored file
    keys: dict[str, int | str] = field(default_factory=dict)   # Dataverse file ids or paths on the server
    gunzip: bool = False                                    # served gzip-compressed, stored decompressed

    def source(self, name: str) -> str:
        return self.url.format(self.keys.get(name, name))


@dataclass(frozen=True)
class Archive:
    """One zip or tar archive whose files are written flat into the destination folder."""
    landing: str
    url: str
    archive_sha256: str
    prefix: str                  # folder of the files inside the archive
    sha256: dict[str, str]       # stored name -> checksum of the extracted file


SOURCES = {
    # NuFast-Earth (P. B. Denton and S. J. Parke, arXiv:2511.04735), MIT license, at a fixed commit: the original
    # C++ engine that tests/test_nufast_earth.py compiles to check nmosyn/osc.py, a port of its algorithm.
    'nufast_earth': Files(
        landing='https://github.com/PeterDenton/NuFast-Earth',
        url='https://raw.githubusercontent.com/PeterDenton/NuFast-Earth/327586dddcb8c01282feb7560d029b63ea1257f4/{}',
        keys={'NuFastEarth.cpp': 'src/NuFastEarth.cpp', 'Oscillation.cpp': 'src/Oscillation.cpp',
              'Earth.cpp': 'src/Earth.cpp', 'Geometry.cpp': 'src/Geometry.cpp', 'Solar.cpp': 'src/Solar.cpp',
              'NuFastEarth.h': 'include/NuFastEarth.h', 'Earth.h': 'include/Earth.h', 'Matrix.h': 'include/Matrix.h',
              'Geometry.h': 'include/Geometry.h', 'Solar.h': 'include/Solar.h', 'LICENSE': 'LICENSE',
              'Readme.md': 'Readme.md'},
        sha256={
            'NuFastEarth.cpp': 'a7ff0f58beec38af22b418653be267f7c999c570d651b7da5541ea340bdf088e',
            'Oscillation.cpp': '03ed2663f22eea78c27968c6aa781aba457cb7974e8de957e618cb6947d1e9d9',
            'Earth.cpp': 'fc0e6c2e9d6b3e6006740ce4f10b3c5dac131fdc5e39f690e1edc5cad330a368',
            'Geometry.cpp': '799fb9b91a5508260e3e6fcedeed602c1ffbbbd7736190eb75eb35b5fe94362a',
            'Solar.cpp': '757717f07a0963b2854cdc89c4855d7da65bfc3a4b21a60687f4c63832971315',
            'NuFastEarth.h': '6c47861b038e7b2408868bec1e3ec83d9bc54572b7b58cbba651b9ebe51d9dfb',
            'Earth.h': '7e1030e911cca1c042a1420af81b4b5dae15e88193d954da26359e317a9cb4cc',
            'Matrix.h': '77eca8a4bb92f729ab6f2f1066c45fb57e573c3245e09223b1ede6b8d0bf1447',
            'Geometry.h': 'd5bd161698b254ddebce4f4b5b410fdf1b89b33a7de4526bd146d131f96678f7',
            'Solar.h': 'e56abc55c2993eb400658afde4f0a1405f6ff7794f715b4f8c294d11ac6a717e',
            'LICENSE': '9f6bef5fefac5e99e335706bbd6fb3a8b5f32f87f9440b9ae9c251977c81e095',
            'Readme.md': '02351cf8bcb063eb344586a1b0ea335a34b187fc99974ac940ad1fc7b16a33cb',
        }),
    # IceCube DeepCore, PRD 108, 012014 (2023). Dataverse serves the ingested tables tab-separated by default;
    # the checksums refer to that format (the ?format=original files are comma-separated .csv).
    'icecube_b4ritm': Files(
        landing='https://doi.org/10.7910/DVN/B4RITM',
        url=HARVARD,
        keys={
            'data.tab': 11646859,
            'mc_mu.tab': 11646851,
            'mc_nu_nc.tab': 11646850,
            'mc_nue_cc.tab': 11646852,
            'mc_numu_cc.tab': 11646856,
            'mc_nutau_cc.tab': 11646855,
            'hs_nu_nc_nue_cc.tab': 11646854,
            'hs_numu_cc.tab': 11646858,
            'hs_nutau_cc.tab': 11646853,
            'readme.md': 11674676,
        },
        sha256={
            'data.tab': 'c1b9baa1bc53f41bb43a4daf7cbeb953c2c6b74c8e13e59afb5764274413671a',
            'mc_mu.tab': 'af11edbfe0b14f76d5fcb7912f857dd331956deeef3efcba388b17ec3f67a5fd',
            'mc_nu_nc.tab': '4ea029d862cb0c67bd0d704af437ebcbda80ed6ac8c57e5db6364c595004e89e',
            'mc_nue_cc.tab': 'e7040158d7d4194e0d080c653651f4cbe7c7ba78d8a03731032c4f9751d58fc3',
            'mc_numu_cc.tab': '0354af4398b5d02004ea0b6631ca81a348bb5d08b406b2670b0abeadaac2bb53',
            'mc_nutau_cc.tab': '5558c64dc43acd9c9ad0c0751a1c5d28fef3f3b4d3cc4c632d47c176022627e5',
            'hs_nu_nc_nue_cc.tab': 'b60aad105805818191ecc8bdacef920fe662a39a7b64f1b17c5d18ccbbf96dd6',
            'hs_numu_cc.tab': 'e392b277091b98b47ece769fe50976130da5e0e3a0ca1be5ec1d53bec2568ab9',
            'hs_nutau_cc.tab': '55e1608d05cd5b27891ec052762431b3b0b764b61062de9414712f582e920067',
            'readme.md': '0f8e7bbc6059d18376843370e36e98a0cfa69fbc4bd767df694c6d65f861149f',
        }),
    # IceCube DeepCore 9.3 years, PRL 134, 091801 (2025): likelihood maps in both orderings.
    'icecube_u20mmb': Files(
        landing='https://doi.org/10.7910/DVN/U20MMB',
        url=HARVARD,
        keys={
            'README_icecube_9.3y.txt': 10497185,      # README.txt in the release
            'saved_no_delta_LLH_map.npy': 10490065,
            'saved_io_delta_LLH_map.npy': 10490064,
            'script.py': 10497184,
        },
        sha256={
            'README_icecube_9.3y.txt': 'd22ce6c141e62515f2a539238bbd80d032357f832043ebc20e46716bc448c0ae',
            'saved_no_delta_LLH_map.npy': '7aadd4ae1ca4624fe6e0909ebf17b45f3f0b8bdc4270d1fa66585fe2c3b5c608',
            'saved_io_delta_LLH_map.npy': '103bf82f163c5f3c4fb9278b762aebd23bca083481fa703e273dee890664d48b',
            'script.py': '3f1790ad12d9612ef1005a1f73e9792ef6c4bb52c2169c5e59e6ea38a712e976',
        }),
    # KM3NeT/ORCA6 433 kt-y open data v0.5, JHEP 10 (2024) 206.
    'km3net_orca6': Files(
        landing='https://opendata.km3net.de/dataset.xhtml?persistentId=doi:10.5072/FK2/Y0UXVW',
        url=KM3NET,
        keys={
            'ORCA6_433kt-y_opendata_v0.5.root': 154,
            'bins_433kt-y_v0.5.json': 153,
            'README.md': 156,
            'chi2_landscape_NO.csv': 157,
            'chi2_landscape_IO.csv': 155,
        },
        sha256={
            'ORCA6_433kt-y_opendata_v0.5.root':
                '79453657367c8b8582f6ec384837e6d0ac3ce62fc45378984d4d6d33c58441c3',
            'bins_433kt-y_v0.5.json':
                'b3bee5fa5665aaa94d4aea750b6890df37a364f39c7f9bb12d9530a7edb3e3e1',
            'README.md':
                'c75844611f03059849dfc01f97a6f08e36a3c2e8bb2d86c61549d9dd85354d5f',
            'chi2_landscape_NO.csv':
                '121b1dcd6b8d46b92555b42c5647c77b487ca5c3ce0a7e2e66bcab7803a5838c',
            'chi2_landscape_IO.csv':
                'bbc13926f749fafae6de50b383e1fe84b440fbaf8173877727295c89ef75f82a',
        }),
    # Honda et al., PRD 92, 023004 (2015): azimuth-averaged tables, solar minimum, no mountain.
    # The host serves these tables over plain http only; the checksums guard the content.
    'honda': Files(
        landing='http://www-rccn.icrr.u-tokyo.ac.jp/mhonda/public/nflx2014/index.html',
        url='http://www-rccn.icrr.u-tokyo.ac.jp/mhonda/public/nflx2014/{}.gz',
        gunzip=True,
        sha256={
            'spl-ally-20-01-solmin.d': 'fb26dccb15317d2ca1d7ba54af1cfa17a188127c8f74f7f7c23c5beb96b3fc83',
            'frj-ally-20-01-solmin.d': 'e9397af19d58514600e646b14c6449ac2bdea972476dda6d288db375b52131fa',
        }),
    # NOvA 2024 data release, FERMILAB-DATA-2025-05 (Zenodo version of 2026-07-09), PRL 136, 011802 (2026).
    'nova_2024': Archive(
        landing='https://doi.org/10.5281/zenodo.21284589',
        url='https://zenodo.org/api/records/21284589/files/data_release_2024.zip/content',
        archive_sha256='9ef7b65dda776590c07209f4c5a9879c41a9cc08677b3016189d87b57b365858',
        prefix='data_release_2024/',
        sha256={
            'README.md':
                '8f3d6eca55513d40ab1302bd3438dfec885ac2cadc46bd06bd28573749ee73a9',
            'NOvA_2024_data_histograms.root':
                'c3fee51cb3c57e6b0196480c0e4a2bbd5ee24959c5868daad8de530cd3e58eb9',
            'NOvA_2024_prediction_with_systs_histograms.root':
                '35be5a68210fec77072ce864b0c6bdd3655c2829526b1ce327614f24c70233f9',
            'contours_NOvAOnly_cond.root':
                '61a385b6db4b40005086019fd4dc8e4e9cb9b4e808a89f8a1001b61c43c966b8',
            'contours_NOvAOnly_noncond.root':
                '113470616563c705d420302060d2032234a1522aefd495e725a68eca9844aeff',
            'contours_RCDB1D_cond.root':
                'fffc59cc5aa4878f463ca79571a845dde2b8987afa85ab9f24095d6f05230500',
            'contours_RCDB1D_noncond.root':
                '516c102c39bd117bff2376a21e52735ea2dde455191b68f2186dea8c6d2164dc',
            'contours_RCDB2D_cond.root':
                'ec930dd5be7f5416cb44025a04448f8ce58dcd74795409568b68a04964f5d33b',
            'contours_RCDB2D_noncond.root':
                'd6fc8ff0fe35d8265dac58821f775b55e01dd8179ce2a73443bac18e604ef7a4',
        }),
    # NOvA near-detector flux for the 2017 analysis, FERMILAB-DATA-2017-01.
    'nova_flux_2017': Files(
        landing='https://publicdocs.fnal.gov/cgi-bin/ShowDocument?docid=8',
        url='https://publicdocs.fnal.gov/cgi-bin/RetrieveFile?docid=8&version=1&filename={}',
        sha256={
            'FHC_Flux_NOvA_ND_2017.root':
                '8e2f7cb411c01a8938c89c330bc5855def7c423f32b9eced4e87f2d1155e26e0',
            'FHC_Flux_numu_NOvA_ND_2017.txt':
                'c9477c7517e5269361994b8a39eccb2c54732920d1917e538339b1758a19c505',
            'FHC_Flux_numubar_NOvA_ND_2017.txt':
                'd429ccc573469a924815c248b521bd2aea396f8e46770b5884d3c5292c5be959',
            'FHC_Flux_nue_NOvA_ND_2017.txt':
                '0e80cd4362f5e391039c43a31c18a124a0d7c1f97685c2ea50731ee29f59a56b',
            'FHC_Flux_nuebar_NOvA_ND_2017.txt':
                'd483e1a5c90e78066197c1b405cb2627c18f3aff13e4b291782faf3ed65d5e55',
            'RHC_Flux_NOvA_ND_2017.root':
                'eb0f753b042aa58be145a8ab7f10424ee8161379d7481f89c170f151283ab1af',
            'RHC_Flux_numu_NOvA_ND_2017.txt':
                'edd8b876ed37d1f6f535d00f4434b7b93a8156212f9f151729aabf0bc1e59ed5',
            'RHC_Flux_numubar_NOvA_ND_2017.txt':
                '7ba49a5a3a2fdadd7fe095be6cb260c48c45328a9e4a9220d18cbe07942c664b',
            'RHC_Flux_nue_NOvA_ND_2017.txt':
                '0e3085fd495294dbf3e17dbdb32b9b29b0b5aab846cff08119b4443a913f81af',
            'RHC_Flux_nuebar_NOvA_ND_2017.txt':
                '1622dce9549da4bec50f63a018a19cac6450a01599b4f326511173382c8a7590',
        }),
    # T2K neutrino beam flux prediction 2020 (a gzip-compressed tar file).
    't2k_flux_2020': Archive(
        landing='https://t2k-experiment.org/results/neutrino-beam-flux-prediction-2020/',
        url='https://t2k-experiment.org/wp-content/uploads/T2Kflux2020.gz',
        archive_sha256='0c8e921ede08508d192cb203ebfb6386d15fbc7ab6ad099ebfd503cae12a8569',
        prefix='t2kflux_2020_public_release/',
        sha256={
            'README.pdf':
                '367fe6275b02adcb61da2892a725139baa9fcbfe0f0b612ca64d96ebd466e6c0',
            't2kflux_2020_covariance.root':
                '2c683ec927e7632a347091c70dc82a1f4be565d171cd46d898ebc4e13672245f',
            't2kflux_2020_plus250kA_nominal.root':
                'ebbc3ebd9a6d8e7e306c50ec348c2b428382b1cacc45a7386b4090fa9d5ab583',
            't2kflux_2020_plus250kA_nominal_nd280.txt':
                'f15c739ecc4049438e66366487d61a7ba0be35dc217487ae74fd621bf4c56f49',
            't2kflux_2020_plus250kA_nominal_sk.txt':
                'cf70149d8f03ae9dc9a629a2808686dd1dfcdca63b3ee0c9ea7557143c2d0b58',
            't2kflux_2020_plus250kA_runcond.root':
                '86f25030e5da7309071e883c352c4d76946b2e4b7a1c30597acd85ec703d284e',
            't2kflux_2020_plus250kA_runcond_nd280.txt':
                'c5647504c9192a5ea106448d61365e87050b98beb9ff1a86bf8ebaa876f5d70b',
            't2kflux_2020_plus250kA_runcond_sk.txt':
                '75ae2a7161890c76850ba0247f4a62a1a49aa7bba06dda29dfebacf6117438c7',
            't2kflux_2020_minus250kA_nominal.root':
                'ca07f81b38675f463e74bba5cd60b67bcbca6ad9423107ad5bfb81aace7187b6',
            't2kflux_2020_minus250kA_nominal_nd280.txt':
                'fe1679f3304a82064691db79f079252823c91ae796238e6064588b30772dbd44',
            't2kflux_2020_minus250kA_nominal_sk.txt':
                '2379f71da48ebffee6f73c984e1de298173e51d5d9495f299279d2ef299f8b73',
            't2kflux_2020_minus250kA_runcond.root':
                '017cb6c8dde1be3ba5c779b0142454739ff91075aaedd4512176c32ee82bbb37',
            't2kflux_2020_minus250kA_runcond_nd280.txt':
                '87990a2852fc50f60ea3fe5f790f057ffe1156f3f903bd3dcd72a68377f59a4e',
            't2kflux_2020_minus250kA_runcond_sk.txt':
                'af969e4459f43399094d8de5b90fcee6dde81b7cbf284b0a2660626598066276',
        }),
    # GENIE cross sections of the DUNE GLoBES configuration, ancillary files of arXiv:2103.04797.
    'dune_globes_xsec': Files(
        landing='https://arxiv.org/abs/2103.04797',
        url='https://arxiv.org/src/2103.04797v2/anc/dune_globes/xsec/{}',
        sha256={
            'xsec_cc.dat': 'bb950e7fb645bc5cdb052d17e2076534be2cbfd24e83ddc851f6d906ece93efd',
            'xsec_nc.dat': '613cb9005dae60e028e429f125b537b11bce3f7aaa389d6dc329f4a9ce7f28d9',
        }),
    # The paper of the golden event sample, arXiv version 2 (PRD 108, 012014), CC BY 4.0: the source of the published
    # 90% CL contour, which scripts/digitize_icecube_fig26.py reads from the vector paths of its Fig. 26.
    'icecube_prd108': Files(
        landing='https://arxiv.org/abs/2304.12236v2',
        url='https://arxiv.org/pdf/{}',
        keys={'2304.12236v2.pdf': '2304.12236v2'},
        sha256={'2304.12236v2.pdf': 'b0c5ab838d73a7b1bd34f43bddbf5f8a03792bb823a47f49de5a6ebcb41c74b7'}),
    # GENIE 2.12.10 pre-computed splines (DefaultPlusValenciaMEC): the input of scripts/make_ch2_xsec.py and
    # scripts/make_water_xsec.py, which write the cross sections of data/genie_ch2_xsec/ (NOvA) and
    # data/genie_water_xsec/ (T2K). Opt-in: the archive has 498 MiB.
    'genie_xsec_2_12_10': Archive(
        landing='https://scisoft.fnal.gov/scisoft/packages/genie_xsec/v2_12_10/',
        url='https://scisoft.fnal.gov/scisoft/packages/genie_xsec/v2_12_10/'
            'genie_xsec-2.12.10-noarch-DefaultPlusValenciaMEC.tar.bz2',
        archive_sha256='fe1b584e7014bba6c4cba5646e1031f344e9efbf799a2aa26b706e28c40a4481',
        prefix='genie_xsec/v2_12_10/NULL/DefaultPlusValenciaMEC/data/',
        sha256={
            'README':
                'fc41d8b89f553279d7bc78a907888598eed5d5c6b3a08ccef42bf9da50bd347e',
            'gxspl-FNALsmall.xml':
                'c28d6e6596d0b67b900b32e8dfdaf48ebf8fc938b942c1db0374f2f27b26539a',
        }),
}
OPT_IN = {'genie_xsec_2_12_10'}     # large releases that only regenerate committed tables, fetched by name


def sha256(path: Path) -> str:
    """SHA-256 of a file, as a hex string."""
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def stale(release: Files | Archive, folder: Path) -> list[str]:
    """Names of the files that are missing from folder or have a wrong checksum."""
    return [name for name, digest in release.sha256.items()
            if not (folder / name).is_file() or sha256(folder / name) != digest]


def fetch(url: str, target: Path, gunzip: bool = False) -> None:
    """Stream url into target, decompressing gzip on the fly if asked; retry the failures that may clear."""
    request = urllib.request.Request(url, headers=HEADERS)
    for wait in (*RETRY_WAIT, None):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                source = gzip.GzipFile(fileobj=response) if gunzip else response
                with target.open('wb') as out:
                    shutil.copyfileobj(source, out, 1 << 20)
            return
        except (OSError, EOFError, zlib.error, http.client.HTTPException) as err:
            target.unlink(missing_ok=True)
            permanent = (isinstance(err, urllib.error.HTTPError)
                         and err.code < 500 and err.code not in TRANSIENT)
            if permanent or wait is None:
                raise Failure(f'download of {url} failed: {err}') from err
            print(f'  {url}: {err}; trying again in {wait} s', flush=True)
            time.sleep(wait)


def verify(path: Path, digest: str) -> None:
    """Remove path and raise Failure if its checksum differs from digest."""
    if sha256(path) != digest:
        path.unlink()
        raise Failure(f'{path.name} has an unexpected SHA-256 and was removed; the release may have changed')


def unpack(archive: Path, prefix: str, names: list[str], folder: Path) -> None:
    """Write the named files of a zip or tar archive into folder, without the archive's folder prefix."""
    bundle = zipfile.ZipFile(archive) if zipfile.is_zipfile(archive) else tarfile.open(archive)
    member = bundle.open if isinstance(bundle, zipfile.ZipFile) else bundle.extractfile
    with bundle:
        for name in names:
            with member(prefix + name) as src, (folder / name).open('wb') as out:
                shutil.copyfileobj(src, out)


def download(release: Files | Archive, names: list[str], folder: Path) -> None:
    """Fetch the named files of a release into folder and verify them; raise Failure on the first problem."""
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(release, Archive):
        part = folder / '.archive.part'
        try:
            fetch(release.url, part)
            verify(part, release.archive_sha256)
            unpack(part, release.prefix, names, folder)
        finally:
            part.unlink(missing_ok=True)
        for name in names:
            verify(folder / name, release.sha256[name])
        return
    for name in names:
        part = folder / f'{name}.part'
        fetch(release.source(name), part, release.gunzip)
        verify(part, release.sha256[name])
        part.replace(folder / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('names', nargs='*', metavar='release', help=f'any of: {", ".join(SOURCES)}')
    parser.add_argument('--check', action='store_true', help='verify the files on disk, download nothing')
    parser.add_argument('--dest', type=Path, default=EXTERNAL, help='destination (default data/external)')
    args = parser.parse_args()
    unknown = sorted(set(args.names) - SOURCES.keys())
    if unknown:
        parser.error(f'unknown release {", ".join(unknown)}; choose from {", ".join(SOURCES)}')

    failed = []
    for key in args.names or [k for k in SOURCES if k not in OPT_IN]:
        release, folder = SOURCES[key], args.dest / key
        missing = stale(release, folder)
        if not missing:
            print(f'{key}: {len(release.sha256)} files ok')
        elif args.check:
            print(f'{key}: missing or wrong: {", ".join(missing)}')
            failed.append(key)
        else:
            print(f'{key}: fetching {", ".join(missing)} from {release.landing}', flush=True)
            try:
                download(release, missing, folder)
            except Failure as err:
                print(f'{key}: {err}', file=sys.stderr, flush=True)
                failed.append(key)
                continue
            print(f'{key}: {len(release.sha256)} files ok')
    if failed and args.check:
        sys.exit(f'Not verified: {", ".join(failed)}. Run without --check to download them.')
    if failed:
        pages = '\n'.join(f'  {key}: {SOURCES[key].landing}' for key in failed)
        sys.exit(f'Not fetched:\n{pages}\nRun the script again to resume: the verified files are kept and '
                 f'only the missing ones are fetched. Or get the files by hand from the pages above.')


if __name__ == '__main__':
    main()
