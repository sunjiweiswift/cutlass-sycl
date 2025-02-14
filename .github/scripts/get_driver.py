import argparse
import os
import subprocess
import sys
import tarfile
from enum import Enum
from pathlib import Path

from utils import (
    GITHUB_API,
    AuthMethod,
    LoginBundle,
    change_dir,
    download_file,
    extract_archive,
    get_logger,
    get_ssl_context,
    github_request,
    interactive_login,
    netrc_login,
)

logger = get_logger(__name__)

GFX_DRIVERS_INSTANCE = "gfx-assets-build.fm.intel.com"
GFX_DRIVERS_ARTIFACTORY_API_URL_TEMPLATE = (
    "https://{instance}/artifactory/api/archive/download/{repo}/{branch}/{version}/{path}?archiveType=tgz"
)
COMPUTE_RUNTIME_RELEASE_TEMPLATE = "repos/intel/compute-runtime/releases/tags/{version}"
IGC_RELEASE_TEMPLATE = "repos/intel/intel-graphics-compiler/releases/tags/{version}"


class DriverType(Enum):
    INTERNAL = "internal"
    PUBLIC = "public"


def _unpack_deb_package(packages_root: Path) -> Path:
    unpacked_dir = packages_root / "unpacked"
    unpacked_dir.mkdir(exist_ok=True)

    deb_packages = packages_root.glob("*.*deb")

    with change_dir(packages_root):
        for deb_package in deb_packages:

            logger.info(f"Processing file: {deb_package}")

            logger.info("Unpacking .deb package...")
            subprocess.run(["ar", "x", str(deb_package)], check=True)

            logger.info("Unpacking data archives...")
            data_archives = ("data.tar.xz", "data.tar.gz", "data.tar.zst")
            for data_archive in data_archives:
                data_archive_path = packages_root / data_archive
                if data_archive_path.exists():
                    logger.info(f"Unpacking {data_archive}...")
                    if data_archive == "data.tar.zst":
                        try:
                            subprocess.run(
                                [
                                    "tar",
                                    "--use-compress-program=unzstd",
                                    "-xf",
                                    str(data_archive_path),
                                    "-C",
                                    str(unpacked_dir),
                                ],
                                check=True,
                            )
                        except subprocess.CalledProcessError:
                            logger.info(f"No archive in {deb_package}")
                            sys.exit(1)
                    else:
                        try:
                            extract_archive(data_archive_path, unpacked_dir)
                        except tarfile.ExtractError:
                            logger.info(f"No archive in {deb_package}")
                            sys.exit(1)

            temp_files = (
                "data.tar.xz",
                "data.tar.gz",
                "data.tar.zst",
                "control.tar.xz",
                "control.tar.gz",
                "control.tar.zst",
                "debian-binary",
            )
            for temp_file in temp_files:
                temp_file_path = packages_root / temp_file
                if temp_file_path.exists():
                    temp_file_path.unlink()

    return unpacked_dir


def _create_ocloc_link(driver_binaries_root: Path):
    try:
        ocloc_binary = next(driver_binaries_root.rglob("ocloc?*"))
        ocloc_symlink = ocloc_binary.parent / "ocloc"
        ocloc_symlink.symlink_to(ocloc_binary.name)
    except StopIteration:
        logger.info("Cannot create symlink to ocloc. It either exists already as a binary or not found at all.")


def _download_internal_driver_packages(
    destination: Path,
    auth_method: AuthMethod,
    driver_version: str,
    driver_branch: str,
    driver_repo: str,
    driver_path: str,
):
    if auth_method is AuthMethod.NETRC:
        auth = netrc_login(GFX_DRIVERS_INSTANCE)
    elif auth_method is AuthMethod.INTERACTIVE:
        auth = interactive_login()
    else:
        auth = LoginBundle(os.environ["ARTIFACTORY_USER"], os.environ["ARTIFACTORY_TOKEN"])

    url = GFX_DRIVERS_ARTIFACTORY_API_URL_TEMPLATE.format(
        instance=GFX_DRIVERS_INSTANCE,
        repo=driver_repo,
        branch=driver_branch,
        version=driver_version,
        path=driver_path,
    )
    downloaded_archive = download_file(
        url,
        destination,
        "driver.tgz",
        ssl_context=get_ssl_context(),
        auth=auth,
    )

    extract_archive(downloaded_archive, destination)


def _download_public_driver_packages(
    destination: Path, auth_method: AuthMethod, compute_runtime_version: str, igc_version: str
):
    if auth_method is AuthMethod.NETRC:
        auth = netrc_login(GITHUB_API)
    elif auth_method is AuthMethod.INTERACTIVE:
        auth = interactive_login()
    else:
        auth = LoginBundle(os.environ["GITHUB_USER"], os.environ["GITHUB_TOKEN"])

    compute_runtime_assets = github_request(
        COMPUTE_RUNTIME_RELEASE_TEMPLATE.format(version=compute_runtime_version),
        auth=auth,
        ssl_context=get_ssl_context(),
    )["assets"]

    igc_assets = github_request(
        IGC_RELEASE_TEMPLATE.format(version=igc_version),
        auth=auth,
        ssl_context=get_ssl_context(),
    )["assets"]

    asset_urls = list()
    for asset in compute_runtime_assets:
        url = asset["browser_download_url"]
        if any(suffix in url for suffix in (".deb", ".ddeb")):
            asset_urls.append(url)

    for asset in igc_assets:
        url = asset["browser_download_url"]
        if any(package_name in url for package_name in ("intel-igc-core", "intel-igc-opencl_")):
            asset_urls.append(url)

    for asset_url in asset_urls:
        download_file(asset_url, destination)


def cli_target():
    parser = argparse.ArgumentParser(description="Download drivers")
    parser.add_argument("--destination", type=Path, help="Where to download driver")
    parser.add_argument(
        "--auth",
        dest="auth_method",
        default=AuthMethod.ENV,
        choices=AuthMethod,
        type=AuthMethod,
        help="Authentication method",
    )

    subparsers = parser.add_subparsers(dest="driver_type", required=True, help="Type of driver to download")

    # Public driver subparser
    public_parser = subparsers.add_parser(DriverType.PUBLIC.value, help="Download public driver")
    public_parser.add_argument("--compute-runtime-version", required=True, help="Version of the compute-runtime")
    public_parser.add_argument("--igc-version", required=True, help="Version of the intel-graphics-compiler")

    # Internal driver subparser
    internal_parser = subparsers.add_parser(DriverType.INTERNAL.value, help="Download internal driver")
    internal_parser.add_argument(
        "--driver-repo",
        required=False,
        help="Repository of the internal driver. Default: gfx-driver-builds",
        default="gfx-driver-builds",
    )
    internal_parser.add_argument(
        "--driver-branch",
        required=True,
        help="Branch of the internal driver. For example: ci/comp_igc",
    )
    internal_parser.add_argument(
        "--driver-version",
        required=True,
        help="Version of the internal driver. For example: gfx-driver-ci-comp_igc-25012",
    )
    internal_parser.add_argument(
        "--driver-path",
        required=False,
        default="artifacts/Linux/Ubuntu/22.04/Release",
        help=(
            "Path to the folder on artifactory which contains .deb packages."
            " Default: artifacts/Linux/Ubuntu/22.04/Release"
        ),
    )

    args = parser.parse_args()

    destination: Path = args.destination
    auth_method: AuthMethod = args.auth_method
    driver_type = DriverType(args.driver_type)

    if not destination.exists():
        destination.mkdir(parents=True)

    if driver_type is DriverType.PUBLIC:
        _download_public_driver_packages(
            destination,
            auth_method,
            args.compute_runtime_version,
            args.igc_version,
        )

    if driver_type is DriverType.INTERNAL:
        _download_internal_driver_packages(
            destination,
            auth_method,
            args.driver_version,
            args.driver_branch,
            args.driver_repo,
            args.driver_path,
        )

    unpacked = _unpack_deb_package(destination)
    _create_ocloc_link(unpacked / "usr" / "bin")


if __name__ == "__main__":
    exit(cli_target())
