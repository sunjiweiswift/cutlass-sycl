import argparse
import tarfile
from pathlib import Path

from utils import download_file, get_logger

logger = get_logger(__name__)

LLVM_COMPILER_RELEASES_URL = "https://github.com/intel/llvm/releases/download"
COMPILER_ASSET_NAME = "sycl_linux.tar.gz"


def cli_target():
    parser = argparse.ArgumentParser()

    parser.add_argument("--compiler-version", type=str, help="Compiler version to download")
    parser.add_argument("--destination", type=Path, help="Destination")

    args = parser.parse_args()

    destination: Path = args.destination
    copmiler_version = args.compiler_version

    asset_url = f"{LLVM_COMPILER_RELEASES_URL}/{copmiler_version}/{COMPILER_ASSET_NAME}"

    if not destination.exists():
        destination.mkdir(parents=True)

    compiler_archive = download_file(asset_url, destination, "sycl.tgz")

    with tarfile.open(compiler_archive, mode="r:gz") as tar:
        logger.info(f"Extracting {compiler_archive} to {destination}...")
        tar.extractall(path=destination, filter="data")
        logger.info("Extracted.")


if __name__ == "__main__":
    exit(cli_target())
