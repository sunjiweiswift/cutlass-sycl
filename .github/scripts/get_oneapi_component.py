import argparse
import shutil
import subprocess
from pathlib import Path

from utils import download_file, get_logger, temp_env

logger = get_logger(__name__)


def cli_target():
    parser = argparse.ArgumentParser()

    parser.add_argument("--installer-url", type=str, help="URL to the installer")
    parser.add_argument("--destination", type=Path, help="Destination")

    args = parser.parse_args()

    installer_cache = Path.home() / "intel"
    if installer_cache.exists():
        logger.info(f"Removing installer cache {installer_cache}")
        shutil.rmtree(Path.home() / "intel")

    destination: Path = args.destination
    installer_url = args.installer_url

    if not destination.exists():
        destination.mkdir(parents=True)

    with temp_env(no_proxy=""):
        installer = download_file(installer_url, destination, "installer.sh")

    cmd = [
        "bash",
        installer,
        "-s",
        "-a",
        "-s",
        f"--install-dir={destination}",
        "--action",
        "install",
        "--eula",
        "accept",
    ]

    cmd = [str(arg) for arg in cmd]

    logger.info(f"Running {cmd}...")
    subprocess.run(cmd, check=True)
    logger.info("Done.")


if __name__ == "__main__":
    exit(cli_target())
