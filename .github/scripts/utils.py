import base64
import contextlib
import getpass
import json
import logging
import netrc
import os
import ssl
import tarfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CERTS = {
    "SHA256.crt": "http://certificates.intel.com/repository/certificates/IntelSHA256RootCA-Base64.crt",
    "SHA384.crt": "https://certificates.intel.com/repository/certificates/IntelSHA384RootCA-Base64.crt",
}
GITHUB_API = "api.github.com"
GITHUB_API_URL = "https://api.github.com"


class AuthMethod(str, Enum):
    INTERACTIVE = "interactive"
    NETRC = "netrc"
    ENV = "env"


@dataclass
class LoginBundle:
    username: str
    password: str

    @property
    def basic(self) -> str:
        return base64.b64encode(f"{self.username}:{self.password}".encode()).decode()


def interactive_login() -> LoginBundle:
    current_user = getpass.getuser()
    username = input(f"Username [{current_user}]: ")
    if not username:
        username = current_user
    password = getpass.getpass("Password: ")
    return LoginBundle(username, password)


def netrc_login(entry: str) -> LoginBundle:
    netrc_file = Path().home() / ".netrc"

    if not netrc_file.exists():
        raise FileNotFoundError("Unable to find .netrc file in home dir.")

    auth = netrc.netrc(netrc_file)

    data = auth.authenticators(entry)

    if not data:
        raise KeyError(f"Unable to find {entry} entry in .netrc")

    username, _, password = data

    return LoginBundle(username, password)


def get_ssl_context() -> ssl.SSLContext:
    certs_path = Path("/tmp/certs")

    if not certs_path.exists():
        certs_path.mkdir(parents=True)

    for cert_name, cert_url in CERTS.items():
        cert = certs_path / cert_name
        if not cert.exists():
            with urlopen(cert_url) as cert_data:
                cert.write_bytes(cert_data.read())

    context = ssl.create_default_context()
    context.load_default_certs()

    for cert in certs_path.glob("*.crt"):
        context.load_verify_locations(cert)

    return context


def get_logger(name: str, file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s][%(module)s.%(funcName)s][%(levelname)s]: %(message)s")

    handlers: List[logging.Handler] = list()

    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)

    if file:
        file_handler = logging.FileHandler(file)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


logger = get_logger(__name__)


def github_request(
    endpoint: str,
    method: str = "GET",
    auth: LoginBundle | None = None,
    data: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> dict[str, Any]:
    url = f"{GITHUB_API_URL}/{endpoint}"

    req = Request(url, method=method)

    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

    if auth:
        headers |= {"Authorization": f"Bearer {auth.password}"}

    if extra_headers:
        headers |= extra_headers

    if headers:
        for key, value in headers.items():
            req.add_header(key, value)

    if data:
        req.data = json.dumps(data).encode("utf-8")

    logger.info(f"Requesting: {url}")
    try:
        with urlopen(req, context=ssl_context) as response:
            response_data = response.read().decode("utf-8")
            return json.loads(response_data)
    except HTTPError as e:
        logger.error(f"HTTPError: {e.code} - {e.reason}")
        logger.error(e.read().decode())
        raise


def download_file(
    url: str,
    dest: Path,
    file_name: str | None = None,
    ssl_context: ssl.SSLContext | None = None,
    auth: LoginBundle | None = None,
) -> Path:
    if not file_name:
        file_name = Path(urlparse(url).path).name

    req = Request(url)

    if auth:
        req.add_header("Authorization", f"Basic {auth.basic}")

    downloaded_file = dest / file_name

    logger.info(f"Downloading {url} to {dest}")
    try:
        with urlopen(req, context=ssl_context) as response, downloaded_file.open("wb") as out_file:
            total_length = response.getheader("content-length")
            if total_length is None:
                out_file.write(response.read())
            else:
                total_length = int(total_length)
                chunk_size = total_length // 5
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    done = int(50 * downloaded / total_length)
                    logger.info(f"[{'=' * done}{' ' * (50 - done)}] {downloaded / total_length:.2%}")
    except HTTPError as er:
        logger.info(er.read().decode())

    logger.info("Downloaded")

    return downloaded_file


@contextlib.contextmanager
def change_dir(path: Path | str):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@contextlib.contextmanager
def temp_env(**env_vars):
    prev_env = os.environ.copy()
    os.environ.update(env_vars)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(prev_env)


def extract_archive(archive: Path, destination: Path):
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(path=destination, filter="data")
