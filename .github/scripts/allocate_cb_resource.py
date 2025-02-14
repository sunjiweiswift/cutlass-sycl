import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from utils import LoginBundle, get_logger, get_ssl_context

SEPARATOR = f"{'#' * 8}\n"
CB_SERVER_API = "https://ecmd.jf.intel.com/rest/v1.0"

logger = get_logger(__name__)


def send_cb_api_request(
    endpoint: str,
    data: Dict[str, str] | None = None,
    session_id: str | None = None,
    method: str = "GET",
) -> Dict:
    headers = {"accept": "application/json"}
    if session_id:
        headers["Cookie"] = f"sessionId={session_id}"

    url = f"{CB_SERVER_API}/{endpoint}"
    if data:
        request_data = urlencode(data)
        url = f"{url}?{request_data}"

    req = Request(url, headers=headers, method=method)

    try:
        with urlopen(req, context=get_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        logger.error(e.read().decode())
        raise


def get_session():
    try:
        auth = LoginBundle(os.environ["CB_USER"], os.environ["CB_PASSWORD"])
    except KeyError:
        raise ValueError("CB_USER and CB_PASSWORD environment variables must be set")

    res = send_cb_api_request(
        endpoint="sessions",
        data={"password": auth.password, "userName": auth.username},
        method="POST",
    )

    return res["sessionId"]


def get_resource(session_id: str, resource_name: str) -> Dict:
    res = send_cb_api_request(
        endpoint=f"resources/{resource_name}",
        session_id=session_id,
    )

    return res["resource"]


def modify_resource(session_id: str, resource_name: str, **params: Any) -> Dict:
    res = send_cb_api_request(
        endpoint=f"resources/{resource_name}",
        data=params,
        session_id=session_id,
        method="PUT",
    )

    return res


def lock_resource(session_id: str, resource_name: str):
    resource = get_resource(session_id, resource_name)
    current_description = resource["description"]
    modify_resource(session_id, resource_name=resource_name, resourceDisabled=True)
    modify_resource(
        session_id,
        resource_name=resource_name,
        description=f"{current_description}\n{SEPARATOR}Locked for cutlass CI at {datetime.now()}",
    )

    logger.info(f"Resource {resource_name} locked.")


def unlock_resource(session_id: str, resource_name: str):
    resource = get_resource(session_id, resource_name)
    current_description = resource["description"]
    modify_resource(session_id, resource_name=resource_name, resourceDisabled=False)
    modify_resource(
        session_id, resource_name=resource_name, description=current_description.split(SEPARATOR)[0].strip()
    )

    logger.info(f"Resource {resource_name} unlocked.")


def wait_for_resource(session_id: str, resource_name: str):
    lock_resource(session_id, resource_name)
    resource = get_resource(session_id, resource_name)
    while int(resource["stepCount"]) > 0:
        logger.info("Resource is busy, waiting for it to become available...")
        resource = get_resource(session_id, resource_name)
        time.sleep(10)


def cli_target():
    parser = argparse.ArgumentParser(description="Allocate CloudBees CD/RO resource")
    parser.add_argument("resource_name", type=str, help="Name of the resource to allocate")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--unlock", action="store_true", help="Unlock the resource")
    group.add_argument("--lock", action="store_true", help="Lock the resource")

    args = parser.parse_args()

    logger.info("Requesting session...")
    session_id = get_session()
    logger.info("Session received.")

    if args.lock:
        wait_for_resource(session_id, args.resource_name)
        exit(0)

    if args.unlock:
        unlock_resource(session_id, args.resource_name)
        exit(0)


if __name__ == "__main__":
    exit(cli_target())
