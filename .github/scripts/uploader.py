import argparse
import logging
import os
import sys
from parser import BenchmarkType
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from models.common import CommonBaseModel, DataType, Platform, Reference, Run, RunType
from models.cutlass import CutlassBenchmark, Layout
from models.utils import create_or_update, get_or_create, split_unique_values
from models.xetla import XetlaBenchmark
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

ignore = CommonBaseModel

logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

run_columns = ["run_type", "sha", "branch", "platform", "data_type"]


def convert_to_native_type(value) -> Any:
    if value == "nan" or pd.isna(value):
        return None
    elif isinstance(value, np.generic):
        return value.item()
    else:
        return value


def construct_run_item(session: Session, data: dict):
    reference = get_or_create(
        session, Reference, lookup_by={"sha": data["sha"]}, sha=data["sha"], branch=data["branch"]
    )

    run_type = get_or_create(session, RunType, type=data["run_type"])
    platform = get_or_create(session, Platform, name=data["platform"])
    data_type = get_or_create(session, DataType, type=data["data_type"])

    run = get_or_create(
        session, Run, run_type=run_type, reference_rel=reference, platform=platform, data_type=data_type
    )

    session.flush()

    return run


def construct_cutlass_benchmark_data(session: Session, data: pd.DataFrame, run: Run):
    grouped = data.groupby(["layout"])

    for name, group in grouped:
        layout_data = {col: convert_to_native_type(val) for col, val in zip(grouped.keys, name)}

        layout = get_or_create(session, Layout, name=layout_data["layout"])

        tests_data: List[Dict] = (
            group[
                [
                    "name",
                    "real_time",
                    "cpu_time",
                    "total_runtime_ms",
                    "avg_runtime_ms",
                    "avg_tflops",
                    "avg_throughput",
                    "best_bandwidth",
                    "best_runtime_ms",
                    "best_tflop",
                    "alpha",
                    "beta",
                    "batch",
                    "m",
                    "k",
                    "n",
                    "status",
                ]
            ]
            .copy()
            .to_dict("records")
        )

        for test_data in tests_data:
            unique_data, variable_data = split_unique_values(CutlassBenchmark, test_data)
            create_or_update(
                session, CutlassBenchmark, update_by={**unique_data, "run": run, "layout": layout}, **variable_data
            )

    session.flush()


def construct_xetla_benchmark_data(session: Session, data: pd.DataFrame, run: Run):
    tests_data: List[Dict] = (
        data[["batch", "m", "k", "n", "tflops", "hbm", "status"]]
        .drop_duplicates(subset=["batch", "m", "k", "n"])
        .copy()
        .to_dict("records")
    )

    for test_data in tests_data:
        unique_data, variable_data = split_unique_values(XetlaBenchmark, test_data)
        create_or_update(session, XetlaBenchmark, update_by={**unique_data, "run": run}, **variable_data)

    session.flush()


def prepare_transaction(session: Session, data: pd.DataFrame, construct_benchmark_data):
    grouped = data.groupby([col for col in run_columns if col in data.columns])

    for name, group in grouped:
        run_data = {col: convert_to_native_type(val) for col, val in zip(grouped.keys, name)}

        run = construct_run_item(session, run_data)

        construct_benchmark_data(session, group, run)


def cli_target():
    db_connection_string = os.getenv("DB_CONNECTION_STRING")

    engine = create_engine(f"postgresql://{db_connection_string}")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark-type",
        dest="benchmark_type",
        type=BenchmarkType,
        choices=BenchmarkType,
        help="Benchmark type",
        required=True,
    )
    parser.add_argument(
        "--csv-data-file", dest="csv_data_file", type=Path, help="Path to file with CSV data", required=True
    )
    args = parser.parse_args()

    data = pd.read_csv(args.csv_data_file)
    data = data.replace("", None)

    construct_benchmark_data = getattr(sys.modules[__name__], f"construct_{args.benchmark_type.value}_benchmark_data")

    db_session = sessionmaker(engine)

    with db_session() as session:
        prepare_transaction(session, data, construct_benchmark_data)
        session.commit()


if __name__ == "__main__":
    exit(cli_target())
