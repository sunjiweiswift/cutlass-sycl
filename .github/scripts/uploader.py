import argparse
import json
import logging
import os
import sys
from parser import BenchmarkType
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from models.common import CommonBaseModel, ComponentSet, ComponentsVersion, DataType, Platform, Reference, Run, RunType
from models.cutlass import CutlassBenchmarkV2, Layout, TestConfiguration, TestGroup
from models.utils import create_or_update, get_or_create, split_unique_values
from models.xetla import XetlaBenchmark
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

ignore = CommonBaseModel

logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

run_columns = [
    "run_type",
    "sha",
    "branch",
    "platform",
    "data_type",
    "workflow",
    "compiler",
    "driver",
    "c_compiler_version",
    "cxx_compiler_version",
    "driver_version",
]
cutlass_result_columns = [
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
    "status",
]


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
    component_set = get_or_create(
        session, ComponentSet, configuration=json.dumps(dict(compiler=data["compiler"], driver=data["driver"]))
    )
    components_version = get_or_create(
        session,
        ComponentsVersion,
        configuration=json.dumps(
            dict(
                c_compiler_version=data["c_compiler_version"],
                cxx_compiler_version=data["cxx_compiler_version"],
                driver_version=data["driver_version"],
            )
        ),
    )

    run = get_or_create(
        session,
        Run,
        run_type=run_type,
        reference_rel=reference,
        platform=platform,
        data_type=data_type,
        workflow=data["workflow"],
        component_set=component_set,
        components_version=components_version,
    )

    session.flush()

    return run


def construct_cutlass_benchmark_data(session: Session, data: pd.DataFrame, run: Run):

    test_group = get_or_create(session, TestGroup, tag=data["tag"].iloc[0])

    layout_grouped = data.groupby(["layout"])

    for name, layout_group in layout_grouped:
        layout_data = {col: convert_to_native_type(val) for col, val in zip(layout_grouped.keys, name)}
        layout = get_or_create(session, Layout, name=layout_data["layout"])

        config_columns = layout_group.columns.drop(cutlass_result_columns + run_columns + ["layout", "tag"]).to_list()
        config_grouped = layout_group.groupby([col for col in config_columns])

        for params, config_group in config_grouped:
            config_data = {}
            for col, val in zip(config_grouped.keys, params):
                converted_val = convert_to_native_type(val)
                if converted_val != type(converted_val)(-1):
                    config_data[col] = converted_val

            test_configuration = get_or_create(session, TestConfiguration, parameters=json.dumps(config_data))

            tests_data: List[Dict] = config_group[cutlass_result_columns].replace(-1, 0).copy().to_dict("records")

            for test_data in tests_data:
                unique_data, variable_data = split_unique_values(CutlassBenchmarkV2, test_data)
                create_or_update(
                    session,
                    CutlassBenchmarkV2,
                    update_by={
                        **unique_data,
                        "run": run,
                        "layout": layout,
                        "test_configuration": test_configuration,
                        "test_group": test_group,
                    },
                    **variable_data,
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
