import argparse
import logging
import os
import sys
from parser import BenchmarkType
from pathlib import Path
from typing import Dict, List

import pandas as pd
from models.common import DataType, Platform, Reference, Run, RunType
from models.cutlass import CUTLASSBenchmark, Layout
from models.xetla import XeTLABenchmark
from sqlalchemy import create_engine, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)


def insert_run(engine, data: pd.DataFrame):
    with Session(engine) as session:
        references_data: List[Dict] = data[["sha", "branch"]].drop_duplicates().to_dict("records")
        run_types_data: List[Dict] = (
            data[["run_type"]].drop_duplicates().rename(columns={"run_type": "type"}).to_dict("records")
        )
        platforms_data: List[Dict] = (
            data[["platform"]].drop_duplicates().rename(columns={"platform": "name"}).to_dict("records")
        )
        data_types_data: List[Dict] = (
            data[["data_type"]].drop_duplicates().rename(columns={"data_type": "type"}).to_dict("records")
        )

        session.execute(insert(Reference).on_conflict_do_nothing(index_elements=["sha"]), references_data)
        session.execute(insert(RunType).on_conflict_do_nothing(index_elements=["type"]), run_types_data)
        session.execute(insert(Platform).on_conflict_do_nothing(index_elements=["name"]), platforms_data)
        session.execute(insert(DataType).on_conflict_do_nothing(index_elements=["type"]), data_types_data)

        run_type_ids = {rt.type: rt.run_type_id for rt in session.query(RunType.type, RunType.run_type_id)}
        platform_ids = {p.name: p.platform_id for p in session.query(Platform.name, Platform.platform_id)}
        data_type_ids = {dt.type: dt.data_type_id for dt in session.query(DataType.type, DataType.data_type_id)}

        data["run_type_id"] = data["run_type"].map(run_type_ids)
        data["platform_id"] = data["platform"].map(platform_ids)
        data["data_type_id"] = data["data_type"].map(data_type_ids)

        run_data: List[Dict] = (
            data[["sha", "run_type_id", "platform_id", "data_type_id"]]
            .copy()
            .rename(columns={"sha": "reference"})
            .drop_duplicates()
            .to_dict("records")
        )
        session.execute(
            insert(Run).on_conflict_do_nothing(
                index_elements=["run_type_id", "reference", "platform_id", "data_type_id"]
            ),
            run_data,
        )

        session.commit()


def get_run_id(engine, data: pd.DataFrame):
    with Session(engine) as session:
        unique_runs = data[["run_type", "sha", "platform", "data_type"]].drop_duplicates().values.tolist()
        stmt = (
            select(Run.run_id, RunType.type, Reference.sha, Platform.name, DataType.type)
            .join(RunType, Run.run_type_id == RunType.run_type_id)
            .join(Reference, Run.reference == Reference.sha)
            .join(Platform, Run.platform_id == Platform.platform_id)
            .join(DataType, Run.data_type_id == DataType.data_type_id)
            .where(tuple_(RunType.type, Reference.sha, Platform.name, DataType.type).in_(unique_runs))
        )

        runs_data = pd.DataFrame(
            session.execute(stmt).fetchall(), columns=["run_id", "run_type", "sha", "platform", "data_type"]
        )

        session.commit()

        return pd.merge(data, runs_data, on=["run_type", "sha", "platform", "data_type"], how="left")


def insert_cutlass_benchmark_data(
    engine: Engine,
    data: pd.DataFrame,
):
    with Session(engine) as session:
        layout_data: List[Dict] = (
            data[["layout"]].drop_duplicates().rename(columns={"layout": "name"}).to_dict("records")
        )

        session.execute(insert(Layout).on_conflict_do_nothing(index_elements=["name"]), layout_data)

        layout_ids = {lt.name: lt.layout_id for lt in session.query(Layout.name, Layout.layout_id)}

        data["layout_id"] = data["layout"].map(layout_ids)

        tests_data: List[Dict] = (
            data[
                [
                    "run_id",
                    "layout_id",
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

        tests_stmt = insert(CUTLASSBenchmark).values(tests_data)

        tests_stmt = tests_stmt.on_conflict_do_update(
            index_elements=["run_id", "layout_id", "name", "alpha", "beta", "batch", "m", "k", "n"],
            set_={
                "real_time": tests_stmt.excluded.real_time,
                "cpu_time": tests_stmt.excluded.cpu_time,
                "total_runtime_ms": tests_stmt.excluded.total_runtime_ms,
                "avg_runtime_ms": tests_stmt.excluded.avg_runtime_ms,
                "avg_tflops": tests_stmt.excluded.avg_tflops,
                "avg_throughput": tests_stmt.excluded.avg_throughput,
                "best_bandwidth": tests_stmt.excluded.best_bandwidth,
                "best_runtime_ms": tests_stmt.excluded.best_runtime_ms,
                "best_tflop": tests_stmt.excluded.best_tflop,
                "status": tests_stmt.excluded.status,
            },
        )

        session.execute(tests_stmt)

        session.commit()


def insert_xetla_benchmark_data(
    engine: Engine,
    data: pd.DataFrame,
):
    with Session(engine) as session:
        tests_data: List[Dict] = (
            data[["run_id", "batch", "m", "k", "n", "tflops", "hbm", "status"]]
            .drop_duplicates(subset=["run_id", "batch", "m", "k", "n"])
            .copy()
            .to_dict("records")
        )

        tests_stmt = insert(XeTLABenchmark).values(tests_data)

        tests_stmt = tests_stmt.on_conflict_do_update(
            index_elements=["run_id", "batch", "m", "k", "n"],
            set_={
                "tflops": tests_stmt.excluded.tflops,
                "hbm": tests_stmt.excluded.hbm,
                "status": tests_stmt.excluded.status,
            },
        )

        session.execute(tests_stmt)

        session.commit()


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

    insert_run(engine, data)
    data = get_run_id(engine, data)  # get run_id column according to run details

    insert_benchmark_data = getattr(sys.modules[__name__], f"insert_{args.benchmark_type.value}_benchmark_data")

    insert_benchmark_data(engine, data)


if __name__ == "__main__":
    exit(cli_target())
