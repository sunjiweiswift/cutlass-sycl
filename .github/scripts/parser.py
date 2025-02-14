import argparse
import json
import re
from abc import abstractmethod
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from utils import get_logger

logger = get_logger(__name__)


class BenchmarkType(Enum):
    CUTLASS = "cutlass"
    XETLA = "xetla"


class Parser:
    def __init__(self, log_file: Path, run_info: Dict) -> None:
        self.log_file = log_file
        self.run_info = run_info

    @cached_property
    def contents(self):
        with open(self.log_file, "r") as file:
            log_content = file.readlines()
        return log_content

    @abstractmethod
    def get_test_runs(self) -> pd.DataFrame:
        pass

    def generate_csv(self):
        runs = self.get_test_runs()

        if runs.empty:
            logger.warning("No test runs found in log file")
            return

        for column, value in self.run_info.items():
            runs.insert(0, column, value)

        runs.to_csv(self.log_file.parent / f"{self.log_file.stem}.csv", index=False)


class CUTLASSParser(Parser):
    @cached_property
    def contents(self):
        with open(self.log_file, "r") as file:
            log_content = json.load(file)
        return log_content

    def get_test_runs(self) -> List[List[str]] | pd.DataFrame:
        df = pd.json_normalize(self.contents["benchmarks"])

        df["data_type"] = df["name"].str.split("/", expand=True)[1]

        df["status"] = df.apply(
            lambda row: (
                f"Failed ({row['error_message']})"
                if "error_occurred" in df.columns and row["error_occurred"] is not np.nan
                else "Passed"
            ),
            axis=1,
        )

        df = df.rename(columns={"l": "batch", "label": "layout"})

        df = df.fillna({"alpha": 0, "beta": 0, "m": 0, "k": 0, "n": 0, "batch": 0})
        df[["alpha", "beta", "m", "k", "n", "batch"]] = df[["alpha", "beta", "m", "k", "n", "batch"]].astype(int)

        df.drop(
            [
                "family_index",
                "per_family_instance_index",
                "run_name",
                "run_type",
                "repetitions",
                "repetition_index",
                "threads",
                "iterations",
                "error_occurred",
                "error_message",
            ],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        df = df.drop_duplicates(subset=["name", "alpha", "beta", "m", "k", "n", "batch", "data_type", "layout"])

        return df


class XeTLAParser(Parser):
    TEST_BLOCK_START_PATTERN = r"\[\s+RUN\s+\]"
    TEST_BLOCK_END_PATTERN = r"\[\s+(OK|FAILED)\s+\]"

    TEST_METHOD = r"(wg_swizzle_[mn]_first)"
    TEST_PROBLEM_SIZE = r"Problem size MKN:(\d+)x(\d+)x(\d+)"
    TEST_BATCH_SIZE = r"Running on test iter: (\d+)"
    TEST_PERFORMANCE_TFLOPS = r"Tflops\s*\[.*average:\s+(.*)\]"
    TEST_PERFORMANCE_HBM = r"HBM\(GBs\)\s*\[.*average:\s+(.*)\]"
    TEST_STATUS = r"(PASSED|FAILED)"

    DATA_COLUMNS = ("batch", "m", "k", "n", "tflops", "hbm", "status")

    def get_test_runs(self) -> pd.DataFrame:
        test_blocks = self.extract_test_blocks()
        test_runs = [self.parse_test_block(block) for block in test_blocks]
        test_runs = pd.DataFrame(test_runs, columns=self.DATA_COLUMNS)
        test_runs = test_runs.drop_duplicates()
        return test_runs

    def extract_test_blocks(self) -> List[List[str]]:
        test_blocks = []
        test_block: List[str] = []

        inside_test_block = False

        for line in self.contents:
            if re.search(self.TEST_BLOCK_START_PATTERN, line):
                inside_test_block = True
                test_block = []

            if inside_test_block:
                test_block.append(line)

            if re.search(self.TEST_BLOCK_END_PATTERN, line):
                if inside_test_block:
                    test_blocks.append(test_block)
                inside_test_block = False

        return test_blocks

    def parse_test_block(self, test_block: List[str]) -> Tuple[str, ...]:
        methods = {}
        for line in test_block:
            if test_method_match := re.search(self.TEST_METHOD, line):
                test_method = test_method_match.group(1)
                methods[test_method] = {}
                continue

            if test_problem_size_match := re.search(self.TEST_PROBLEM_SIZE, line):
                methods[test_method]["M"], methods[test_method]["K"], methods[test_method]["N"] = (
                    test_problem_size_match.groups()
                )
                continue

            if test_batch_size_match := re.search(self.TEST_BATCH_SIZE, line):
                methods[test_method]["batch"] = test_batch_size_match.group(1)
                continue

            if test_performance_tflops_match := re.search(self.TEST_PERFORMANCE_TFLOPS, line):
                methods[test_method]["tflops"] = test_performance_tflops_match.group(1)
                continue

            if test_performance_hbm_match := re.search(self.TEST_PERFORMANCE_HBM, line):
                methods[test_method]["hbm"] = test_performance_hbm_match.group(1)
                continue

            if test_status_match := re.search(self.TEST_STATUS, line):
                methods[test_method]["status"] = test_status_match.group(1)
                continue

        best_method = "wg_swizzle_n_first"
        try:
            if float(methods["wg_swizzle_n_first"]["tflops"]) < float(methods["wg_swizzle_m_first"]["tflops"]):
                best_method = "wg_swizzle_m_first"
        except KeyError:
            methods[best_method]["tflops"] = methods[best_method]["hbm"] = ""

        return (
            methods[best_method]["batch"],
            methods[best_method]["M"],
            methods[best_method]["K"],
            methods[best_method]["N"],
            methods[best_method]["tflops"],
            methods[best_method]["hbm"],
            methods[best_method]["status"],
        )


_PARSER_CLASS_MAPPING = {
    BenchmarkType.CUTLASS: CUTLASSParser,
    BenchmarkType.XETLA: XeTLAParser,
}


def cli_target():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark-type",
        dest="benchmark_type",
        type=BenchmarkType,
        choices=BenchmarkType,
        help="Benchmark type",
        required=True,
    )
    parser.add_argument("--log-file", dest="log_file", type=Path, help="Log file to parse", required=True)
    parser.add_argument(
        "--run-info",
        dest="run_info",
        nargs="*",
        type=lambda pair: pair.split("="),
        help="Run information",
        default={},
    )

    args = parser.parse_args()

    args.run_info = dict(args.run_info)

    parser_class = _PARSER_CLASS_MAPPING[args.benchmark_type]
    parser_instance = parser_class(args.log_file, args.run_info)
    parser_instance.generate_csv()


if __name__ == "__main__":
    exit(cli_target())
