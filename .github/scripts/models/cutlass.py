from typing import Optional

from models.common import CommonBaseModel, Run
from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Layout(CommonBaseModel):
    layout_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)


class TestConfiguration(CommonBaseModel):
    test_configuration_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parameters: Mapped[str] = mapped_column(String(500), unique=True)


class TestGroup(CommonBaseModel):
    test_group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tag: Mapped[str] = mapped_column(String(255), unique=True)


class CutlassBenchmarkV2(CommonBaseModel):
    __table_args__ = (UniqueConstraint("run_id", "layout_id", "name", "test_configuration_id", "test_group_id"),)

    test_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey(Run.run_id))
    layout_id: Mapped[int] = mapped_column(ForeignKey(Layout.layout_id))
    test_configuration_id: Mapped[int] = mapped_column(ForeignKey(TestConfiguration.test_configuration_id))
    test_group_id: Mapped[int] = mapped_column(ForeignKey(TestGroup.test_group_id))
    name: Mapped[str] = mapped_column(String(255))
    real_time: Mapped[float] = mapped_column(Float)
    cpu_time: Mapped[float] = mapped_column(Float)
    total_runtime_ms: Mapped[Optional[float]] = mapped_column(Float)
    avg_runtime_ms: Mapped[Optional[float]] = mapped_column(Float)
    avg_tflops: Mapped[Optional[float]] = mapped_column(Float)
    avg_throughput: Mapped[Optional[float]] = mapped_column(Float)
    best_bandwidth: Mapped[Optional[float]] = mapped_column(Float)
    best_runtime_ms: Mapped[Optional[float]] = mapped_column(Float)
    best_tflop: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(255))

    run: Mapped[Run] = relationship(Run)
    layout: Mapped[Layout] = relationship(Layout)
    test_configuration: Mapped[TestConfiguration] = relationship(TestConfiguration)
    test_group: Mapped[TestGroup] = relationship(TestGroup)
