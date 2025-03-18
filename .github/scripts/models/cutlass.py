from typing import Optional

from models.common import CommonBaseModel, Run
from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Layout(CommonBaseModel):
    layout_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)


class CutlassBenchmark(CommonBaseModel):
    __table_args__ = (UniqueConstraint("run_id", "layout_id", "name", "alpha", "beta", "batch", "m", "k", "n"),)

    test_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey(Run.run_id))
    layout_id: Mapped[int] = mapped_column(ForeignKey(Layout.layout_id))
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
    alpha: Mapped[int] = mapped_column(Integer)
    beta: Mapped[int] = mapped_column(Integer)
    batch: Mapped[int] = mapped_column(Integer)
    m: Mapped[int] = mapped_column(Integer)
    k: Mapped[int] = mapped_column(Integer)
    n: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(255))

    run: Mapped[Run] = relationship(Run)
    layout: Mapped[Layout] = relationship(Layout)
