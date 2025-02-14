from models.common import Run
from sqlalchemy import Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Layout(Base):
    __tablename__ = "layout"
    __table_args__ = {"schema": "cutlass_benchmarks"}

    layout_id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)


class CUTLASSBenchmark(Base):
    __tablename__ = "cutlass_benchmark"
    __table_args__ = (
        UniqueConstraint("run_id", "layout_id", "alpha", "beta", "batch", "m", "k", "n"),
        {"schema": "cutlass_benchmarks"},
    )

    test_id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey(Run.run_id))
    layout_id = Column(Integer, ForeignKey(Layout.layout_id))
    name = Column(String(255))
    real_time = Column(Float)
    cpu_time = Column(Float)
    total_runtime_ms = Column(Float)
    avg_runtime_ms = Column(Float)
    avg_tflops = Column(Float)
    avg_throughput = Column(Float)
    best_bandwidth = Column(Float)
    best_runtime_ms = Column(Float)
    best_tflop = Column(Float)
    alpha = Column(Integer)
    beta = Column(Integer)
    batch = Column(Integer)
    m = Column(Integer)
    k = Column(Integer)
    n = Column(Integer)
    status = Column(String(255))

    run = relationship(Run)
    layout = relationship(Layout)
