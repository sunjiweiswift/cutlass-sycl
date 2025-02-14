from models.common import Run
from sqlalchemy import Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class XeTLABenchmark(Base):
    __tablename__ = "xetla_benchmark"
    __table_args__ = (UniqueConstraint("run_id", "batch", "m", "k", "n"), {"schema": "cutlass_benchmarks"})

    test_id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey(Run.run_id))
    batch = Column(Integer)
    m = Column(Integer)
    k = Column(Integer)
    n = Column(Integer)
    tflops = Column(Float)
    hbm = Column(Float)
    status = Column(String(255))

    run = relationship(Run)
