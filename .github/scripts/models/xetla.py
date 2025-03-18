from typing import Optional

from models.common import CommonBaseModel, Run
from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship


class XetlaBenchmark(CommonBaseModel):
    __table_args__ = (UniqueConstraint("run_id", "batch", "m", "k", "n"),)

    test_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey(Run.run_id))
    batch: Mapped[int] = mapped_column(Integer)
    m: Mapped[int] = mapped_column(Integer)
    k: Mapped[int] = mapped_column(Integer)
    n: Mapped[int] = mapped_column(Integer)
    tflops: Mapped[Optional[float]] = mapped_column(Float)
    hbm: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(255))

    run = relationship(Run)
