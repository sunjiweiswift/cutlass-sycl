from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Platform(Base):
    __tablename__ = "platform"
    __table_args__ = {"schema": "cutlass_benchmarks"}

    platform_id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)


class Reference(Base):
    __tablename__ = "reference"
    __table_args__ = {"schema": "cutlass_benchmarks"}

    sha = Column(String(255), primary_key=True)
    branch = Column(String(255), nullable=False)


class RunType(Base):
    __tablename__ = "run_type"
    __table_args__ = {"schema": "cutlass_benchmarks"}

    run_type_id = Column(Integer, primary_key=True)
    type = Column(String(255), unique=True, nullable=False)


class DataType(Base):
    __tablename__ = "data_type"
    __table_args__ = {"schema": "cutlass_benchmarks"}

    data_type_id = Column(Integer, primary_key=True)
    type = Column(String(255), unique=True, nullable=False)


class Run(Base):
    __tablename__ = "run"
    __table_args__ = (
        UniqueConstraint("platform_id", "reference", "run_type_id", "data_type_id"),
        {"schema": "cutlass_benchmarks"},
    )

    run_id = Column(Integer, primary_key=True)
    run_type_id = Column(Integer, ForeignKey(RunType.run_type_id))
    reference = Column(String(255), ForeignKey(Reference.sha))
    platform_id = Column(Integer, ForeignKey(Platform.platform_id))
    data_type_id = Column(Integer, ForeignKey(DataType.data_type_id))

    run_type = relationship(RunType)
    reference_rel = relationship(Reference)
    platform = relationship(Platform)
    data_type = relationship(DataType)
