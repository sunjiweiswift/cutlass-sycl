from models.utils import to_snake_case
from sqlalchemy import ForeignKey, Integer, MetaData, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column, relationship


class CommonBaseModel(DeclarativeBase):
    metadata = MetaData(schema="cutlass_benchmarks")

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return to_snake_case(cls.__name__)


class Platform(CommonBaseModel):
    platform_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)


class Reference(CommonBaseModel):
    sha: Mapped[str] = mapped_column(String(255), primary_key=True)
    branch: Mapped[str] = mapped_column(String(255), nullable=False)


class RunType(CommonBaseModel):
    run_type_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)


class DataType(CommonBaseModel):
    data_type_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)


class ComponentSet(CommonBaseModel):
    component_set_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    configuration: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)


class ComponentsVersion(CommonBaseModel):
    components_version_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    configuration: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)


class Run(CommonBaseModel):
    __table_args__ = (
        UniqueConstraint(
            "platform_id",
            "reference",
            "run_type_id",
            "data_type_id",
            "workflow",
            "component_set_id",
            "components_version_id",
        ),
    )

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_type_id: Mapped[int] = mapped_column(ForeignKey(RunType.run_type_id))
    reference: Mapped[str] = mapped_column(ForeignKey(Reference.sha))
    platform_id: Mapped[int] = mapped_column(ForeignKey(Platform.platform_id))
    data_type_id: Mapped[int] = mapped_column(ForeignKey(DataType.data_type_id))
    workflow: Mapped[str] = mapped_column(String(255), nullable=False)
    component_set_id: Mapped[int] = mapped_column(ForeignKey(ComponentSet.component_set_id))
    components_version_id: Mapped[int] = mapped_column(ForeignKey(ComponentsVersion.components_version_id))

    run_type: Mapped[RunType] = relationship(RunType)
    reference_rel: Mapped[Reference] = relationship(Reference)
    platform: Mapped[Platform] = relationship(Platform)
    data_type: Mapped[DataType] = relationship(DataType)
    component_set: Mapped[DataType] = relationship(ComponentSet)
    components_version: Mapped[DataType] = relationship(ComponentsVersion)
