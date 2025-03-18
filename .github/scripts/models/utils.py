import re
from typing import Dict, List

from sqlalchemy import UniqueConstraint, select, update
from sqlalchemy.orm import Session


def split_unique_values[T](model: type[T], data: dict[str, object]) -> List[Dict]:
    unique_data = {}
    variable_data = {}
    unique_fields = []

    for constraint in model.__table__.constraints:
        if isinstance(constraint, UniqueConstraint):
            unique_fields.extend(constraint.columns.keys())

    for field in data.keys():
        if field in unique_fields:
            unique_data[field] = data[field]
        else:
            variable_data[field] = data[field]

    return [unique_data, variable_data]


def to_snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def create_or_update[T](  # type: ignore
    session: Session, model: type[T], update_by: dict[str, object] | None = None, **kwargs: object
):  # type: ignore
    if update_by is None:
        update_by = kwargs

    stmt = update(model).returning(model).filter_by(**update_by).values(**kwargs)
    instance = session.execute(stmt).scalars().first()

    if instance:
        return instance

    combined_data = {**update_by, **kwargs}
    instance = model(**combined_data)
    session.add(instance)
    return instance


def get_or_create[T](  # type: ignore
    session: Session, model: type[T], lookup_by: dict[str, object] | None = None, **kwargs: object
) -> T:  # type: ignore
    if lookup_by is None:
        lookup_by = kwargs

    stmt = select(model).filter_by(**lookup_by)
    instance = session.execute(stmt).scalars().first()

    if instance:
        return instance

    instance = model(**kwargs)
    session.add(instance)
    return instance
