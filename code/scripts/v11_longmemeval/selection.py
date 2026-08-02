"""LongMemEval item ordering and question-type selection."""

from itertools import zip_longest

from scripts import run_v88_gpt55_longmemeval as common


def round_robin_indices(data: list[dict]) -> list[int]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(data):
        groups.setdefault(item["question_type"], []).append(index)
    return [
        index
        for row in zip_longest(*groups.values())
        for index in row
        if index is not None
    ]


def question_type_indices(
    data: list[dict],
    question_type: str,
    start: int,
    limit: int,
    *,
    reverse: bool,
) -> list[int]:
    matching = [
        index
        for index, item in enumerate(data)
        if item["question_type"] == question_type
    ]
    if not matching:
        raise common.DataValidationError(
            f"unknown --question-type: {question_type}"
        )
    if reverse:
        matching.reverse()
    positions = common.select_indices(len(matching), start, limit)
    return [matching[position] for position in positions]
