"""A fact's inputs and its rendering into a paragraph and a footnote."""

import re

import pytest

from memory.writing.render import fact_block, insert, replacement_block, sources_of, when


def test_sources_of_keeps_the_ref_labels():
    assert sources_of({"sources": ["D1:1", " D1:2 "]}) == ["D1:1", "D1:2"]


def test_sources_of_refuses_an_empty_list():
    with pytest.raises(ValueError, match="sources is required"):
        sources_of({"sources": []})


def test_sources_of_refuses_blank_entries_only():
    with pytest.raises(ValueError, match="sources is required"):
        sources_of({"sources": ["  ", ""]})


def test_when_prefers_the_explicit_date():
    assert when({"date": "2023-05-23"}, "2026-01-01") == "2023-05-23"


def test_when_falls_back_to_the_batch_observation_date():
    assert when({}, "2026-01-01") == "2026-01-01"


def test_when_falls_back_to_today_as_a_last_resort():
    result = when({}, "")

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", result)


def test_fact_block_renders_the_paragraph_and_its_footnote():
    paragraph, note = fact_block(
        "Dave moved to Pudong.", ["D1:1"], "2026-08-09", "r0"
    )

    assert paragraph == (
        "Dave moved to Pudong.[^new-evidence-r0] ^new-block-r0\n"
    )
    assert note == (
        "[^new-evidence-r0]: Time: `2026-08-09`; Sources: D1:1\n"
    )


def test_fact_block_joins_multiple_sources():
    _, note = fact_block("A fact.", ["D1:1", "D1:2"], "2026-08-09", "r0")

    assert "Sources: D1:1, D1:2" in note


def test_replacement_block_links_to_the_fact_it_replaces():
    paragraph, note = replacement_block(
        "Dave moved to Hongqiao.", ["D2:1"], "2026-08-10", "u0", "abc123",
    )

    assert paragraph == (
        "Dave moved to Hongqiao. (replaces [the earlier note](#^abc123))"
        "[^new-evidence-u0] ^new-block-u0\n"
    )
    # The footnote itself is unaffected by the replaces clause: remember and
    # update cite and date their evidence the same way.
    assert note == fact_block(
        "irrelevant", ["D2:1"], "2026-08-10", "u0"
    )[1]


def test_insert_appends_to_the_end_of_a_file_with_no_footnotes_yet():
    body = insert("# Dave\n", "", "A fact. ^b1\n")

    assert body == "# Dave\n\nA fact. ^b1\n"


def test_insert_lands_above_the_footnote_definitions():
    body = (
        "# Dave\n\n"
        "First fact.[^e1] ^b1\n\n"
        "[^e1]: Time: `2026-01-01`; Sources: D1:1\n"
    )

    updated = insert(body, "", "Second fact.[^e2] ^b2\n")

    assert updated.index("Second fact.") < updated.index("[^e1]: Time:")


def test_insert_creates_a_heading_the_first_time_it_is_used():
    updated = insert("# Dave\n", "Hobbies", "Paints. ^b1\n")

    assert "## Hobbies" in updated
    assert updated.index("## Hobbies") < updated.index("Paints.")


def test_insert_reuses_an_existing_heading_instead_of_duplicating_it():
    body = insert("# Dave\n", "Hobbies", "Paints. ^b1\n")

    updated = insert(body, "Hobbies", "Also gardens. ^b2\n")

    assert updated.count("## Hobbies") == 1
    assert "Also gardens." in updated
