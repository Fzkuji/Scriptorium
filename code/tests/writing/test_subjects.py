"""Subject + kind -> Topic path, and what memory already holds."""

from pathlib import Path

import pytest

from memory.writing.subjects import known_subjects, path_for


def test_a_person_files_under_people():
    assert path_for("Dave Chen", "person") == "topics/people/dave-chen.md"


def test_each_kind_has_its_own_folder():
    assert path_for("Aquarium Club", "project") == (
        "topics/projects/aquarium-club.md"
    )
    assert path_for("Dave and Melanie", "relationship") == (
        "topics/relationships/dave-and-melanie.md"
    )
    assert path_for("Gardening", "theme") == "topics/themes/gardening.md"


def test_an_unknown_kind_falls_back_to_themes():
    assert path_for("Something", "not-a-real-kind") == (
        "topics/themes/something.md"
    )


def test_punctuation_collapses_to_one_hyphen():
    assert path_for("Dave  Chen! (Shanghai)", "person") == (
        "topics/people/dave-chen-shanghai.md"
    )


def test_a_subject_with_no_nameable_characters_is_refused():
    with pytest.raises(ValueError, match="has no name in it"):
        path_for("!!!", "person")


def test_an_empty_memory_says_so(tmp_path: Path):
    assert "empty" in known_subjects(tmp_path)


def test_known_subjects_lists_what_is_already_filed(tmp_path: Path):
    people = tmp_path / "topics" / "people"
    people.mkdir(parents=True)
    (people / "dave-chen.md").write_text("# Dave Chen\n", encoding="utf-8")
    themes = tmp_path / "topics" / "themes"
    themes.mkdir(parents=True)
    (themes / "gardening.md").write_text("# Gardening\n", encoding="utf-8")

    listed = known_subjects(tmp_path)

    assert "dave chen (people)" in listed
    assert "gardening (themes)" in listed


def test_a_subject_spelled_two_ways_still_files_as_one():
    """Reuse depends on the slug, not on how the model happens to spell it."""
    assert path_for("dave chen", "person") == path_for("Dave Chen", "person")
