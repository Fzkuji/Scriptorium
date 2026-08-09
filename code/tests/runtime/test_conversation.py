from memory.conversation import normalize_date


def test_normalize_locomo_date():
    assert normalize_date("1:56 pm on 8 May, 2023") == "2023-05-08"


def test_normalize_longmemeval_slash_date():
    assert normalize_date("2023/05/30 (Tue) 23:40") == "2023-05-30"


def test_normalize_iso_date_with_time():
    assert normalize_date("2024-01-05T12:30:00Z") == "2024-01-05"


def test_normalize_rejects_invalid_iso_date():
    assert normalize_date("2023/02/30 (Thu)") == "2023/02/30 (Thu)"
