from benchmark.quixbugs import classify_test_result, parse_pytest_counts


def test_parse_pytest_counts():
    assert parse_pytest_counts("2 failed, 3 passed in 0.10s") == (3, 2, 5)


def test_classify_assertion_failure():
    category = classify_test_result(1, "FAILED test_x.py::test_y - AssertionError", "", False)
    assert category == "assertion_failure"

