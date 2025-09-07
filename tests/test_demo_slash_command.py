"""
Demo test file to demonstrate the /verify-test slash command
"""


def test_assertion_failure():
    """Test that will fail with assertion error"""
    result = [1, 2, 3, 4]
    expected = [1, 2, 3, 5]  # Intentionally wrong
    assert result == expected, f"Expected {expected} but got {result}"


def test_import_error():
    """Test that will fail with import error"""
    import nonexistent_module  # This will fail


def test_passes():
    """Test that should pass"""
    assert 2 + 2 == 4
