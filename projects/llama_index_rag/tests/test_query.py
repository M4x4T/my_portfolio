import pytest
from rag_core.query import ask_with_sources

def test_customer_never_receives_internal_nodes():
    response = ask_with_sources("Which tools for internal use are available?", "customer")
    assert all(source["access_level"] != "internal" for source in response["sources"])

def test_internal_can_receive_internal_nodes():
    response = ask_with_sources("Which tools for internal use are available?", "internal")
    access_levels = [source["access_level"] for source in response["sources"]]
    assert "internal" in access_levels


def test_invalid_role_raises_error():
    with pytest.raises(ValueError, match="Invalid user role"):
        ask_with_sources("Which tools for internal use are available?", "hacker")