from supportdesk.embed_text import embedding_input


def test_embedding_input_joins_title_heading_and_text() -> None:
    assert (
        embedding_input("Fiber 500 residential plan", "Monthly charge", "The price is 79.")
        == "Fiber 500 residential plan\nMonthly charge\nThe price is 79."
    )


def test_embedding_input_skips_empty_parts() -> None:
    assert embedding_input("Title", "", "Body only.") == "Title\nBody only."
