"""Send a short question to the cheaper model id.

Length is the normalized question, the same string the exact cache hashes.
"""

from supportdesk.cache import normalize_query


def choose_model(
    query: str,
    *,
    simple_max_chars: int,
    cheap_model: str,
    generator_model: str,
) -> str:
    if len(normalize_query(query)) <= simple_max_chars:
        return cheap_model
    return generator_model
