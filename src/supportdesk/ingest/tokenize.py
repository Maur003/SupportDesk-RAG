from typing import Protocol


class TokenCodec(Protocol):
    def encode(self, text: str) -> list[int]:
        """Content token ids. Do not include model special tokens."""

    def decode(self, token_ids: list[int]) -> str:
        """Join token ids back into text."""


class WhitespaceCodec:
    """Test codec: one whitespace-separated word is one token, and it round-trips."""

    def __init__(self) -> None:
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word not in self._word_to_id:
                self._word_to_id[word] = len(self._id_to_word)
                self._id_to_word.append(word)
            ids.append(self._word_to_id[word])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._id_to_word[token_id] for token_id in token_ids)


class HuggingFaceCodec:
    """WordPiece (or the model's own) tokenizer, without [CLS] / [SEP].

    bge-small-en-v1.5 allows 512 tokens including the two special tokens, so a
    400-token content chunk still fits at embed time.
    """

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        return list(encoded)

    def decode(self, token_ids: list[int]) -> str:
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=True))


def token_count(codec: TokenCodec, text: str) -> int:
    return len(codec.encode(text))
