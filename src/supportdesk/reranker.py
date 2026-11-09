"""Cross-encoder reranker. Scores are raw logits, not probabilities."""

import math

import torch

from supportdesk.config import RERANK_MODEL


class MiniLMReranker:
    def __init__(self, model_name: str, revision: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.revision = revision
        self.model_id = f"{model_name}@{revision}"
        # Identity keeps the ms-marco logit. The default activation is a sigmoid.
        self._model = CrossEncoder(
            model_name,
            revision=revision,
            device="cpu",
            activation_fn=torch.nn.Identity(),
        )
        _check_raw_logit(self._model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One raw logit per passage. The query is the question, with no prefix."""

        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        raw = self._model.predict(
            pairs,
            activation_fn=torch.nn.Identity(),
            apply_softmax=False,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )
        values = [float(value) for value in _flat(raw)]
        if len(values) != len(passages):
            raise RuntimeError(
                f"reranker returned {len(values)} scores for {len(passages)} passages"
            )
        return values


def _flat(raw: object) -> list[float]:
    if hasattr(raw, "reshape"):
        return [float(value) for value in raw.reshape(-1).tolist()]
    return [float(raw)]


def _check_raw_logit(model: object) -> None:
    """Fail if predict applies a sigmoid. A logit and its sigmoid must agree."""

    pair = [("northline quota", "a household bill")]
    logit = _flat(
        model.predict(
            pair,
            activation_fn=torch.nn.Identity(),
            apply_softmax=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    )[0]
    probability = _flat(
        model.predict(
            pair,
            activation_fn=torch.nn.Sigmoid(),
            apply_softmax=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    )[0]
    expected = 1.0 / (1.0 + math.exp(-logit))
    if abs(expected - probability) > 1e-4:
        raise RuntimeError(f"{RERANK_MODEL} did not return a raw logit")
