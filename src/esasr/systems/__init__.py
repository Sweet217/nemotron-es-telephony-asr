"""System registry. Systems are imported lazily so optional deps stay optional."""
from __future__ import annotations

from .base import ASRSystem


def make_system(spec: str) -> ASRSystem:
    """Build a system from a short spec string.

    nemotron[:LANG[:LOOKAHEAD]]   e.g. nemotron, nemotron:auto, nemotron:es-US:0
    nemotron-nemo[:LANG[:PATH.nemo]]  NeMo runtime (CUDA if available, else CPU)
    whisper[:HF_REPO[@auto]]      e.g. whisper, whisper:mlx-community/whisper-large-v3-turbo
    deepgram[:LANG]               needs DEEPGRAM_API_KEY
    """
    kind, *rest = spec.split(":")
    if kind == "nemotron":
        from .nemotron_hf import NemotronHF

        lang = rest[0] if rest else "es-US"
        la = int(rest[1]) if len(rest) > 1 else 3
        return NemotronHF(language=lang, num_lookahead_tokens=la)
    if kind == "nemotron-nemo":
        from .nemotron_nemo import NemotronNeMo

        lang = rest[0] if rest else "es-US"
        path = ":".join(rest[1:]) if len(rest) > 1 else None  # optional local .nemo (e.g. fine-tuned)
        return NemotronNeMo(language=lang, model=path) if path else NemotronNeMo(language=lang)
    if kind == "whisper":
        from .whisper_mlx import WhisperMLX

        # "whisper" | "whisper:<hf-repo>" ; language is Spanish unless the repo spec ends in "@auto"
        repo = spec.split(":", 1)[1] if ":" in spec else "mlx-community/whisper-large-v3-turbo"
        lang = "es"
        if repo.endswith("@auto"):
            repo, lang = repo[: -len("@auto")], None
        return WhisperMLX(model=repo, language=lang)
    if kind == "deepgram":
        from .deepgram import DeepgramNova3

        return DeepgramNova3(language=rest[0] if rest else "es")
    raise ValueError(f"unknown system spec {spec!r}")
