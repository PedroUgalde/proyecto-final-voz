"""
Beam Search con lexicón de palabras válidas.

Explora múltiples hipótesis de transcripción y prioriza
secuencias que formen palabras del vocabulario.
"""

import torch

from .config import CHAR_TO_IDX, IDX_TO_CHAR, TARGET_WORDS
from .predict_ctc import greedy_decode


def build_lexicon(words: list[str] | None = None) -> dict[str, list[int]]:
    """Lexicón: palabra → secuencia de índices de caracteres."""
    words = words or TARGET_WORDS
    blank = CHAR_TO_IDX["<blank>"]
    lex = {}
    for w in words:
        lex[w] = [blank] + [CHAR_TO_IDX[c] for c in w] + [blank]
    return lex


def ctc_beam_search(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    vocabulary: list[str] | None = None,
) -> str:
    """
    Beam search CTC con prefijos.

    Args:
        log_probs: (T, num_classes) probabilidades log
    """
    vocabulary = vocabulary or TARGET_WORDS
    blank = CHAR_TO_IDX["<blank>"]
    T, C = log_probs.shape
    log_probs = log_probs.cpu().numpy()

    # beams: {prefix_str: (log_prob_blank_ending, log_prob_non_blank_ending)}
    beams = {"": (0.0, -float("inf"))}

    for t in range(T):
        new_beams: dict[str, tuple[float, float]] = {}

        for prefix, (pb, pnb) in beams.items():
            for c in range(C):
                lp = log_probs[t, c]

                if c == blank:
                    nb, nnb = new_beams.get(prefix, (-float("inf"), -float("inf")))
                    nb = _logsumexp(nb, pb + lp, pnb + lp)
                    new_beams[prefix] = (nb, nnb)
                else:
                    char = IDX_TO_CHAR[c]
                    new_prefix = prefix + char

                    if prefix and char == prefix[-1]:
                        # Mismo carácter: extensión o repetición
                        _, nnb_old = new_beams.get(new_prefix, (-float("inf"), -float("inf")))
                        nnb = _logsumexp(nnb_old, pnb + lp)
                        new_beams[new_prefix] = new_beams.get(new_prefix, (-float("inf"), -float("inf")))
                        new_beams[new_prefix] = (new_beams[new_prefix][0], nnb)

                        repeat_prefix = prefix  # repetición → mismo prefijo
                        pb_r, pnb_r = new_beams.get(repeat_prefix, (-float("inf"), -float("inf")))
                        pnb_r = _logsumexp(pnb_r, pb + lp)
                        new_beams[repeat_prefix] = (pb_r, pnb_r)
                    else:
                        pb_old, nnb_old = new_beams.get(new_prefix, (-float("inf"), -float("inf")))
                        nnb = _logsumexp(nnb_old, pb + lp, pnb + lp)
                        new_beams[new_prefix] = (pb_old, nnb)

        # Mantener top beam_width
        scored = [
            (pref, _logsumexp(pb, pnb))
            for pref, (pb, pnb) in new_beams.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {pref: new_beams[pref] for pref, _ in scored[:beam_width]}

    best_prefix = max(beams, key=lambda p: _logsumexp(*beams[p]))

    # Priorizar palabras del lexicón
    if best_prefix in vocabulary:
        return best_prefix

    candidates = sorted(
        ((w, _lexicon_score(w, beams)) for w in vocabulary),
        key=lambda x: x[1],
        reverse=True,
    )
    return candidates[0][0] if candidates else best_prefix


def _logsumexp(*args: float) -> float:
    import numpy as np
    arr = np.array([a for a in args if a > -1e10])
    if len(arr) == 0:
        return -float("inf")
    m = arr.max()
    return float(m + np.log(np.sum(np.exp(arr - m))))


def _lexicon_score(word: str, beams: dict) -> float:
    if word in beams:
        return _logsumexp(*beams[word])
    # Penalizar palabras no vistas en beams
    for pref, scores in beams.items():
        if word.startswith(pref):
            return _logsumexp(*scores) - 5.0
    return -float("inf")


def predict_beam_search(
    model,
    audio_path: str,
    beam_width: int = 10,
    device=None,
) -> str:
    """Predice con beam search + lexicón."""
    from .dataset import extract_mfcc_file

    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    mfcc = extract_mfcc_file(audio_path)
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    x_len = torch.tensor([mfcc.shape[0]], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, x_len)[0]
        log_probs = logits.log_softmax(dim=-1)

    result = ctc_beam_search(log_probs, beam_width=beam_width)
    if not result:
        result = greedy_decode(logits)
    return result
