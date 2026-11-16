"""Make narration text speakable: math symbols become the words a lecturer says.

TTS engines read ``A = U D V^T`` as "A equals udv to the power of t" and
``||Av||`` as noise. ``speakable`` rewrites the common notation into spoken
English before any voice sees it (OpenAI TTS and the Gemini presenter). It is
deterministic and conservative: plain prose passes through unchanged, and the
script text itself is never modified (only what is spoken).
"""

import re

_NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
            "6": "six", "7": "seven", "8": "eight", "9": "nine"}
_GREEK = {
    "σ": "sigma", "Σ": "Sigma", "λ": "lambda", "θ": "theta", "μ": "mu", "π": "pi",
    "α": "alpha", "β": "beta", "ε": "epsilon", "δ": "delta", "Δ": "Delta",
}
_SYMBOLS = [
    ("≥", " is at least "), ("≤", " is at most "), ("≈", " is about "), ("≠", " is not equal to "),
    ("→", " goes to "), ("×", " times "), ("·", " times "), ("∞", " infinity "), ("√", " the square root of "),
    ("ᵀ", " transpose"), ("²", " squared"), ("³", " cubed"), ("⊥", " perpendicular to "),
]
_LATEX_NAMES = re.compile(r"\\(sigma|Sigma|lambda|theta|mu|pi|alpha|beta|epsilon|delta|Delta)\b")


def _subscript(match: re.Match) -> str:
    base, sub = match.group(1), match.group(2)
    if sub.isdigit():
        return f"{base} " + " ".join(_NUMBERS[d] for d in sub)
    if sub in ("{k+1}", "k+1"):
        return f"{base} sub k plus one"
    return f"{base} sub {sub.strip('{}')}"


def speakable(text: str) -> str:
    out = text or ""
    if not re.search(r"[=^_|\\$≥≤≈≠→×·∞√ᵀ²³⊥σΣλθμπαβεδΔ]", out):
        return out  # plain prose: nothing to do
    out = out.replace("$", "")
    out = _LATEX_NAMES.sub(lambda m: m.group(1), out)
    for char, word in _GREEK.items():
        out = out.replace(char, f" {word} ")
    for char, word in _SYMBOLS:
        out = out.replace(char, word)
    out = re.sub(r"\|\|\s*([^|]+?)\s*\|\|(?:_?F)?", r"the norm of \1", out)
    # Superscripts: braced or a bare token; never swallow the following space.
    out = re.sub(r"\^(?:\{\s*T\s*\}|T(?![A-Za-z]))", " transpose", out)
    out = re.sub(r"\^(?:\{\s*-1\s*\}|-1)", " inverse", out)
    out = re.sub(r"\^(?:\{\s*2\s*\}|2(?!\d))", " squared", out)
    out = re.sub(r"\^(?:\{\s*([A-Za-z0-9+\-]+)\s*\}|([A-Za-z0-9]+))",
                 lambda m: f" to the power {m.group(1) or m.group(2)}", out)
    out = re.sub(r"\b([A-Za-z]+)\s*_\s*(\{k\+1\}|\{?[A-Za-z0-9]{1,3}\}?)", _subscript, out)
    out = re.sub(r"\s*=\s*", " equals ", out)
    out = re.sub(r"\s*\+\s*", " plus ", out)
    out = re.sub(r"\s+[-−]\s+", " minus ", out)
    out = out.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s{2,}", " ", out)).strip()
