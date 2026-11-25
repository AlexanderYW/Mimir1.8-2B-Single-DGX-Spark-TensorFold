"""The prompts tools/check.py and tools/reference.py share. Standard library only."""

# a long prompt: past the dense engine's stock 1,024-row prompt chunk, so the PrefixLM prompt must run as one chunk
_FACTS = [
    "Københavns Rådhus blev indviet i 1905 og tegnet af Martin Nyrop.",
    "Øresundsbroen åbnede for trafik den 1. juli 2000.",
    "Den Lille Havfrue blev afsløret på Langelinie i 1913.",
    "Rundetårn blev bygget under Christian IV og stod færdigt i 1642.",
    "Tivoli åbnede første gang den 15. august 1843.",
]
_LONG = " ".join(_FACTS[i % len(_FACTS)] for i in range(110)) + (
    " Kodeordet til arkivet er BLÅBÆR-47. " + " ".join(_FACTS[(i + 2) % len(_FACTS)] for i in range(110))
    + "\n\nHvad er kodeordet til arkivet? Svar kun med kodeordet.")

PROMPTS = [
    ("dansk", "Forklar kort, hvordan en hashtabel håndterer kollisioner."),
    ("english", "Explain briefly how a hash map handles collisions."),
    ("fakta", "Hvad er hovedstaden i Danmark, og hvilken flod eller havn ligger byen ved? Svar i to sætninger."),
    ("regning", "Hvad er 17 * 23? Svar kun med tallet."),
    ("lang", _LONG),
]
