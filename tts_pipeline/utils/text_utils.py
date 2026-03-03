"""
French medical text normalisation utilities.

Covers:
  - Abbreviation expansion (medical + general French)
  - Unicode normalisation
  - Lowercasing
  - Special-character removal (keeps accented French letters)
  - Whitespace collapsing
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict

# ---------------------------------------------------------------------------
# Abbreviation tables
# ---------------------------------------------------------------------------

#: French medical abbreviations → their spoken form.
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    r"\bDr\.?\b":    "docteur",
    r"\bDrs\.?\b":   "docteurs",
    r"\bPr\.?\b":    "professeur",
    r"\bMme\.?\b":   "madame",
    r"\bMmes\.?\b":  "mesdames",
    r"\bM\.?\b":     "monsieur",
    r"\bMM\.?\b":    "messieurs",
    # Units
    r"\bbpm\b":      "battements par minute",
    r"\bmg/kg\b":    "milligrammes par kilogramme",
    r"\bmg/L\b":     "milligrammes par litre",
    r"\bmg\b":       "milligrammes",
    r"\bml\b":       "millilitres",
    r"\bcL\b":       "centilitres",
    r"\bL\b":        "litres",
    r"\bcm\b":       "centimètres",
    r"\bmm\b":       "millimètres",
    r"\bμm\b":       "micromètres",
    r"\bnm\b":       "nanomètres",
    r"\bkg\b":       "kilogrammes",
    r"\bg\b":        "grammes",
    r"\bμg\b":       "microgrammes",
    r"\bng\b":       "nanogrammes",
    r"\bmmHg\b":     "millimètres de mercure",
    r"\bkPa\b":      "kilopascals",
    r"\b°C\b":       "degrés celsius",
    r"\b°F\b":       "degrés fahrenheit",
    r"\bU/L\b":      "unités par litre",
    r"\bmEq/L\b":    "milliéquivalents par litre",
    r"\bmmol/L\b":   "millimoles par litre",
    r"\bµmol/L\b":   "micromoles par litre",
    # Routes
    r"\bIV\b":       "intraveineuse",
    r"\bIM\b":       "intramusculaire",
    r"\bSC\b":       "sous-cutané",
    r"\bVO\b":       "voie orale",
    r"\bSL\b":       "sublinguale",
    r"\bIN\b":       "intranasale",
    # Clinical abbreviations
    r"\bSAU\b":      "service des urgences",
    r"\bUSI\b":      "unité de soins intensifs",
    r"\bREA\b":      "réanimation",
    r"\bECG\b":      "électrocardiogramme",
    r"\bEEG\b":      "électroencéphalogramme",
    r"\bIRM\b":      "imagerie par résonance magnétique",
    r"\bTDM\b":      "tomodensitométrie",
    r"\bRx\b":       "radiographie",
    r"\bHTA\b":      "hypertension artérielle",
    r"\bDT2\b":      "diabète de type deux",
    r"\bDT1\b":      "diabète de type un",
    r"\bIMC\b":      "indice de masse corporelle",
    r"\bFC\b":       "fréquence cardiaque",
    r"\bFR\b":       "fréquence respiratoire",
    r"\bTA\b":       "tension artérielle",
    r"\bSaO2\b":     "saturation en oxygène",
    r"\bSp?O2\b":    "saturation en oxygène",
    r"\bFiO2\b":     "fraction inspirée en oxygène",
    r"\bPaO2\b":     "pression partielle en oxygène",
    r"\bPaCO2\b":    "pression partielle en dioxyde de carbone",
    r"\bINR\b":      "rapport normalisé international",
    r"\bTP\b":       "taux de prothrombine",
    r"\bCRP\b":      "protéine C-réactive",
    r"\bNFS\b":      "numération formule sanguine",
    r"\bBU\b":       "bandelette urinaire",
    r"\bASA\b":      "acide acétylsalicylique",
    r"\bACE\b":      "enzyme de conversion de l'angiotensine",
    r"\bAVC\b":      "accident vasculaire cérébral",
    r"\bSCA\b":      "syndrome coronarien aigu",
    r"\bIDM\b":      "infarctus du myocarde",
    r"\bEP\b":       "embolie pulmonaire",
    r"\bTVP\b":      "thrombose veineuse profonde",
    r"\bBPCO\b":     "bronchopneumopathie chronique obstructive",
    r"\bIRC\b":      "insuffisance rénale chronique",
    r"\bIRA\b":      "insuffisance rénale aiguë",
    r"\bICC\b":      "insuffisance cardiaque congestive",
}

#: Generic French abbreviations.
GENERAL_ABBREVIATIONS: Dict[str, str] = {
    r"\betc\.?\b":   "et cetera",
    r"\bcf\.?\b":    "confer",
    r"\bvs\.?\b":    "versus",
    r"\bex\.?\b":    "exemple",
    r"\bn°":         "numéro",
    r"\bp\.?\s?ex\.?": "par exemple",
    r"\bc\.?à\.?d\.?": "c'est à dire",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_abbreviations(
    text: str,
    extra: Dict[str, str] | None = None,
) -> str:
    """
    Replace known abbreviations with their spoken French equivalents.

    *extra* lets callers supply additional patterns beyond the built-in tables.
    """
    tables = [GENERAL_ABBREVIATIONS, MEDICAL_ABBREVIATIONS]
    if extra:
        tables.append(extra)
    for table in tables:
        for pattern, replacement in table.items():
            text = re.sub(pattern, replacement, text)
    return text


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_special_chars: bool = True,
    expand_abbrevs: bool = True,
    extra_abbreviations: Dict[str, str] | None = None,
) -> str:
    """
    Full normalisation pipeline for French (medical) text.

    Steps (in order):
      1. Expand abbreviations
      2. Unicode NFC normalisation
      3. Lowercase
      4. Remove non-speech characters
      5. Collapse whitespace
    """
    if expand_abbrevs:
        text = expand_abbreviations(text, extra=extra_abbreviations)

    # NFC: ensures accented letters are single code-points
    text = unicodedata.normalize("NFC", text)

    if lowercase:
        text = text.lower()

    if remove_special_chars:
        # Keep: ASCII letters, French accented letters, digits,
        #        spaces, hyphens, apostrophes, basic punctuation.
        text = re.sub(
            r"[^\w\s\-'\.,:;!?àâäéèêëîïôùûüçœæÀÂÄÉÈÊËÎÏÔÙÛÜÇŒÆ]",
            " ",
            text,
        )

    # Collapse runs of whitespace / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text
