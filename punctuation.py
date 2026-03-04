"""
punctuation.py
==============
Gestion de la ponctuation pour la dictée médicale en français.

Deux mécanismes :
  1. Commandes verbales  → l'utilisateur dit "virgule", "point", etc.
  2. Restauration auto   → modèle NLP (deepmultilingualpunctuation)

Usage autonome :
  python punctuation.py "le patient présente une fièvre virgule
                         une toux point à la ligne diagnostic point virgule"
"""

import re
import sys


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMMANDES VERBALES
# Ordre important : les plus longues en premier pour éviter les collisions
# ══════════════════════════════════════════════════════════════════════════════

VERBAL_COMMANDS = [
    # Sauts de ligne
    (r"\bnouveau paragraphe\b",                         "\n\n"),
    (r"\bnouvelle ligne\b",                             "\n"),
    (r"\bà la ligne\b",                                 "\n"),
    (r"\bsaut de ligne\b",                              "\n"),

    # Ponctuation de fin — AVANT "point" seul pour éviter collision
    (r"\bpoints? d[' ]interrogation\b",                 "?"),
    (r"\bpoints? d[' ]exclamation\b",                   "!"),
    (r"\bpoints? virgule\b",                            ";"),
    (r"\bpoints? de suspension\b",                      "..."),

    # "deux points" / "de points" / "des points" → deux-points (:)
    # Whisper transcrit souvent "deux points" en "de points"
    (r"\b(?:deux|de|des)\s+points?\b",                  ":"),

    # "point" ou "points" seul → point final
    (r"\bpoints?\b",                                    "."),

    # Ponctuation interne
    (r"\bvirgule\b",                                    ","),
    (r"\btiret\b",                                      "-"),
    (r"\bslash\b",                                      "/"),

    # Parenthèses / guillemets
    (r"\bouvr(?:ez|ir) (?:la )?parenth[èe]se\b",       "("),
    (r"\bferm(?:ez|er) (?:la )?parenth[èe]se\b",       ")"),
    (r"\bparenth[èe]se (?:ouvrante|ouverte)\b",         "("),
    (r"\bparenth[èe]se fermante\b",                     ")"),
    (r"\bouvr(?:ez|ir) (?:les )?guillemets\b",          '"'),
    (r"\bferm(?:ez|er) (?:les )?guillemets\b",          '"'),
]


def apply_verbal_commands(text: str) -> str:
    """Remplace les commandes verbales par les symboles de ponctuation."""
    result = text.lower()

    for pattern, replacement in VERBAL_COMMANDS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Nettoyer les espaces autour de la ponctuation
    result = re.sub(r"\s+([.,;:?!])", r"\1", result)
    result = re.sub(r"([.,;:?!])(?=[^\s\n])", r"\1 ", result)
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r" \n", "\n", result)
    result = result.strip()

    result = _capitalize_sentences(result)
    return result


def _capitalize_sentences(text: str) -> str:
    """Met une majuscule après chaque fin de phrase."""
    text = text[:1].upper() + text[1:] if text else text

    def _upper_match(m):
        return m.group(0)[:-1] + m.group(0)[-1].upper()

    text = re.sub(r"[.?!]\s+[a-zàâäéèêëîïôùûüç]", _upper_match, text)
    text = re.sub(r"\n+[a-zàâäéèêëîïôùûüç]",        _upper_match, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 2. RESTAURATION AUTOMATIQUE (deepmultilingualpunctuation)
# ══════════════════════════════════════════════════════════════════════════════

def restore_punctuation_auto(text: str) -> str:
    """
    Utilise deepmultilingualpunctuation pour ajouter la ponctuation manquante.
    Installation : pip install deepmultilingualpunctuation
    Retourne le texte original si le modèle n'est pas disponible.
    """
    try:
        from deepmultilingualpunctuation import PunctuationModel
        model = PunctuationModel(model="kredor/punctuate-all")
        result = model.restore_punctuation(text)
        return _capitalize_sentences(result)
    except ImportError:
        return text
    except Exception as e:
        print(f"  [warn] Restauration auto échouée : {e}")
        return text


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE COMBINÉ
# ══════════════════════════════════════════════════════════════════════════════

def process(text: str, auto: bool = True) -> str:
    """
    Pipeline complet :
      1. Commandes verbales (priorité — toujours appliquées)
      2. Restauration auto  (optionnelle, si peu de ponctuation détectée)
    """
    text = apply_verbal_commands(text)

    if auto:
        punct_ratio = sum(1 for c in text if c in ".,;:?!") / max(len(text), 1)
        if punct_ratio < 0.02:
            text = restore_punctuation_auto(text)

    return text


# ══════════════════════════════════════════════════════════════════════════════
# TEST EN LIGNE DE COMMANDE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:])
    else:
        raw = (
            "le patient présente une fièvre virgule une toux et des céphalées point "
            "à la ligne diagnostic deux points pneumonie bactérienne point virgule "
            "traitement deux points amoxicilline 1g trois fois par jour point"
        )

    print("Texte brut :")
    print(raw)
    print()
    result = process(raw, auto=False)
    print("Après traitement :")
    print(result)
