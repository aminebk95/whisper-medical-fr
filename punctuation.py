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

    # "deux points" / "de points" / "des points" / "de pointe" → deux-points (:)
    # Whisper transcrit souvent "deux points" en "de points" ou "de pointe"
    (r"\b(?:deux|de|des)\s+pointe?s?\b",                ":"),

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

    # Supprimer la ponctuation dupliquée (Whisper + commande verbale)
    result = re.sub(r",\s*,+", ",", result)        # ,, ou , , → ,
    result = re.sub(r"\.\s*\.+", ".", result)      # .. ou . . → .
    result = re.sub(r";\s*;+", ";", result)        # ;; → ;
    result = re.sub(r":\s*:+", ":", result)        # :: → :
    result = re.sub(r",\s*\.", ".", result)         # ,. → .
    result = re.sub(r"\.\s*,", ".", result)         # ., → .

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
# 3. CORRECTION ORTHOGRAPHIQUE — LanguageTool
# ══════════════════════════════════════════════════════════════════════════════

# Instance partagée pour éviter de redémarrer le serveur à chaque appel
_lt_tool = None

# Règles à désactiver pour le français médical
# (termes techniques souvent inconnus du correcteur standard)
DISABLED_RULES = [
    "FRENCH_WHITESPACE",          # espaces insécables : géré manuellement
    "UPPERCASE_SENTENCE_START",   # on gère les majuscules nous-mêmes
    "COMMA_PARENTHESIS_WHITESPACE",
]


def _get_lt_tool():
    """Charge LanguageTool une seule fois (serveur Java local)."""
    global _lt_tool
    if _lt_tool is None:
        import language_tool_python
        _lt_tool = language_tool_python.LanguageTool(
            "fr",
            config={"disabledRuleIds": ",".join(DISABLED_RULES)},
        )
    return _lt_tool


def correct_orthography(text: str) -> str:
    """
    Corrige l'orthographe et la grammaire via LanguageTool (serveur local).
    Installation : pip install language_tool_python
                   (télécharge automatiquement LanguageTool ~200MB la 1ère fois)
    Retourne le texte original si LanguageTool n'est pas disponible.
    """
    try:
        tool = _get_lt_tool()
        corrected = tool.correct(text)
        return corrected
    except ImportError:
        print("  [info] LanguageTool non installé → pip install language_tool_python")
        return text
    except Exception as e:
        print(f"  [warn] LanguageTool échoué : {e}")
        return text


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE COMBINÉ
# ══════════════════════════════════════════════════════════════════════════════

def process(text: str, auto_punct: bool = True, spellcheck: bool = True) -> str:
    """
    Pipeline complet :
      1. Commandes verbales   (toujours appliquées)
      2. Restauration auto    (si peu de ponctuation détectée)
      3. Correction LanguageTool (orthographe + grammaire)
    """
    # Étape 1 — commandes verbales
    text = apply_verbal_commands(text)

    # Étape 2 — ponctuation automatique si texte peu ponctué
    if auto_punct:
        punct_ratio = sum(1 for c in text if c in ".,;:?!") / max(len(text), 1)
        if punct_ratio < 0.02:
            text = restore_punctuation_auto(text)

    # Étape 3 — correction orthographique LanguageTool
    if spellcheck:
        text = correct_orthography(text)

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
    result = process(raw, auto_punct=False, spellcheck=False)
    print("Après traitement :")
    print(result)
