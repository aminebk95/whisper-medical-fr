import csv
import os
import re
from striprtf.striprtf import rtf_to_text

RTF_PATH = "expression médicale.rtf"
AUDIO_DIR = "data/wav"
CSV_PATH = "data/dataset.csv"

if not os.path.exists(RTF_PATH):
    raise FileNotFoundError(f"Le fichier {RTF_PATH} n'existe pas")

os.makedirs(os.path.dirname(CSV_PATH) if os.path.dirname(CSV_PATH) else ".", exist_ok=True)

def clean_sentence(line):
    line = line.strip()
    line = re.sub(r"^[\-\•\*]\s*", "", line)  # supprimer tirets
    return line

try:
    with open(RTF_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = rtf_to_text(f.read())
except Exception as e:
    raise IOError(f"Erreur lors de la lecture du fichier RTF: {e}")

sentences = []
for line in text.split("\n"):
    line = clean_sentence(line)
    if len(line) > 5:
        sentences.append(line)

try:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "audio_path", "transcription"])

        for i, sentence in enumerate(sentences, start=1):
            audio_name = f"audio_{i:03}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_name)

            writer.writerow([i, audio_path, sentence])
except Exception as e:
    raise IOError(f"Erreur lors de l'écriture du fichier CSV: {e}")

print(f"✅ CSV créé avec {len(sentences)} phrases")
