"""
Script de test pour vérifier chaque étape du pipeline
"""
import os
import sys

def test_step_1():
    """Test de la conversion MP3 → WAV"""
    print("\n" + "="*50)
    print("TEST 1: Conversion MP3 → WAV")
    print("="*50)
    
    mp3_dir = "DATA prete"
    wav_dir = "data/wav"
    
    # Vérifier le répertoire MP3
    if not os.path.exists(mp3_dir):
        print(f"❌ Le répertoire {mp3_dir} n'existe pas")
        return False
    
    # Compter les fichiers MP3
    mp3_files = [f for f in os.listdir(mp3_dir) if f.endswith(".mp3")]
    print(f"✅ Trouvé {len(mp3_files)} fichiers MP3")
    
    # Vérifier le répertoire WAV
    if os.path.exists(wav_dir):
        wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
        print(f"✅ Trouvé {len(wav_files)} fichiers WAV")
        
        if len(wav_files) > 0:
            print(f"✅ Conversion OK: {len(wav_files)}/{len(mp3_files)} fichiers")
            return True
        else:
            print("⚠️  Aucun fichier WAV trouvé. Exécutez 01_mp3_to_wav.py")
            return False
    else:
        print("⚠️  Le répertoire data/wav/ n'existe pas. Exécutez 01_mp3_to_wav.py")
        return False

def test_step_2():
    """Test de la création du CSV"""
    print("\n" + "="*50)
    print("TEST 2: Création du CSV")
    print("="*50)
    
    rtf_path = "expression médicale.rtf"
    csv_path = "data/dataset.csv"
    
    # Vérifier le fichier RTF
    if not os.path.exists(rtf_path):
        print(f"❌ Le fichier {rtf_path} n'existe pas")
        return False
    print(f"✅ Fichier RTF trouvé: {rtf_path}")
    
    # Vérifier le fichier CSV
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"✅ Fichier CSV trouvé avec {len(lines)-1} entrées (sans l'en-tête)")
            
            # Vérifier l'en-tête
            if len(lines) > 0:
                header = lines[0].strip()
                if "id" in header and "audio_path" in header and "transcription" in header:
                    print("✅ En-tête CSV correct")
                    return True
                else:
                    print("❌ En-tête CSV incorrect")
                    return False
        return True
    else:
        print("⚠️  Le fichier data/dataset.csv n'existe pas. Exécutez 02_rtf_to_csv.py")
        return False

def test_step_3():
    """Test de la préparation du dataset"""
    print("\n" + "="*50)
    print("TEST 3: Préparation du dataset")
    print("="*50)
    
    dataset_path = "data/whisper_dataset"
    
    if os.path.exists(dataset_path):
        # Vérifier la structure
        required_dirs = ["train", "validation", "test"]
        found_dirs = []
        
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path) and item in required_dirs:
                found_dirs.append(item)
        
        print(f"✅ Dataset trouvé: {dataset_path}")
        print(f"✅ Splits trouvés: {', '.join(found_dirs)}")
        
        if len(found_dirs) == len(required_dirs):
            print("✅ Structure du dataset correcte")
            return True
        else:
            missing = set(required_dirs) - set(found_dirs)
            print(f"⚠️  Splits manquants: {', '.join(missing)}")
            return False
    else:
        print("⚠️  Le dataset n'existe pas. Exécutez 03_prepare_dataset.py")
        return False

def test_step_4():
    """Test de l'entraînement du modèle"""
    print("\n" + "="*50)
    print("TEST 4: Modèle entraîné")
    print("="*50)
    
    model_path = "model/whisper-medical-fr"
    
    if os.path.exists(model_path):
        required_files = ["config.json", "pytorch_model.bin"]
        found_files = []
        
        for file in os.listdir(model_path):
            if file in required_files or file.startswith("pytorch_model"):
                found_files.append(file)
        
        print(f"✅ Modèle trouvé: {model_path}")
        print(f"✅ Fichiers trouvés: {len(found_files)}")
        
        if len(found_files) > 0:
            print("✅ Modèle semble être sauvegardé correctement")
            return True
        else:
            print("⚠️  Fichiers du modèle manquants")
            return False
    else:
        print("⚠️  Le modèle n'existe pas. Exécutez 04_train_whisper.py")
        return False

def test_step_5():
    """Test de la transcription"""
    print("\n" + "="*50)
    print("TEST 5: Test de transcription")
    print("="*50)
    
    test_audio = "data/wav/audio_001.wav"
    
    if not os.path.exists(test_audio):
        print(f"⚠️  Fichier de test {test_audio} n'existe pas")
        return False
    
    print(f"✅ Fichier audio de test trouvé: {test_audio}")
    print("💡 Pour tester la transcription, exécutez: python 05_test_model.py")
    return True

def main():
    """Exécute tous les tests"""
    print("\n" + "="*50)
    print("PIPELINE DE TEST - Whisper Fine-tuning")
    print("="*50)
    
    results = []
    
    # Tests
    results.append(("Étape 1: MP3 → WAV", test_step_1()))
    results.append(("Étape 2: RTF → CSV", test_step_2()))
    results.append(("Étape 3: Préparation dataset", test_step_3()))
    results.append(("Étape 4: Modèle entraîné", test_step_4()))
    results.append(("Étape 5: Test transcription", test_step_5()))
    
    # Résumé
    print("\n" + "="*50)
    print("RÉSUMÉ DES TESTS")
    print("="*50)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nRésultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés!")
    else:
        print(f"\n⚠️  {total - passed} étape(s) nécessite(nt) d'être exécutée(s)")
        print("\nOrdre d'exécution recommandé:")
        for i, (name, result) in enumerate(results, 1):
            if not result:
                print(f"  {i}. Exécutez le script correspondant à: {name}")

if __name__ == "__main__":
    main()
