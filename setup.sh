#!/bin/bash
set -e

echo "===================================================================="
echo "🌱 Pipeline-ul de Automatizare și Configurare Mediu AgriGuard AI Pro"
echo "===================================================================="

echo "1. Instalare dependențe Python din requirements.txt..."
pip install -r requirements.txt

echo "2. Creare structură de directoare obligatorie..."
mkdir -p data/raw data/processed models scripts demo_files

echo "3. Lansare script de generare și regionalizare date pedoclimatice (Moldova)..."
python scripts/simulate_moldova_data.py

echo "4. Descărcare și verificare dataset-uri externe prin API-ul Kaggle..."
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "Descărcare Crop Recommendation Dataset..."
    kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data/raw/
    unzip -o data/raw/crop-recommendation-dataset.zip -d data/raw/
    mv data/raw/Crop_recommendation.csv data/raw/Crop_recommendation.csv 2>/dev/null || true
    
    echo "Descărcare PlantVillage (Plant Disease)..."
    kaggle datasets download -d emmarex/plantdisease -p data/raw/
    mkdir -p data/raw/plantvillage
    unzip -o data/raw/plantdisease.zip -d data/raw/plantvillage/
    echo "✅ Toate datele au fost descărcate și structurate!"
else
    echo "⚠️ Notă: Nu s-a detectat cheia API Kaggle (~/.kaggle/kaggle.json)."
    echo "Pentru reproducerea antrenării, plasați manual 'Crop_recommendation.csv' în data/raw/"
    echo "și imaginile dezarhivate în data/raw/plantvillage/"
fi

echo "===================================================================="
echo "✅ AgriGuard AI este gata de rulare! Executați: streamlit run frontend/app.py"
echo "===================================================================="