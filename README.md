# AgriGuard AI — Asistent Inteligent pentru Fermieri

**AgriGuard AI** este o platformă digitală avansată care combină **Computer Vision** și **Analiza de Date Tabulare** pentru a oferi fermierilor un diagnostic precis al bolilor plantelor și recomandări inteligente pentru optimizarea culturilor.

Sistemul rezolvă problema "Cutiei Negre" (Black Box) în AI prin implementarea tehnologiei **XAI (Explainable AI)**, arătând utilizatorului exact ce zone ale frunzei au declanșat alerta de boală.

---

## Echipa si Institutia
Proiectul a fost dezvoltat în cadrul instituției de învățământ, respectând criteriile de colaborare și mentorat impuse de regulament.

Autorii Proiectului
Elev: Nicolai Sîrețanu
      Alexandru Caldare

Mentorat și Apartenență
Instituție: Liceul Teoretic Ion Pelivan, Răzeni

Mentor: Nadejda Sinițîn 

---

## Caracteristici Principale

- **Diagnostic Vizual (99.52% Acuratețe):** Identifică 15 tipuri de boli și stări de sănătate ale plantelor folosind arhitectura **EfficientNet-B0**.
- **Analiză Hibridă de Sol (98% Acuratețe):** Folosește un model **XGBoost** pentru a analiza parametrii pedologici (Azot, Fosfor, Potasiu, pH) și condițiile climatice.
- **Explicabilitate Neurală (Grad-CAM):** Generează hărți termice în timp real care evidențiază focarele de infecție direct pe imaginea încărcată.
- **Interfață Dashboard Modernă:** Dezvoltată în **Streamlit**, optimizată pentru mobil și desktop, oferind un flux de utilizare intuitiv pentru fermieri.
- **Cloud-Native:** Pregătit pentru deployment instantaneu pe Streamlit Community Cloud.

---

## Arhitectura Tehnică

Proiectul este construit pe o infrastructură hibridă de Machine Learning:

1. **Ramura de Viziune (Deep Learning):**
   - Framework: PyTorch
   - Model: EfficientNet-B0 (Transfer Learning + Fine-Tuning)
   - Tehnologie XAI: Grad-CAM (Gradient-weighted Class Activation Mapping)

2. **Ramura de Mediu (Tabular Data):**
   - Algoritm: XGBoost Classifier
   - Preprocesare: Scikit-Learn (StandardScaler & LabelEncoder)

3. **Frontend:**
   - Framework: Streamlit
   - Procesare Imagine: OpenCV & Pillow

---

## Structura Proiectului

```text
AgriGuard-AI/
├── frontend/
│   └── app.py
├── models/
│   ├── vision_model_rtx_finetuned.pth
│   ├── xgboost_soil_model.pkl
│   ├── soil_scaler.pkl
│   └── soil_label_encoder.pkl
├── requirements.txt
└── README.md