# AgriGuard AI — Asistent Inteligent pentru Fermieri

**AgriGuard AI** este o platformă digitală avansată care combină **Computer Vision** și **Analiza de Date Tabulare** pentru a oferi fermierilor un diagnostic precis al bolilor plantelor și recomandări inteligente pentru optimizarea culturilor.

Sistemul rezolvă problema "Cutiei Negre" (Black Box) în AI prin implementarea tehnologiei **XAI (Explainable AI)**, arătând utilizatorului exact ce zone ale frunzei au declanșat alerta de boală.

---

## Echipa si Institutia
Proiectul a fost dezvoltat în cadrul instituției de învățământ, respectând criteriile de colaborare și mentorat impuse de regulament.

**Autorii Proiectului:**
* Elev: Nicolai Sîrețanu
* Elev: Alexandru Caldare

**Mentorat și Apartenență:**
* Instituție: Liceul Teoretic "Ion Pelivan", Răzeni
* Mentor: Nadejda Sinițîn 

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

## Justificarea Alegerii Modelelor Tehnice

Pentru a construi un sistem pregătit pentru mediul agricol real (Production-Ready), arhitectura a fost selectată pe baza eficienței și robusteței, nu doar a acurateței în condiții de laborator:

* **De ce EfficientNet-B0 pentru Viziune? (Raport Acuratețe/Resurse)**
  Agricultura se desfășoară adesea în zone cu semnal slab la internet. Am ales varianta B0 deoarece este extrem de ușoară (~20MB) și pregătită pentru "Edge-Computing" (dispozitive mobile), spre deosebire de modelele masive (ResNet/VGG). Datorită tehnicii de *Compound Scaling*, rețeaua scalează matematic adâncimea și rezoluția, reușind să capteze micro-texturi esențiale (ex. sporii de mucegai) și integrându-se perfect cu motorul XAI (Grad-CAM).

* **De ce XGBoost pentru Sol? (Robustețe la Zgomot)**
  Deep Learning-ul este ineficient pe date tabulare (structurate). Pentru parametrii de sol și climă (N, P, K, pH), am folosit **XGBoost** — standardul absolut în industrie pentru astfel de date. Senzorii agricoli din teren pot da erori de citire; XGBoost gestionează nativ zgomotul senzorilor (sensor noise), relațiile non-liniare și previne prăbușirea sistemului în cazul unor date extreme. În plus, are o latență de execuție de doar câteva milisecunde.

* **De ce un Sistem Hibrid Decuplat? (Human-in-the-Loop)**
  Dacă fermierul încarcă o imagine neclară sau complet compromisă, sistemul nu devine inutil. Datorită arhitecturii decuplate, ramura de mediu va continua să ofere recomandări valoroase de cultivare. AI-ul nostru colaborează cu omul: diagnosticul vizual se confirmă încrucișat cu profilul solului, imitând exact procesul cognitiv al unui inginer agronom.

---

## Kit de Testare pentru Juriu (Demo Rapid)

Pentru a facilita procesul de evaluare, am pregătit un set de fișiere de test în directorul `demo_files/`. Vă recomandăm să rulați următoarele scenarii:

**Scenariul 1: Detecția unei infecții fungice**
* **Imagine:** Încărcați fișierul `demo_files/test_2_cartof_early_blight.jpg`
* **Parametri Sol (Sidebar):** Umiditate: 85%, Precipitații: 200 mm, Temperatură: 22 °C
* **Rezultat:** Sistemul va detecta boala cu încredere maximă, Grad-CAM va evidenția cu roșu petele necrotice, iar sistemul tabular va sugera cultura optimă pentru un microclimat umed.

**Scenariul 2: Confirmarea stării de sănătate**
* **Imagine:** Încărcați fișierul `demo_files/test_1_rosie_sanatoasa.jpg`
* **Parametri Sol (Sidebar):** Azot (N): 40, Fosfor (P): 50, pH Sol: 6.5
* **Rezultat:** Sistemul va confirma starea de sănătate. Harta XAI va avea o distribuție uniformă, fără focare roșii.

---

## Structura Proiectului

AgriGuard-AI/
├── frontend/
│   └── app.py
├── demo_files/
│   └── (Imagini de test pentru juriu)
├── models/
│   ├── vision_model_rtx_finetuned.pth
│   ├── xgboost_soil_model.pkl
│   ├── soil_scaler.pkl
│   └── soil_label_encoder.pkl
├── requirements.txt
└── README.md

## Instalare și Rulare Locală
### Clonați depozitul:

git clone [https://github.com/utilizator/AgriGuard-AI.git](https://github.com/utilizator/AgriGuard-AI.git)
cd AgriGuard-AI

### Instalați dependențele:

pip install -r requirements.txt

### Rulați aplicația:

streamlit run frontend/app.py

## Metodologie și Antrenament
Modelul de viziune a fost antrenat pe setul de date PlantVillage, trecând printr-un proces de Fine-Tuning în 5 epoci pe o unitate GPU NVIDIA RTX 3060.

Epoca 1: 82.98% acuratețe

Epoca 5: 99.52% acuratețe

Sistemul de explicabilitate Grad-CAM extrage gradienții din ultimul strat convoluțional (features[-1]) pentru a vizualiza pixelii critici în procesul de luare a deciziei.

Dezvoltat pentru Competiția ONIA 2026.