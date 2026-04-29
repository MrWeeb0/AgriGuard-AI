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

---

## Analiză Tehnică și Justificare (Q&A Juriu)

### 1. Analiza Exploratorie a Datelor (EDA)
Înainte de antrenare, am realizat o etapă de EDA pentru ambele seturi de date:
* **Vision (PlantVillage):** Am analizat distribuția claselor pentru a asigura un antrenament echilibrat. Am observat că trăsăturile vizuale ale bolilor fungice (ex: petele concentrice) sunt constante, ceea ce a permis modelului să învețe rapid.
* **Tabular (Soil Data):** Am generat o matrice de corelație (Heatmap) pentru a identifica interdependența dintre nutrienți (N, P, K) și pH. Aceasta a confirmat că anumiți parametri sunt predictori critici pentru anumite culturi (ex: Azotul pentru orez).

### 2. De ce EfficientNet-B0 și doar 5 epoci?
Am ales **EfficientNet-B0** datorită tehnologiei de **Compound Scaling**, care scalează echilibrat adâncimea și rezoluția rețelei. Este de 10 ori mai eficient decât ResNet-50, oferind performanțe similare cu resurse mult mai mici (Edge AI).
* **5 Epoci:** Folosind **Transfer Learning** (greutăți pre-antrenate pe ImageNet), modelul a extras deja "primitivele" vizuale. Am monitorizat curba de Loss și am observat că după epoca 5, acuratețea pe validare stagna, indicând atingerea punctului optim înainte de **Overfitting**.

### 3. Metrici: Acuratețe vs F1-Score
Deși raportăm acuratețea de 99.52%, am monitorizat constant **Recall-ul** și **F1-Score**. În agricultură, un "False Negative" (boală ratată) este mai periculos decât un "False Positive". Modelul nostru a fost optimizat să minimizeze ratele de omisiune pe clasele critice de infecție.

### 4. De ce XGBoost pentru Sol?
Am comparat XGBoost cu algoritmi precum SVM, KNN sau RandomForest:
* **vs KNN/SVM:** Acestea sunt sensibile la zgomotul senzorilor și nu gestionează bine relațiile non-liniare complexe din datele tabulare fără un tuning excesiv.
* **XGBoost:** Este liderul industriei pentru date structurate. Utilizează **Gradient Boosting** pentru a corecta erorile arborilor precedenți, este imun la valori aberante (outliers) și are o latență de predicție aproape nulă.

### 5. Preprocesarea Datelor
* **Vision:** Redimensionare la 224x224, normalizare conform standardelor ImageNet și Augmentare (rotații/flip) pentru a simula fotografii făcute în teren sub diverse unghiuri.
* **Tabular:** Am aplicat **StandardScaler** pentru a aduce toți parametrii la aceeași scară (medie 0, deviație 1) și **Label Encoding** pentru transformarea categoriilor în valori numerice.

---

## Arhitectura Sistemului

AgriGuard-AI/
    frontend/
        app.py
    demo_files/
        test_rosie_sanatoasa.jpg
        test_cartof_early_blight.jpg
    models/
        vision_model_rtx_finetuned.pth
        xgboost_soil_model.pkl
        soil_scaler.pkl
        soil_label_encoder.pkl
    requirements.txt
    README.md

---

## Metodologie de Antrenament
Antrenamentul a fost realizat pe o unitate GPU NVIDIA RTX 3060:
- **Dataset Vision:** PlantVillage (15 clase)
- **Dataset Tabular:** Crop Recommendation Dataset (22 culturi)
- **Framework-uri:** PyTorch (Viziune), XGBoost (Tabular), Streamlit (Deployment)

---

**Dezvoltat pentru Competiția ONIA 2026.**