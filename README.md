# AgriGuard AI — Asistent Inteligent pentru Fermieri

**AgriGuard AI** este o platformă digitală avansată care combină **Computer Vision** și **Analiza de Date Tabulare** pentru a oferi fermierilor un diagnostic precis al bolilor plantelor și recomandări inteligente pentru optimizarea culturilor. 

Sistemul rezolvă problema opacității algoritmice ("Black Box") în inteligența artificială prin implementarea tehnologiei **XAI (Explainable AI)**, indicându-le utilizatorilor prin hărți termice precise zonele frunzei care au declanșat alerta de boală.

---

## Echipa și Instituția

Proiectul a fost dezvoltat în cadrul instituției de învățământ, respectând riguros criteriile de colaborare, originalitate și mentorat impuse de regulamentul competiției.

* **Autori:** * Nicolai Sîrețanu (Elev)
  * Alexandru Caldare (Elev)
* **Mentorat și Apartenență:**
  * Instituție: Liceul Teoretic "Ion Pelivan", Răzeni
  * Mentor: Nadejda Sinițîn

---

## Caracteristici Principale și Performanțe Obținute

* **Diagnostic Vizual (99.54% Acuratețe de Validare):** Identifică cu precizie chirurgicală 15 tipuri de boli și stări de sănătate ale frunzelor folosind o rețea **EfficientNet-B0** optimizată prin Transfer Learning adaptiv.
* **Analiză Pedoclimatică Hibridă (98.85% Acuratețe de Test):** Folosește un clasificator avansat **XGBoost** pentru a analiza parametrii chimici ai solului (Azot, Fosfor, Potasiu, pH) și condițiile de mediu (temperatură, umiditate, precipitații) pentru a recomanda cultura optimă dintr-un spectru de 22 de opțiuni.
* **Explicabilitate Neurală (Grad-CAM):** Generează hărți termice în timp real suprapuse peste imaginile încărcate, oferind transparență totală asupra deciziilor luate de rețeaua neurală convoluțională.
* **Arhitectură Decuplată Robustă:** Cele două servicii de Machine Learning (pentru date nestructurate și structurate) sunt integrate nativ într-o aplicație web intuitivă dezvoltată în Streamlit.

---

## Analiză Tehnică Detaliată (Ghid de Aliniere la Criteriile ML)

### 1. Analiza Exploratorie a Datelor (EDA) profunda
Pentru a asigura calitatea datelor înainte de antrenare, am implementat un pipeline complet de EDA în notebook-urile dedicate:
* **Date Structurate (Sol):** Am generat o matrice de corelație (Heatmap) pentru a evalua dependențele liniare dintre macro-nutrienți și pH, am analizat echilibrul perfect al claselor (20 de mostre stratificate per cultură în setul de test) și am rulat distribuții de tip Boxplot pentru identificarea și izolarea valorilor aberante (*outliers*).
* **Date Nestructurate (Vision):** Am inspectat distribuția volumului de imagini per clasă în datasetul *PlantVillage*, extrăgând rezoluțiile native și analizând intensitatea medie a pixelilor pentru a calibra corect transformările de normalizare ImageNet.

### 2. Justificarea Arhitecturii și Prevenirea Data Leakage
* **EfficientNet-B0 (Transfer Learning):** S-a ales această arhitectură datorită tehnologiei sale de *Compound Scaling*, oferind o eficiență computațională de 10 ori mai mare decât modele tradiționale precum ResNet-50, ideală pentru execuția pe servere cloud cu resurse limitate sau dispozitive mobile (*Edge AI*).
* **Strategia de Antrenament în Două Etape:** Modelul a fost antrenat inițial cu clasificatorul final deblocat (faza *Frozen*, 5 epoci, `lr=0.001`), urmat de o dezghețare totală a parametrilor pentru ajustări fine (faza *Fine-Tuning*, 5 epoci, `lr=0.0001`).
* **Securizarea Split-ului:** Pentru a elimina eroarea gravă de *Data Leakage* (scurgerea de date între fazele de antrenare), am blocat generatorul matematic `random_split` din PyTorch cu un seed global fix (`seed=42`). Astfel, setul de validare a rămas complet independent și virgin pe parcursul ambelor stadii de optimizare.

### 3. Evaluare Multi-Metrică Riguroasă (Validare Independentă)
Sistemul nu raportează doar acuratețea simplă, ci a fost evaluat folosind metrici statistice complete pe seturi de date pe care modelele nu le-au văzut niciodată în timpul instruirii:
* **Raportul de Viziune (4,128 de imagini de test):** Modelul a obținut un scor general Macro Avg F1-Score de **0.99**. Recall-ul stabil de 1.00 pe clase critice patogene (ex. *Potato Early Blight*) demonstrează că am minimizat la maximum riscul unui diagnostic fals-negativ (omiterea unei boli).
* **Raportul Tabular (440 de mostre stratificate):** Clasificatorul XGBoost a înregistrat un F1-Score ponderat de **0.99**. Performanța sa a fost validată în raport cu un model baseline de tip **Random Forest** (care a atins ~97%), confirmând superioritatea algoritmului bazat pe *Gradient Boosting* pe date tabulare complexe.

### 4. Testarea la Limită și Reziliența (Stress Testing)
Pentru a simula comportamentul aplicației pe teren, am supus modelul de viziune la teste de stres sintetice:
* **Condiții moderate de câmp:** Adăugând rotații, variații de luminozitate și blur de autofocus (specifice camerei unui smartphone), modelul a reținut o acuratețe remarcabilă de **99.40%**.
* **Condiții distructive extreme:** În cazul unor distorsiuni extreme de culoare și zgomot optic masiv, acuratețea vizuală s-a degradat controlat până la 7.40%. Acest comportament a validat logic arhitectura noastră hibridă: în condiții de vizibilitate eșuată sau senzori optici defecți, ramura pedoclimatică XGBoost preia rolul principal pentru a ghida fermierul fără întreruperi.

---

## Discuție Etică și Impact Social

Implementarea AgriGuard AI a fost ghidată în totalitate de principii solide de responsabilitate tehnologică:

1. **Atenuarea Biasului de Dataset:** Recunoaștem că setul de date *PlantVillage* reflectă imagini de laborator din regiuni specifice. Soluția noastră XAI (Grad-CAM) funcționează ca un filtru etic de siguranță: fermierul poate valida instant dacă AI-ul ia decizia pe baza leziunii celulare reale sau dacă este indus în eroare de reflexia luminii sau fundal.
2. **Prevenirea Riscurilor de Tratament:** Un diagnostic greșit poate îndruma un fermier spre aplicarea inutilă și toxică de pesticide chimice sau, din contra, spre ignorarea unei infecții severe. Aplicația noastră include avertismente clare prin care specifică faptul că platforma reprezintă un instrument de alertă timpurie și screening, încurajând decizia umană finală.
3. **Sustenabilitate și Amprenta de Mediu:** Prin optimizarea culturilor recomandate în funcție de nutrienți, AgriGuard AI combate fenomenul de supra-fertilizare cu Azot (N), protejând pânza freatică și reducând emisiile de gaze cu efect de seră din agricultură.
4. **Confidențialitate Totală:** Aplicația rulează inferența direct în memorie, necolectând și nestocând imaginile utilizatorilor sau coordonatele GPS ale terenurilor, asigurând protecția secretului comercial al exploatațiilor agricole.

---

## Structura Proiectului

```text
AgriGuard-AI/
│
├── data/
│   └── raw/
│       ├── Crop_recommendation.csv
│       └── plantvillage/                  # Structurat pe cele 15 clase de boli
│
├── frontend/
│   └── app.py                             # Interfața Streamlit în limba română
│
├── models/
│   ├── vision_model_rtx_finetuned.pth     # Modelul EfficientNet-B0 (99.54%)
│   ├── xgboost_soil_model.pkl             # Modelul XGBoost Tabular (98.85%)
│   ├── soil_scaler.pkl                    # Scalatorul StandardScaler
│   └── soil_label_encoder.pkl             # LabelEncoder-ul pentru culturi
│
├── notebooks/
│   ├── 01_eda_soil_data.ipynb             # EDA complet pe parametrii de sol
│   ├── 02_eda_vision_data.ipynb           # Analiză imagini și augmentări
│   ├── 03_train_vision_model.ipynb        # Antrenare robustă viziune
│   └── 04_train_tabular_model.ipynb       # Antrenare și comparare model sol
│
├── requirements.txt                       # Dependențe configurate pentru Cloud
└── README.md                              # Raportul tehnic oficial al proiectului 

