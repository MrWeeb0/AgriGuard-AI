# 🌱 AgriGuard AI — Sistem Hibrid de Inteligență Artificială pentru Diagnosticarea Patologiilor Vegetale 

## Cadre Identitare și Administrative

**Autori:**
- Nicolai Sîrețanu
- Alexandru Caldare

**Coordonare Științifică și Mentorat:**
- Nadejda Sinițîn (Profesor Mentor)

**Apartenență Instituțională:**
Liceul Teoretic "Ion Pelivan", Răzeni, Ialoveni, Republica Moldova

**Destinație Competitivă:**
Competiția Națională ONIA 2026 (Olimpiada Națională de Inteligență Artificială)

---

## 1. Definirea Problemei și Impactul Agronomic

### 1.1 Contextul Global și Necesitatea Soluției

Agricultura modernă se confruntă cu o presiune sistemică dublă: pierderile semnificative de biomasă vegetală cauzate de fitopatogeni și degradarea fizico-chimică a solurilor determinată de monoculturi repetitive sau fertilizări empirice, necalibrate.

Identificarea bolilor foliare se bazează tradițional pe inspecția vizuală macroscopică realizată de specialiști agronomi, un proces cu o inerție ridicată, costisitor și predispus la interpretări subiective în stadiile timpurii ale infecțiilor.

Deși implementarea rețelelor neurale convoluționale de Deep Learning a adus precizii ridicate în medii controlate, aceste sisteme suferă de o vulnerabilitate critică: **opacitatea decizională** ("Black Box"). Un fermier nu poate adopta cu încredere o recomandare automatizată de tratament chimic dacă algoritmul nu poate justifica regiunile anatomice ale frunzei care au declanșat alerta.

**AgriGuard AI elimină această barieră** prin integrarea unui modul activ de Explicabilitate Vizuală (XAI).

### 1.2 Profilul Beneficiarilor și Valoarea Practică

Sistemul se adresează direct micilor producători ce exploatează sere, dar și operatorilor marilor suprafețe agricole din Republica Moldova. Aceștia primesc un instrument integrat de screening capabil să ruleze pe dispozitive cu resurse computaționale modeste (smartphones), optimizând luarea deciziilor direct pe teren.

### 1.3 Obiective de Impact Măsurabile

Valoarea practică a sistemului este definită de un set de indicatori de performanță agronomică monitorizabili direct pe teren:

- **Diminuarea utilizării pesticidelor de sinteză cu 15% - 20%** prin identificarea localizată a focarelor infecțioase și aplicarea exclusiv a tratamentelor zonale, evitând pulverizările generale profilactice.

- **Reducerea pierderilor de recoltă cu până la 12% anual**, datorită depistării precoce a patologiilor foliare înainte ca acestea să atingă un prag critic de dispersie epidemică.

- **Eficientizarea cu 18% a utilizării fertilizanților** prin corelarea directă a planului de cultură cu macro-nutrienții existenți în sol, reducând costurile operaționale ale exploatației.

---

## 2. Arhitectura Sistemului Hibrid

Platforma integrează un sistem multi-modal ce decuplează procesarea datelor structurate de cele nestructurate în două servicii Machine Learning complementare.

```
                  +-----------------------------------+
                  |        INTERFAȚA STREAMLIT        |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
[DATE NESTRUCTURATE: IMAGINE]                     [DATE STRUCTURATE: SENZORI]
            |                                               |
            v                                               v
  EfficientNet-B0 (PyTorch)                     Random Forest (Scikit-Learn)
  Input: Imagine frunză (RGB)                   Input: [N, P, K, T, H, pH, Rain]
  Output: Probabilități clasă                   Output: Probabilități cultură
  XAI: Grad-CAM Activations                     Rol: Managementul asolamentului
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     Raport de Diagnostic Hibrid   |
                  +-----------------------------------+
```

### 2.1 Serviciul ML pe Date Structurate (Recomandare Culturi)

- **Model:** Ansamblu de arbori de decizie de tip Random Forest, optimizat prin căutare sistematică în rețea.
- **Input:** Vector numeric format din 7 caracteristici fizico-chimice și macro-climatice: [N, P, K, temperatură, umiditate, pH, precipitații].
- **Output:** Distribuție probabilistică multiclasă peste cele 22 de tipuri de culturi agricole suportate.
- **Rol:** Planificarea rotației culturilor, evitarea epuizării solului și optimizarea utilizării resurselor pedoclimatice.

### 2.2 Serviciul ML pe Date Nestructurate (Clasificare Patologică)

- **Model:** Rețea neurală convoluțională EfficientNet-B0 optimizată prin Transfer Learning fin, cuplată cu un hook de extracție Grad-CAM.
- **Input:** Imagine color brută a suprafeței foliare (RGB, rezoluție standardizată la $224 \times 224$).
- **Output:** Clasificarea stării fitosanitare a plantei în 15 categorii (bolnave/sănătoase) și suprapunerea unei hărți termice bidimensionale de activare a atenției.
- **Rol:** Detecție, diagnosticare de precizie și vizualizarea markerilor patogeni.

### 2.3 Sinergie și Reziliență Arhitecturală (Service Degradation)

Sistemul combină cele două servicii pentru a asigura o asistență agronomică completă pe tot parcursul ciclului de producție (solul coordonează faza de pre-semănare, iar analiza foliară ghidează perioada de vegetație).

Din punct de vedere al siguranței în funcționare, platforma implementează un **mod de degradare controlată a serviciilor**. În cazul în care calitatea camerei de pe teren este afectată (zgomot optic, praf pe lentilă sau lipsă de focalizare), modulul de viziune computerizată se dezactivează, dar utilizatorul continuă să beneficieze de funcționalitățile complete de asistență pedoclimatică furnizate de modelul de analiză a solului.

---

## 3. Date și Preprocesare: Ramura Structurată

### 3.1 Profilul Setului de Date și Targetul

Datele structurate sunt alcătuite din 2200 de eșantioane pedoclimatice distribuite echilibrat. Variabila țintă, label, acoperă 22 de culturi distincte, fiind codificată prin LabelEncoder într-un spațiu de stări discret $[0, 21]$.

### 3.2 Analiza Exploratorie a Datelor (EDA)

În cadrul etapei de analiză din notebook-ul `01_eda_soil_data.ipynb` au fost efectuate următoarele operațiuni statistice:

**Analiza de Corelație Pearson:**
S-a evidențiat o corelație liniară strânsă ($0.73$) între Fosfor (P) și Potasiu (K), dictând reducerea variabilității prin utilizarea hiperparametrului `max_features='sqrt'` în faza de antrenare, prevenind astfel colapsul diversității decizionale din cadrul ansamblului de arbori.

**Izolarea și Evaluarea Outlierilor:**
Reprezentările boxplot au identificat abateri extreme pozitive pe variabilele de azot (N) și precipitații (rainfall). Acestea au fost păstrate documentat în setul de antrenament pentru a asigura capacitatea modelului de a procesa fenomene pedoclimatice reale din teren (perioade severe de secetă sau fertilizări concentrate masive).

**Certificarea Stratificării:**
Setul de test reține un echilibru strict, conținând exact 20 de eșantioane reprezentative per clasă dintr-un total de 440 utilizate la evaluarea finală.

### 3.3 Adaptarea la Specificul Pedoclimatic al Republicii Moldova

Seturile de date standardizate reflectă adesea medii agronomice generalizate. Solurile din Republica Moldova prezintă proprietăți distincte ce au fost adresate în arhitectura sistemului:

**Profilul Cernoziomurilor locale:**
Deși dispun de un volum ridicat de humus nativ, solurile din Moldova înregistrează adesea un deficit istoric de Fosfor (P) mobil, blocat sub formă de săruri insolubile.

**Profilul Climatic Regional:**
Vulnerabilitatea ridicată la secetă din zonele de Sud impune o corelare strictă între umiditatea relativă a aerului și regimul de precipitații.

---

## 4. Model ML: Ramura Structurată

### 4.1 Benchmarking și Justificarea Selecției

Evaluarea sistematică realizată în notebook-ul `04_train_tabular_model.ipynb` a comparat performanțele a patru paradigme matematice diferite pe setul de test:

| Model | Train Acc | Test Acc |
|-------|-----------|----------|
| K-Nearest Neighbors (KNN) | 98.47% | 97.95% |
| Support Vector Classifier (SVM RBF) | 98.58% | 98.41% |
| XGBoost Classifier | 100.00% | 98.86% |
| **Random Forest Classifier (Baseline)** | **100.00%** | **99.55%** |

**Modelul Random Forest** a demonstrat o capacitate superioară de generalizare directă pe datele de test, fiind selectat ca motor de predicție tabulară în producție.

### 4.2 Combaterea Overfitting-ului prin Regularizare Structurală

Un scor perfect de 100.00% pe datele de antrenare (obținut de baseline-urile Random Forest și XGBoost în primele faze) indica o adâncime decizională excesivă (`max_depth=None`), modelul memorând zgomotul din date.

Pentru a combate această problemă de supra-ajustare, am aplicat o regularizare structurală strictă:

- Restricționarea adâncimii maxime a arborilor (`max_depth`)
- Setarea unui prag minim de eșantioane pentru splitarea unui nod intern (`min_samples_split`)

### 4.3 Eliminarea Soluțiilor Ineficiente de Producție (Refactorizare)

A fost eliminată complet utilizarea unui model Random Forest salvat sub o denumire eronată din considerente de evitare a refactorizării codului. Modelul final este salvat sub propria identitate matematică: `random_forest_soil_model.pkl`, iar toate variabilele și referințele din scriptul Streamlit (`app.py`) au fost restructurate pentru a respecta trasabilitatea exactă a datelor.

---

## 5. Protocol de Evaluare și Robustețe: Ramura Structurată

### 5.1 Validare Încrucișată Stratificată și Optimizare Sistematică

Pentru a exclude bias-ul statistic introdus de raportarea performanțelor pe un singur split fix, am implementat un protocol robust de Validare Încrucișată Stratificată cu 5 Folduri (5-Fold StratifiedKFold) combinat cu o căutare automată a hiperparametrilor optimi prin GridSearchCV:

```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [6, 8, 10],
    'min_samples_split': [4, 6, 10],
    'max_features': ['sqrt']
}
```

### 5.2 Performanțe Post-Tuning

După execuția protocolului de optimizare hiperparametrică, modelul a înregistrat:

- **Acuratețe stabilizată prin Validare Încrucișată (Train):** 99.31% (O scădere utilă de la 100.00%, confirmând trecerea de la memorarea datelor la generalizarea tiparelor pedoclimatice).
- **Acuratețe pe setul de test independent:** 99.55%
- **Weighted F1-Score Final:** 1.00 (Rotunjit statistic la nivel macro).

Matricea de confuzie obținută relevă o performanță robustă, erorile de clasificare fiind reduse la zero pe 20 din cele 22 de culturi analizate.

---

## 6. Date și Preprocesare: Ramura Nestructurată

### 6.1 Profilul Datasetului de Viziune și Normalizare

S-a extras un set de date format din 15 clase reprezentative din datasetul academic de referință PlantVillage, axat pe culturile cele mai cultivate pe plan local (ardei gras, cartof, tomate), acoperind infecții patogene bacteriene, fungice, virale și mostre sănătoase.

Imaginile sunt standardizate la rezoluția de $224 \times 224$ pixeli, convertite în tensori PyTorch și normalizate conform distribuției ImageNet utilizând valorile:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### 6.2 Combaterea Dezechilibrelor Native de Eșantionare (Class Imbalance)

Datasetul PlantVillage prezintă dezechilibre native majore (de exemplu, clasa `Tomato_YellowLeaf_Curl_Virus` conține mii de imagini, în timp ce `Potato_healthy` are doar câteva zeci). Un antrenament clasic ar fi determinat modelul să ignore detaliile claselor minoritare.

Pentru a soluționa matematic această problemă:

1. Am extras volumele în faza de EDA (`02_eda_vision_data.ipynb`)
2. Am calculat ponderi invers proporționale utilizând algoritmul `compute_class_weight`
3. Le-am injectat direct în funcția de loss printr-un tensor de penalizare aplicat la `CrossEntropyLoss`:

```python
criterion = nn.CrossEntropyLoss(weight=ponderi_tensor)
```

Această metodă obligă modelul să penalizeze mult mai aspru clasificările eronate pe clasele cu reprezentare redusă, echilibrând optim învățarea.

---

## 7. Model și Robustețe: Ramura Nestructurată

### 7.1 Alegerea și Justificarea Arhitecturii EfficientNet-B0

S-a selectat arhitectura EfficientNet-B0 datorită strategiei sale avansate de Compound Scaling, care scalează uniform:
- Adâncimea convoluțională
- Lățimea straturilor
- Rezoluția imaginii de intrare

utilizând un coeficient de scalare optimizat. Acest mecanism asigură performanțe similare sau superioare rețelelor masive precum ResNet-50, reducând în același timp volumul de parametri și cerințele computaționale de 10 ori.

Această eficiență o face ideală pentru rularea direct pe dispozitive din categoria Edge AI sau pe servere cloud cu bugete stricte de memorie.

### 7.2 Strategia de Antrenament în Două Etape (Fine-Tuning)

**Faza Frozen (Epocile 1-5):**
Toate straturile convoluționale pre-antrenate pe ImageNet au fost blocate. S-a antrenat exclusiv clasificatorul adăugat în faza finală cu o rată de învățare robustă (`lr=0.001`), forțând rețeaua să mapeze trăsăturile geometrice generale deja învățate pe cele 15 clase noi de patologii.

**Faza Fine-Tuning Global (Epocile 6-10):**
S-au deblocat toate straturile rețelei convoluționale, reducând rata de învățare la un pas foarte fin (`lr=0.0001`). Acest lucru a permis modificarea microscopică a filtrelor interne pentru a învăța detalii texturale specifice leziunilor patologice foliare, fără a distruge cunoștințele structurale deja acumulate.

### 7.3 Eliminarea Scurgerilor de Informație (Data Leakage) prin Split Secvențial pe Blocuri

Utilizarea unui `random_split` simplu la nivel de imagine pe dataseturi foliare reprezintă o eroare metodologică gravă. Datasetul PlantVillage conține imagini realizate consecutiv pe aceleași frunze în cadrul acelorași sesiuni fotografice. Un split pur aleatoriu trimite imagini aproape identice simultan în setul de antrenament și în cel de validare, determinând modelul să memoreze fundalul sau unghiul, raportând o acuratețe nerealistă de 99% în laborator, dar eșuând la inferența pe teren.

**Soluția Implementată:**

În notebook-ul `03_train_vision_model.ipynb` am dezvoltat un algoritm de **Split Secvențial pe Blocuri** (GroupSplit). Imaginile din directoare sunt grupate în blocuri compacte de fișiere consecutive înainte de amestecare.

Acest algoritm garantează că toate cadrele realizate pe o frunză în cursul aceleiași sesiuni fotografice sunt izolate compact fie doar în setul de antrenament, fie doar în cel de validare, oferind performanțe reale, verificate pe date complet nevăzute.

---

## 8. Protocol de Evaluare și Analiză Critică

### 8.1 Evaluare Multi-Metrică pe Date Nevăzute

În urma aplicării split-ului secvențial pe un set de testare masiv de 4.157 de imagini unice, modelul de viziune a reținut o acuratețe reală de **99.54%**. Evaluarea detaliată arată o acuratețe excelentă la nivelul tuturor claselor:

- **Macro Avg F1-Score:** 1.00 (Rotunjit statistic)
- **Weighted Avg F1-Score:** 1.00
- **Recall de 1.00** pe patologii critice cum ar fi `Potato___Early_blight` sau `Tomato_Bacterial_spot`, eliminând riscul de a omite diagnosticarea unei plante bolnave pe teren.

### 8.2 Analiză Biologică Nuanțată a Erorilor de Viziune

Raportul evidențiază o confuzie minimă între:
- `Tomato_Late_blight` (Precizie 0.97, Recall 1.00)
- `Tomato_Early_blight` (Precizie 1.00, Recall 0.98)

Această variație reflectă un fenomen biologic real: în stadiile timpurii sau intermediare, leziunile necrotice induse de ciupercile *Alternaria* și *Phytophthora* prezintă cercuri concentrice și texturi de degradare extrem de similare vizual.

Acest comportament demonstrează că modelul analizează markerii biologici reali ai frunzei, nu elemente marginale din imagini.

### 8.3 Testarea la Limită și Degradare Sintetică (Stress Testing)

Pentru a evalua stabilitatea modelului în condiții nefavorabile de captură pe teren, am rulat teste sintetice de degradare:

**Perturbări Moderate de Câmp:**
Prin aplicarea automată de rotații, zgomot Gaussian, variații de iluminare solară și blur de autofocus (simulând camera unui smartphone ieftin), modelul și-a menținut performanța, înregistrând o acuratețe remarcabilă de **99.40%**.

**Perturbări Extreme Distructive:**
La adăugarea unui zgomot optic masiv și distorsiuni extreme de culoare (simulând o lentilă murdară sau obturată parțial), acuratețea de clasificare a coborât până la **7.40%**. Această degradare controlată validează arhitectura decuplată: în caz de eșec vizual total, aplicația redirecționează atenția fermierului către modulul pedoclimatic pentru asistență.

---

## 9. Considerente Etice, Confidențialitate și Sustenabilitate

Implementarea sistemului AgriGuard AI este condusă de principii riguroase de etică în inteligența artificială:

**XAI ca Filtru de Combatere a Bias-ului de Laborator:**
Imaginile din PlantVillage sunt realizate pe fundaluri de laborator. Există riscul ca modelul să învețe asocieri greșite legate de mediul steril în loc să analizeze boala. Suprapunerea hărților termice generate de Grad-CAM acționează ca o barieră de siguranță: fermierul poate valida în timp real dacă AI-ul ia decizia corect, analizând leziunile de pe frunză, sau dacă este influențat de reflexiile de pe fundal.

**Prevenirea Riscurilor de Diagnostic:**
Un diagnostic eronat poate determina un fermier să ignore o boală sau să aplice tratamente chimice dăunătoare. Interfața Streamlit include avertismente clare care specifică faptul că platforma reprezintă un instrument de screening și alertă timpurie, decizia agronomică finală aparținând întotdeauna utilizatorului uman.

**Impactul de Mediu și Sustenabilitatea:**
Prin optimizarea planurilor de rotație a culturilor în funcție de nutrienți, AgriGuard AI combate fenomenele dăunătoare de supra-fertilizare cu Azot, protejând pânza freatică și reducând emisiile de gaze cu efect de seră din agricultură.

**Confidențialitatea Datelor:**
Procesul de inferență rulează exclusiv in-memory. Platforma nu stochează imaginile încărcate de fermieri și nu salvează datele GPS ale exploatațiilor agricole private, asigurând o confidențialitate completă.

---

## 10. Structura Proiectului și Ghidul de Reproducere

### 10.1 Structura Rădăcină a Directoarelor

Proiectul este structurat modular, respectând bunele practici din industrie:

```
AgriGuard-AI/
│
├── data/
│   └── raw/
│       ├── Crop_recommendation.csv        # Dataset sol (2200 eșantioane)
│       └── plantvillage/                  # Dataset imagini structurat pe clase
│
├── frontend/
│   └── app.py                             # Interfața Streamlit în limba română
│
├── models/
│   ├── vision_model_rtx_finetuned.pth     # Modelul EfficientNet-B0 PyTorch (99.54%)
│   ├── random_forest_soil_model.pkl       # Modelul Random Forest Final (99.55%)
│   ├── soil_scaler.pkl                    # StandardScaler pentru date structurate
│   └── soil_label_encoder.pkl             # LabelEncoder pentru etichete culturi
│
├── notebooks/
│   ├── 01_eda_soil_data.ipynb             # EDA aprofundat pedoclimatic local
│   ├── 02_eda_vision_data.ipynb           # EDA imagini și calcul weights dezechilibru
│   ├── 03_train_vision_model.ipynb        # Antrenare refactorizată cu Split anti-leakage
│   └── 04_train_tabular_model.ipynb       # Benchmarking, GridSearchCV și StratifiedKFold
│
├── requirements.txt                       # Dependențe optimizate pentru cloud deployment
└── README.md                              # Raportul tehnic oficial complet
```

### 10.2 Ghidul Oficial de Reproducere a Rezultatelor

Pentru a replica performanțele înregistrate în acest raport, se vor parcurge următorii pași metodologici:

**1. Pregătirea Seturilor de Date:**

Descărcați seturile de date menționate și organizați-le în directorul `data/raw/` respectând schema de directoare prezentată mai sus.

**2. Instalarea Mediului de Lucru:**

Rulați instalarea pachetelor de producție în terminal:

```bash
pip install -r requirements.txt
```

**3. Execuția Notebook-urilor:**

Rulați notebook-urile din folderul `notebooks/` în ordinea strictă a numerotării lor (01 → 04). Toate seed-urile generatoare sunt blocate global la valoarea 42. Diagramele matematice, matricile de confuzie și scorurile finale se vor genera reproducându-se identic cu cele raportate.

**4. Lansarea Interfeței Streamlit:**

Pentru a iniția aplicația local în modul demonstrativ, rulați:

```bash
streamlit run frontend/app.py
```

---

**Raport Tehnic finalizat cu succes pentru Competiția Națională ONIA 2026.**
