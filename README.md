# 🌱 AgriGuard AI — Sistem Hibrid de Inteligență Artificială pentru Diagnosticarea Patologiilor Vegetale și Optimizarea Culturilor cu Explicabilitate Algoritmică (XAI)

---

## Cadre Identitare și Administrative

* **Autori:** Nicolai Sîrețanu, Alexandru Caldare
* **Coordonare Științifică și Mentorat:** Nadejda Sinițîn
* **Apartenență Instituțională:** Liceul Teoretic "Ion Pelivan", Răzeni, Ialoveni, Republica Moldova
* **Destinație Competitivă:** Competiția Națională ONIA 2026 (Olimpiada Națională de Inteligență Artificială)

---

## 1. Definirea Problemei și Impactul Agronomic

### 1.1 Contextul Global și Necesitatea Soluției
Agricultura modernă se confruntă cu o presiune sistemică dublă: pierderile semnificative de biomasă vegetală cauzate de fitopatogeni și degradarea fizico-chimică a solurilor determinată de monoculturi repetitive sau fertilizări empirice, necalibrate. Identificarea bolilor foliare se bazează tradițional pe inspecția vizuală macroscopică realizată de specialiști agronomi, un proces cu o inerție ridicată, costisitor și predispus la interpretări subiective în stadiile timpurii ale infecțiilor.

Deși implementarea rețelelor neurale convoluționale de Deep Learning a adus precizii ridicate în medii controlate, aceste sisteme suferă de o vulnerabilitate critică: opacitatea decizională ("Black Box"). Un fermier nu poate adopta cu încredere o recomandare automatizată de tratament chimic dacă algoritmul nu poate justifica regiunile anatomice ale frunzei care au declanșat alerta. AgriGuard AI elimină această barieră prin integrarea unui modul activ de Explicabilitate Vizuală (XAI).

### 1.2 Profilul Beneficiarilor și Valoarea Practică
Sistemul se adresează direct micilor producători ce exploatează sere, dar și operatorilor marilor suprafețe agricole din Republica Moldova. Aceștia primesc un instrument integrat de screening capabil să ruleze pe dispozitive cu resurse computaționale modeste (smartphones), optimizând luarea deciziilor direct pe teren.

### 1.3 Obiective de Impact Măsurabile
Valoarea practică a sistemului este definită de un set de indicatori de performanță agronomică monitorizabili direct pe teren:
* **Diminuarea utilizării pesticidelor de sinteză cu 15% - 20%** prin identificarea localizată a focarelor infecțioase și aplicarea exclusiv a tratamentelor zonale, evitând pulverizările generale profilactice.
* **Reducerea pierderilor de recoltă cu până la 12%** anual, datorită depistării precoce a patologiilor foliare înainte ca acestea să atingă un prag de dispersie epidemică.
* **Eficientizarea cu 18% a utilizării fertilizanților** prin corelarea directă a planului de cultură cu macro-nutrienții existenți în sol, reducând costurile operaționale ale exploatației.

---

## 2. Arhitectura Sistemului Hibrid

Platforma integrează un sistem multi-modal ce decuplează procesarea datelor structurate de cele nestructurate în două servicii Machine Learning complementare, unificate la nivel de backend printr-un modul de validare ierarhică a consistenței logice.

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
Filtru OOD: Softmax Threshold                 Tuning: GridSearchCV + SKFold
|                                               |
+-----------------------+-----------------------+
|
v
+-----------------------------------+
|    MOTOR DE FUZIUNE ALGORITMICĂ   |
|     (Intercepție Anomalii Eco)    |
+-----------------+-----------------+
|
v
+-----------------------------------+
|     Raport de Diagnostic Hibrid   |
+-----------------------------------+


### 2.1 Serviciul ML pe Date Structurate (Recomandare Culturi)
* **Model:** Ansamblu de arbori de decizie de tip Random Forest, optimizat prin căutare sistematică în rețea.
* **Input:** Vector numeric format din 7 caracteristici fizico-chimice și macro-climatice: `[N, P, K, temperatură, umiditate, pH, precipitații]`.
* **Output:** Distribuție probabilistică multiclasă peste cele 22 de tipuri de culturi agricole suportate.
* **Rol:** Planificarea rotației culturilor, evitarea epuizării solului și optimizarea utilizării resurselor pedoclimatice.

### 2.2 Serviciul ML pe Date Nestructurate (Clasificare Patologică)
* **Model:** Rețea neurală convoluțională EfficientNet-B0 optimizată prin Transfer Learning fin, cuplată cu un hook de extracție Grad-CAM.
* **Input:** Imagine color brută a suprafeței foliare (RGB, rezoluție standardizată la $224 \times 224$).
* **Output:** Clasificarea stării fitosanitare a plantei în 15 categorii (bolnave/sănătoase) și suprapunerea unei hărți termice bidimensionale de activare a atenției.
* **Rol:** Detecție, diagnosticare de precizie și vizualizarea markerilor patogeni.

### 2.3 Sinergie, Fuziune Inter-Model și Service Degradation
Cele două servicii adresează stadii diferite din fluxul de producție agronomic: solul coordonează faza de pre-semănare, iar analiza foliară ghidează perioada de vegetație. 

Fuziunea algoritmică este realizată printr-o matrice de consistență eco-agronică implementată în backend. Dacă predictivul de sol indică o cultură cu asolament inundat (ex: Orez) dintr-un set de sliders manuali introduși eronat, iar viziunea computerizată identifică o boală foliară a unei culturi xerofite sau de mediu uscat (ex: Tomate - Solanaceae), sistemul interceptează paralelizarea oarbă, semnalând utilizatorului o anomalie critică de consistență biologică în interfață.

Din punct de vedere al siguranței în funcționare, platforma implementează un mod de degradare controlată a serviciilor. În cazul în care calitatea camerei de pe teren este afectată (zgomot optic, praf pe lentilă sau lipsă de focalizare), modulul de viziune computerizată se dezactivează, ale cărui date devin neconcludente, dar utilizatorul continuă să beneficieze de funcționalitățile complete de asistență pedoclimatică furnizate de modelul de analiză a solului.

---

## 3. Date și Preprocesare: Ramura Structurată

### 3.1 Proveniența Datelor și Pipeline-ul de Producție
Sursa primară de date structurate o constituie benchmark-ul public *Kaggle Crop Recommendation Dataset*, alcătuit din 2200 de eșantioane experimentale multi-locație. Pentru securizarea pipeline-ului de producție live împotriva eventualelor avarii sau pierderi de pachete de la senzorii IoT din teren, am integrat în faza de preprocesare un bloc de imputare automatizat bazat pe `SimpleImputer(strategy='median')`. Acesta rulează preventiv înainte de scalarea realizată de `StandardScaler`. Variabila țintă este codificată prin `LabelEncoder` într-un spațiu discret $[0, 21]$.

### 3.2 Analiza Exploratorie a Datelor (EDA)
În cadrul etapei de analiză din notebook-ul `01_eda_soil_data.ipynb` au fost efectuate următoarele operațiuni statistice:
* **Heatmap de Corelație (Pearson):** S-a evidențiat o corelație liniară strânsă ($0.73$) între Fosfor (P) și Potasiu (K), dictând reducerea variabilității prin utilizarea hiperparametrului `max_features='sqrt'` în faza de antrenare, prevenind astfel colapsul diversității decizionale din cadrul ansamblului de arbori.
* **Izolarea și Evaluarea Outlierilor:** Reprezentările boxplot au identificat abateri extreme pozitive pe variabilele de azot (`N`) și precipitații (`rainfall`). Acestea au fost păstrate documentat în setul de antrenament pentru a asigura capacitatea modelului de a procesa fenomene pedoclimatice reale din teren (perioade severe de secetă sau fertilizări concentrate masive).
* **Certificarea Stratificării:** Setul de test reține un echilibru strict, conținând exact 20 de eșantioane reprezentative per clasă dintr-un total de 440 utilizate la evaluarea finală.

### 3.3 Adaptarea la Specificul Pedoclimatic al Republicii Moldova
Seturile de date standardizate reflectă adesea medii agronomice generalizate. Solurile din Republica Moldova prezintă proprietăți distincte ce au fost adresate în arhitectura sistemului:
* **Profilul Cernoziomurilor locale:** Deși dispun de un volum ridicat de humus nativ, solurile din Moldova înregistrează adesea un deficit istoric de Fosfor (P) mobil, blocat sub formă de săruri insolubile din cauza legării chimice cu calcarul activ.
* **Profilul Climatic Regional:** Vulnerabilitatea ridicată la secetă din zonele de Sud impune o corelare strictă între umiditatea relativă a aerului și regimul de precipitații.
* **Arhitectură fără API-uri externe (Zero-API Deployment):** Pentru a asigura funcționarea neîntreruptă pe câmp în zone rurale izolate fără acoperire stabilă de internet, aplicația rulează în regim complet offline și autonom. Sistemul nu depinde de API-uri terțe sau conexiuni la servere externe; toți parametrii de mediu sunt preluați direct de la senzorii hardware locali instalați pe teren sau prin introducere manuală, garantând zero latență și independență totală de rețea.

---

## 4. Model ML: Ramura Structurată

### 4.1 Benchmarking și Justificarea Selecției
Evaluarea sistematică realizată în notebook-ul `04_train_tabular_model.ipynb` a comparat performanțele a patru paradigme matematice diferite pe setul de test:
1. **K-Nearest Neighbors (KNN):** Train Acc: 98.47% | Test Acc: 97.95%
2. **Support Vector Classifier (SVM RBF):** Train Acc: 98.58% | Test Acc: 98.41%
3. **XGBoost Classifier:** Train Acc: 100.00% | Test Acc: 98.86%
4. **Random Forest Classifier (Baseline):** Train Acc: 100.00% | Test Acc: 99.55%

Modelul **Random Forest** a demonstrat o capacitate superioară de generalizare directă pe datele de test, fiind selectat ca motor de predicție tabulară în producție.

### 4.2 Combaterea Overfitting-ului prin Regularizare Structurală
Un scor perfect de 100.00% pe datele de antrenare (obținut de baseline-urile Random Forest și XGBoost în primele faze) indica o adâncime decizională excesivă (`max_depth=None`), modelul memorând zgomotul din date. Pentru a combate această problemă de supra-ajustare, am aplicat o regularizare structurală strictă, forțând restricționarea adâncimii maxime a arborilor (`max_depth`) și setarea unui prag minim de eșantioane necesare pentru splitarea unui nod intern (`min_samples_split`).

### 4.3 Eliminarea Soluțiilor Ineficiente de Producție (Refactorizare)
A fost eliminată complet utilizarea unui model Random Forest salvat sub o denumire eronată din considerente de evitare a refactorizării codului. Modelul final este salvat sub propria identitate matematică: `random_forest_soil_model.pkl`, iar toate variabilele și referințele din scriptul Streamlit (`app.py`) au fost restructurate pentru a respecta trasabilitatea exactă a datelor.

---

## 5. Protocol de Evaluare și Robustețe: Ramura Structurată

### 5.1 Validare Încrucișată Stratificată și Hiperparametrii Optimi Finali
Pentru a exclude bias-ul statistic introdus de raportarea performanțelor pe un singur split fix, am implementat un protocol robust de Validare Încrucișată Stratificată cu 5 Folduri (5-Fold StratifiedKFold) combinat cu o căutare automată a hiperparametrilor optimi prin `GridSearchCV`. 

În urma optimizării, hiperparametrii selectați de algoritm pentru producție sunt:
* `n_estimators`: 100
* `max_depth`: 8
* `min_samples_split`: 6
* `max_features`: 'sqrt'
* `criterion`: 'gini'

### 5.2 Analiză Critică: Limitarea Liniarității Setului de Date
După execuția protocolului de optimizare hiperparametrică, modelul a înregistrat o acuratețe stabilizată prin Validare Încrucișată pe Train de 99.31% și o acuratețe pe setul de test independent de 99.55%.

**Notă critică obligatorie:** Raportarea unui Weighted F1-Score general de 1.00 macro-rotunjit și erori absolute egale cu zero pe 20 din cele 22 de clase reprezintă un indicator clar al faptului că datasetul de referință utilizat este înalt idealizat, liniar și ușor separabil. În condiții reale de teren, zgomotul profund al solurilor din ecosisteme nesabotate (cross-contaminarea cu fertilizanți, variabilitatea bio-disponibilității elementelor chimice cauzată de temperatură și bacterii) generează suprapuneri masive de clase. Acest model reprezintă o bază stabilă de calibrare teoretică (baseline), însă implementarea lui reală necesită un strat extins de calibrare empirică pe mostre colectate local.

---

## 6. Date și Preprocesare: Ramura Nestructurată

### 6.1 Profilul Datasetului de Viziune, Rezoluții Native și Limitări de Mediu
S-a extras un set de date format din 15 clase reprezentative din datasetul academic de referință *PlantVillage*, axat pe culturile cele mai cultivate pe plan local (ardei gras, cartof, tomate). Rezoluția nativă medie a imaginilor înainte de preprocesare este de $256 \times 256$ pixeli. Pipeline-ul le standardizează la rezoluția de $224 \times 224$ pixeli, convertindu-le în tensori PyTorch și normalizându-le conform distribuției ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

**Limitare Critică Documentată:** Datasetul *PlantVillage* este capturat într-un mediu steril de laborator, utilizând fundaluri perfect plane și neutre. Aceasta reprezintă o limitare majoră pentru un deployment în lumea reală. Pe teren, imaginile foliare prezintă zgomot complex de fundal (sol, buruieni, umbre alternante, insecte, alte frunze parțial suprapuse). Pentru a asigura robustețea împotriva datelor din afara distribuției sau a zgomotului ambiental distructiv, am dezvoltat în producție un Filtru Out-of-Distribution (OOD) bazat pe prag de siguranță Softmax combinat cu un mecanism de intercepție descris la secțiunile următoare.

### 6.2 Combaterea Dezechilibrelor Native de Eșantionare (Class Imbalance)
Datasetul PlantVillage prezintă dezechilibre native severe (clasa `Tomato_YellowLeaf_Curl_Virus` conține mii de mostre, în timp ce `Potato_healthy` are doar câteva zeci). Ignorarea acestui aspect forțează rețeaua neurală să își penalizeze erorile în favoarea claselor majoritare.

Pentru a soluționa matematic această problemă, am extras distribuția volumului în EDA (`02_eda_vision_data.ipynb`) și am calculat ponderi invers proporționale utilizând algoritmul `compute_class_weight` din Scikit-Learn. Aceste ponderi au fost transformate într-un tensor PyTorch și injectate direct în funcția de cost:

```python
criterion = nn.CrossEntropyLoss(weight=ponderi_tensor)
```

Această metodă obligă modelul să penalizeze mai dur ratarea unei imagini din clasele minoritare, asigurând un antrenament perfect echilibrat.

---

## 7. Model și Robustețe: Ramura Nestructurată
### 7.1 Alegerea și Justificarea Arhitecturii EfficientNet-B0
S-a selectat arhitectura EfficientNet-B0 datorită strategiei sale avansate de Compound Scaling, care scalează uniform adâncimea convoluțională, lățimea straturilor și rezoluția imaginii de intrare utilizând un coeficient de scalare optimizat. Acest mecanism asigură performanțe similare sau superioare rețelelor masive precum ResNet-50, reducând în același timp volumul de parametri și cerințele computaționale de 10 ori. Această eficiência o face ideală pentru rularea direct pe dispozitive din categoria Edge AI sau pe servere cloud cu bugete stricte de memorie.

### 7.2 Strategia de Antrenament în Două Etape (Fine-Tuning)
Faza Frozen (Epocile 1-5): Toate straturile convoluționale pre-antrenate pe ImageNet au fost complet blocate. S-a antrenat exclusiv clasificatorul adăugat în faza finală cu o rată de învățare robustă (lr=0.001), forțând rețeaua să mapeze trăsăturile geometrice generale deja învățate pe cele 15 clase noi de patologii.

Faza Fine-Tuning Global (Epocile 6-10): S-au deblocat toate straturile rețelei convoluționale, reducând rata de învățare la un pas foarte fin (lr=0.0001). Acest lucru a permis modificarea microscopică a filtrelor interne pentru a învăța detalii texturale specifice leziunilor patologice foliare, fără a distruge cunoștințele structurale deja acumulate.

### 7.3 Eliminarea Scurgerilor de Informație (Data Leakage) prin Split Secvențial pe Blocuri
Utilizarea unui random_split simplu la nivel de imagine pe dataseturi foliare reprezintă o eroare metodologică gravă. Datasetul PlantVillage conține imagini realizate consecutiv pe aceleași frunze în cadrul aceluiași sesiuni fotografice. Un split pur aleatoriu trimite imagini aproape identice simultan în setul de antrenament și în cel de validare, determinând modelul să memoreze fundalul sau unghiul, raportând o acuratețe nerealistă de 99% în laborator, dar eșuând la inferența pe teren.

Soluția Implementată: În notebook-ul 03_train_vision_model.ipynb am dezvoltat un algoritm de Split Secvențial pe Blocuri / GroupSplit. Imaginile din directoare sunt grupate în blocuri compacte de fișiere consecutive înainte de amestecare. Acest algoritm garantează că toate cadrele realizate pe o frunză în cursul aceleiași sesiuni fotografice sunt izolate compact fie doar în setul de antrenament, fie doar în cel de validare, oferind performanțe reale, verificate pe date complet nevăzute.

---

## 8. Protocol de Evaluare și Analiză Critică
### 8.1 Evaluare Multi-Metrică pe Date Nevăzute
În urma aplicării split-ului secvențial pe un set de testare masiv de 4.157 de imagini unice, modelul de viziune a reținut o acuratețe reală de 99.54%. Evaluarea detaliată arată o acuratețe excelentă la nivelul tuturor claselor:

Macro Avg F1-Score: 1.00 (Rotunjit statistic).

Weighted Avg F1-Score: 1.00

Recall de 1.00 pe patologii critice cum ar fi Potato___Early_blight sau Tomato_Bacterial_spot, eliminând riscul de a omite diagnosticarea unei plante bolnave pe teren.

### 8.2 Analiză Biologică Nuanțată a Erorilor de Viziune
Raportul evidențiază o confuzie minimă între Tomato_Late_blight (Precizie 0.97, Recall 1.00) și Tomato_Early_blight (Precizie 1.00, Recall 0.98). Această variație reflectă un fenomen biologic real: în stadiile timpurii sau intermediare, leziunile necrotice induse de ciupercile Alternaria și Phytophthora prezintă tipare texturale și inele concentrice extrem de asemănătoare vizual. Acest comportament demonstrează că modelul analizează markerii biologici reali ai frunzei, nu elemente marginale din imagini.

### 8.3 Intercepția Datelor Out-of-Distribution (Mecanismul de Control OOD)
Pentru a împiedica modelul computer vision să genereze alerte fitosanitare complet eronate atunci când i se prezintă cadre non-plantă (foi de scris, texturi arbitrare, fundaluri zgomotoase sau culturi nesuportate), am implementat în backend un filtru probabilist de siguranță bazat pe pragul Softmax (τ=0.75).

Dacă valoarea maximă a vectorului probabilistic de output scade sub pragul τ, aplicația interceptează automat afișarea diagnosticelor fitosanitare false și activează starea reziduală controlată: "Obiect Necunoscut / Cadru Non-Vegetal Detected", invitând utilizatorul să re-încadreze frunza pe un fundal curat.

### 8.4 Testarea la Limită și Degradare Sintetică (Stress Testing)
Pentru a evaluat stabilitatea modelului în condiții nefavorabile de captură pe teren, am rulat teste sintetice de degradare:

Perturbări Moderate de Câmp: Prin aplicarea automată de rotații, zgomot Gaussian, variații de iluminare solară și blur de autofocus (simulând camera unui smartphone ieftin), modelul și-a menținut performanța, înregistrând o acuratețe remarcabilă de 99.40%.

Perturbări Extreme Distructive: La adăugarea unui zgomot optic masiv și distorsiuni extreme de culoare (simulând o lentilă murdară sau obturată parțial), acuratețea de clasificare a coborât până la 7.40%. Această degradare controlată validează arhitectura decuplată: în caz de eșec vizual total, aplicația redirecționează atenția fermierului către modulul pedoclimatic pentru asistență.

---

## 9. Considerente Etice, Confidențialitate și Sustenabilitate
Implementarea sistemului AgriGuard AI este condusă de principii riguroase de etică în inteligența artificială:

XAI ca Filtru de Combatere a Bias-ului de Laborator: Imaginile din PlantVillage sunt realizate pe fundaluri de laborator. Există riscul ca modelul să învețe asocieri greșite legate de mediul steril în loc să analizeze boala. Suprapunerea hărților termice generate de Grad-CAM acționează ca o barieră de siguranță: fermierul poate valida în timp real dacă AI-ul ia decizia corect, analizând leziunile de pe frunză, sau dacă este influențat de reflexiile de pe fundal.

Prevenirea Riscurilor de Diagnostic: Un diagnostic eronat poate determina un fermier să ignore o boală sau să aplice tratamente chimice dăunătoare. Interfața Streamlit include avertismente clare care specifică faptul că platforma reprezintă un instrument de screening și alertă timpurie, decizia agronomică finală aparlığând întotdeauna utilizatorului uman.

Impactul de Mediu și Sustenabilitatea: Prin optimizarea planurilor de rotație a culturilor în funcție de nutrienți, AgriGuard AI combate fenomenele dăunătoare de supra-fertilizare cu Azot, protejând pânza freatică și reducând emisiile de gaze cu efect de seră din agricultură.

Confidențialitatea Datelor: Processul de inferență rulează exclusiv in-memory. Platforma nu stochează imaginile încărcate de fermieri și nu salvează datele GPS ale exploatațiilor agricole private, asigurând o confidențialitate completă.

---

## 10. Structura Proiectului și Ghidul de Reproducere
### 10.1 Structura Rădăcină a Directoarelor
Proiectul este structurat modular, respectând bunele practici din industrie:

```plaintext
AgriGuard-AI/
│
├── data/
│   └── raw/
│       ├── Crop_recommendation.csv        # Dataset sol (2200 eșantioane)
│       └── plantvillage/                  # Dataset imagini structurat pe clase
│
├── frontend/
│   └── app.py                             # Interfața Streamlit cu Filtru OOD și Fuziune
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

Pregătirea Seturilor de Date: Descărcați seturile de date menționate și organizați-le în directorul data/raw/ respectând schema de directoare prezentată mai sus.

Instalarea Mediului de Lucru: Rulați instalarea pachetelor de producție rulând în terminal:

```bash
pip install -r requirements.txt
```

Execuția Notebook-urilor: Rulați notebook-urile din folderul notebooks/ în ordinea strictă a numerotării lor (01 -> 04). Toate seed-urile generatoare sunt blocate global la valoarea 42. Diagramele matematice, matricile de confuzie și scorurile finale se vor genera reproducându-se identic cu cele raportate.

Lansarea Interfeței Streamlit: Pentru a iniția aplicația local în modul demonstrativ, rulați:

```bash
streamlit run frontend/app.py
```
Raport Tehnic finalizat cu succes pentru Competiția Națională ONIA 2026.