# AgriGuard AI — Sistem Hibrid de Inteligență Artificială pentru Diagnosticarea Patologiilor Vegetale

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
Valoarea practică a sistemului este definită de un set de indicatori de performanță agronomică monitorizabili direct pe teren prin intermediul unui design de studiu pilot controlat (comparând o parcelă experimentală ghidată de AgriGuard AI cu o parcelă de control administrată tradițional):
* **Diminuarea utilizării pesticidelor de sinteză cu 15% - 20%** prin identificarea localizată a focarelor infecțioase și aplicarea exclusiv a tratamentelor zonale, evitând pulverizările generale profilactice.
* **Reducerea pierderilor de recoltă cu până la 12%** anual, datorită depistării precoce a patologiilor foliare înainte ca acestea să atingă un prag de dispersie epidemică.
* **Eficientizarea cu 18% a utilizării fertilizanților** prin corelarea directă a planului de cultură cu macro-nutrienții existenți în sol, reducând costurile operaționale ale exploatației.

---

## 2. Arhitectura Sistemului Hibrid

Platforma integrează un sistem multi-modal ce decuplează procesarea datelor structurate de cele nestructurate în două servicii Machine Learning complementare, unificate la nivel de backend printr-un modul de validare ierarhică a consistenței logice.

```
                ┌─────────────────────────────┐
                │   INTERFAȚA STREAMLIT       │
                └──────────────┬──────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                                      │
            v                                      v
┌──────────────────────────┐      ┌──────────────────────────┐
│ DATE NESTRUCTURATE:      │      │ DATE STRUCTURATE:        │
│ IMAGINE                  │      │ SENZORI                  │
└──────────────┬───────────┘      └──────────┬───────────────┘
               │                             │
               v                             v
┌──────────────────────────┐      ┌──────────────────────────┐
│ EfficientNet-B0 (PyTorch)│      │ Random Forest            │
│ Input: Imagine frunză    │      │ Input: [N, P, K, T,     │
│ (RGB)                    │      │         H, pH, Rain]     │
│ Output: Probabilități    │      │ Output: Probabilități    │
│ clasă                    │      │ cultură                  │
│ Filtru OOD: Softmax τ    │      │ Tuning: GridSearchCV +   │
│ = 0.75                   │      │ StratifiedKFold          │
└──────────────┬───────────┘      └──────────┬───────────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              v
               ┌──────────────────────────────┐
               │  MOTOR DE FUZIUNE            │
               │  ALGORITMICĂ                 │
               │  (Intercepție Anomalii Eco)  │
               └──────────────┬───────────────┘
                              │
                              v
               ┌──────────────────────────────┐
               │  Raport de Diagnostic Hibrid │
               └──────────────────────────────┘
```

### 2.1 Serviciul ML pe Date Structurate (Recomandare Culturi)
* **Model:** Ansamblu de arbori de decizie de tip Random Forest, optimizat prin căutare sistematică în rețea.
* **Input:** Vector numeric format din 7 caracteristici fizico-chimice și macro-climatice: `[N, P, K, temperatură, umiditate, pH, precipitații]`.
* **Output:** Distribuție probabilistică multiclasă peste tipurile de culturi agricole suportate.
* **Rol:** Planificarea rotației culturilor, evitarea epuizării solului și optimizarea utilizării resurselor pedoclimatice.

### 2.2 Serviciul ML pe Date Nestructurate (Clasificare Patologică)
* **Model:** Rețea neurală convoluțională EfficientNet-B0 optimizată prin Transfer Learning fin, cuplată cu un hook de extracție Grad-CAM.
* **Input:** Imagine color brută a suprafeței foliare (RGB, rezoluție standardizată la $224 \times 224$).
* **Output:** Clasificarea stării fitosanitare a plantei în 15 categorii (bolnave/sănătoase) și suprapunerea unei hărți termice bidimensionale de activare a atenției.
* **Rol:** Detecție, diagnosticare de precizie și vizualizarea markerilor patogeni.

### 2.3 Sinergie, Fuziune Inter-Model și Service Degradation
Cele două servicii adresează stadii diferite din fluxul de producție agronomic: solul coordonează faza de pre-semănare, iar analiza foliară ghidează perioada de vegetație. 

Fuziunea algoritmică este realizată printr-o matrice de consistență eco-agronică implementată în backend. Dacă predictivul de sol indică o cultură cu asolament inundat dintr-un set de sliders manuali introduși eronat, iar viziunea computerizată identifică o boală foliară a unei culturi de mediu uscat (ex: Tomate - Solanaceae), sistemul interceptează paralelizarea oarbă, semnalând utilizatorului o anomalie critică de consistență biologică în interfață.

Din punct de vedere al siguranței în funcționare, platforma implementează un mod de degradare controlată a serviciilor. În cazul în care calitatea camerei de pe teren este afectată (zgomot optic, lipsă de focalizare reflectată de filtrul Out-of-Distribution), modulul de viziune computerizată se dezactivează, dar utilizatorul continuă să beneficieze de funcționalitățile complete de asistență pedoclimatică furnizate de modelul de analiză a solului.

---

## 3. Date și Preprocesare: Ramura Structurată

### 3.1 Proveniența Datelor și Pipeline-ul de Producție
Sursa primară de date structurate o constituie benchmark-ul public *Kaggle Crop Recommendation Dataset*, alcătuit din 2200 de eșantioane experimentale multi-locație. Pentru securizarea pipeline-ului de producție live împotriva eventualelor avarii sau pierderi de pachete de la senzorii IoT din teren, am integrat în faza de preprocesare un bloc de imputare automatizat bazat pe `SimpleImputer(strategy='median')`. Acesta rulează preventiv înainte de scalarea realizată de `StandardScaler`. Variabila țintă este codificată prin `LabelEncoder`.

### 3.2 Analiza Exploratorie a Datelor (EDA)
În cadrul etapei de analiză din notebook-ul `01_eda_soil_data.ipynb` au fost efectuate următoarele operațiuni statistice:
* **Heatmap de Corelație (Pearson):** S-a evidențiat o corelație liniară strânsă (**0.73**) între Fosfor (P) și Potasiu (K). Aceasta este singura dependență liniară puternică din setul de date, dictând reducerea variabilității prin utilizarea hiperparametrului `max_features='sqrt'` în faza de antrenare, prevenind astfel colapsul diversității decizionale din cadrul ansamblului de arbori.
* **Izolarea și Evaluarea Outlierilor:** Reprezentările boxplot și violin plot au identificat abateri extreme pozitive pe variabilele de azot (`N`) și precipitații (`rainfall`). Acestea au fost păstrate documentat în setul de antrenament pentru a asigura capacitatea modelului de a procesa fenomene pedoclimatice reale din teren (perioade severe de secetă sau fertilizări concentrate masive).
* **Certificarea Stratificării:** Setul de test reține un echilibru strict, conținând un volum reprezentativ și perfect egal de eșantioane per clasă utilizate la evaluarea finală.

### 3.3 Adaptarea la Specificul Pedoclimatic al Republicii Moldova
Seturile de date standardizate reflectă adesea medii agronomice tropicale generalizate, incluzând culturi complet irelevante regional (orez, cafea, iute). Pentru a regionaliza soluția, am dezvoltat scriptul `scripts/simulate_moldova_data.py`, care generează și integrează în pipeline profile fidele realității naționale:
* **Profilul Cernoziomurilor locale:** Solurile din Republica Moldova prezintă o concentrație natural ridicată de Potasiu mobil ($K \in [140, 240]$ mg/kg), dar înregistrează adesea un deficit istoric de Fosfor (P) bio-disponibil, blocat sub formă de săruri insolubile din cauza legării chimice cu calcarul activ.
* **Profilul Climatic Regional:** Vulnerabilitatea ridicată la secetă din zonele de Centru și Sud impune o corelare strictă între umiditatea relativă moderată a aerului ($45\% - 70\%$) și un regim de precipitații temperat-continental de stepă/silvostepă ($340 - 640$ mm).
* **Arhitectură fără API-uri externe (Zero-API Deployment):** Pentru a asigura funcționarea neîntreruptă pe câmp în zone rurale izolate fără acoperire stabilă de internet, aplicația rulează în regim complet offline și autonom. Sistemul nu depinde de API-uri terțe; toți parametrii de mediu sunt preloați direct de la senzorii hardware locali sau prin introducere manuală, garantând zero latență și independență totală de rețea.

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
Un scor perfect de 100.00% pe datele de antrenare (obținut de baseline-ul Random Forest în primele faze) indica o adâncime decizională excesivă, modelul memorând perfect zgomotul și limitările setului de date idealizat. Pentru a combate această problemă de supra-ajustare, am aplicat o regularizare structurală fermă, impunând limite stricte pe adâncimea arborilor, pe criteriul de splitare și pe numărul minim de mostre reținute într-o frunză. 

În urma calibrării, modelul a fost extras din zona de memorare perfectă, coborând conștient acuratețea de antrenare sub 100%. Acest lucru constituie dovada matematică clară a eliminării overfitting-ului structural.

### 4.3 Eliminarea Soluțiilor Ineficiente de Producție (Refactorizare)
A fost eliminată complet utilizarea modulelor salvate sub denumiri ambigue sau eronate. Modelul final este salvat sub propria identitate matematică stabilă: `random_forest_soil_model.pkl`, iar toate variabilele și referințele de scalare din scriptul Streamlit (`app.py`) au fost restructurate pentru a respecta trasabilitatea exactă a datelor.

---

## 5. Protocol de Evaluare și Robustețe: Ramura Structurată

### 5.1 Validare Încrucișată Stratificată și Hiperparametrii Optimi Finali
Pentru a exclude bias-ul statistic introdus de raportarea performanțelor pe un singur split fix, am implementat un protocol robust de Validare Încrucișată Stratificată cu 5 Folduri (`5-Fold StratifiedKFold`) combinat cu o căutare automată a hiperparametrilor optimi prin `GridSearchCV` acoperind peste 200 de combinații.

În urma procesului de tuning fin, hiperparametrii selectați de algoritm pentru punctul de echilibru stabil (Sweet Spot) sunt:
* `n_estimators`: 200
* `max_depth`: 12
* `min_samples_split`: 12
* `min_samples_leaf`: 4
* `max_features`: 'sqrt'
* `criterion`: 'gini'

### 5.2 Analiză Critică: Limitarea Liniarității Setului de Date
După execuția protocolului de optimizare hiperparametrică și integrarea datelor regionale, modelul a înregistrat următoarele rezultate:
* **Acuratețe pe setul de Antrenare (Train Score):** **97.60%**
* **Acuratețe în Validare Încrucișată (Cross-Validation Score):** **96.85%**

**Notă critică obligatorie:** Faptul că scorul de antrenare se stabilizează la 97.60% arată că modelul face greșeli controlate și extrage doar tipare matematice generalizabile, nu memorare brută. Raportarea anterioară a unor scoruri ideale de 100% reprezenta un indicator al unui dataset înalt idealizat și ușor separabil. În condiții reale de teren, zgomotul profund al solurilor din ecosisteme deschise generează suprapuneri nelineare de clase, iar noul model regularizat la 97.60% oferă o robustețe net superioară la inferența live.

---

## 6. Date și Preprocesare: Ramura Nestructurată

### 6.1 Profilul Datasetului de Viziune, Rezoluții Native și Limitări de Mediu
S-a extras un set de date format din 15 clase reprezentative din datasetul academic de referință *PlantVillage*, axat pe culturile cele mai cultivate pe plan local (ardei gras, cartof, tomate). Rezoluția nativă medie a imaginilor înainte de preprocesare este de $256 \times 256$ pixeli. Pipeline-ul le standardizează la rezoluția de $224 \times 224$ pixeli, convertindu-le în tensori PyTorch.

**Optimizare Spectrală Locală:** În loc să utilizăm orbește valorile standard de normalizare ImageNet, notebook-ul `02_eda_vision_data.ipynb` parcurge statistic colecția de imagini și extrage manual valorile matematice de `mean` (medie) și `std` (deviația standard) specifice canalelor spectrale RGB ale setului nostru foliar, optimizând faza de antrenare a rețelei convoluționale.

**Limitare Critică Documentată:** Datasetul *PlantVillage* este capturat într-un mediu steril de laborator, utilizând fundaluri perfect plane și neutre. Aceasta reprezintă o limitare majoră pentru un deployment în lumea reală, unde imaginile prezintă zgomot complex (sol, umbre alternante, buruieni). Pentru a asigura robustețea, am dezvoltat în producție un Filtru Out-of-Distribution (OOD) bazat pe prag de siguranță Softmax combinat cu un mecanism de intercepție descris la secțiunile următoare. De asemenea, am integrat un algoritm de detecție a neclarității (Blur Detection bazat pe varianța Laplacianului din OpenCV) pentru a respinge cadrele defectuoase din punct de vedere optic.

### 6.2 Combaterea Dezechilibrelor Native de Eșantionare (Class Imbalance)
Datasetul PlantVillage prezintă dezechilibre native severe. Ignorarea acestui aspect forțează rețeaua neurală să își penalizeze erorile în favoarea claselor majoritare.

Pentru a soluționa matematic această problemă, am extras distribuția volumului în EDA (`02_eda_vision_data.ipynb`) și am calculat ponderi invers proporționale utilizând algoritmul `compute_class_weight` din Scikit-Learn. Aceste ponderi au fost transformate într-un tensor PyTorch și injectate direct în funcția de cost Cross-Entropy, obligând modelul să penalizeze mai dur ratarea unei imagini din clasele minoritare, asigurând un antrenament perfect echilibrat.

---

## 7. Model și Robustețe: Ramura Nestructurată

### 7.1 Alegerea și Justificarea Arhitecturii EfficientNet-B0
S-a selectat arhitectura EfficientNet-B0 datorită strategiei sale avansate de Compound Scaling, care scalează uniform adâncimea convoluțională, lățimea straturilor și rezoluția imaginii de intrare utilizând un coeficient de scalare optimizat. Acest mecanism asigură performanțe similare sau superioare rețelelor masive precum ResNet-50, reducând în același timp volumul de parametri și cerințele computaționale de 10 ori, făcând modelul ideal pentru rularea direct pe dispozitive Edge AI (smartphones).

### 7.2 Strategia de Antrenament în Două Etape (Fine-Tuning)
* **Faza Frozen (Epocile 1-5):** Toate straturile convoluționale pre-antrenate pe ImageNet au fost complet blocate. S-a antrenat exclusiv clasificatorul adăugat în faza finală cu o rată de învățare robustă ($lr=0.001$), forțând rețeaua să mapeze trăsăturile geometrice generale deja învățate pe cele 15 clase noi de patologii.
* **Faza Fine-Tuning Global (Epocile 6-10):** S-au deblocat toate straturile rețelei convoluționale, reducând rata de învățare la un pas foarte fin ($lr=0.0001$). Acest lucru a permis modificarea microscopică a filtrelor interne pentru a învăța detalii texturale specifice leziunilor patologice foliare, fără a distruge cunoștințele structurale deja acumulate.

### 7.3 Eliminarea Scurgerilor de Informație (Data Leakage) prin Split Secvențial pe Blocuri
Utilizarea unui split aleatoriu simplu la nivel de imagine pe dataseturi foliare reprezintă o eroare metodologică gravă, deoarece imagini aproape identice realizate pe aceeași frunză ajung simultan în train și validation, raportând o acuratețe nerealistă de 99% în laborator, dar eșuând la inferența pe teren.

**Soluția Implementată:** În notebook-ul `03_train_vision_model.ipynb` am eliminat acest bug critic prin dezvoltarea unei clase wrapper custom `WrapperTransformareDataset` alături de un algoritm de Split Secvențial pe Blocuri / GroupSplit. Imaginile sunt izolate compact în blocuri consecutive înainte de amestecare, garantând că toate cadrele dintr-o sesiune fotografică sunt izolate exclusiv fie doar în setul de antrenament, fie doar în cel de validare, oferind performanțe reale pe date complet nevăzute.

---

## 8. Protocol de Evaluare și Analiză Critică

### 8.1 Evaluare Multi-Metrică și Studiu de Ablație
În urma aplicării split-ului secvențial pe un set de testare independent de 4.157 de imagini unice, modelul de viziune bazat pe EfficientNet-B0 a reținut o acuratețe reală de **99.54%** alături de un Recall de 1.00 pe patologii critice. 

Pentru a valida din punct de vedere academic selectarea modelului optic (cerința C8), am realizat un benchmark pe trei arhitecturi distincte (Studiu de Ablație):

| Arhitectură Model | Volum Parametri | Acuratețe Stabilă | Latență Inferență / Cadru | Justificare Decizie Tehnologică |
| :--- | :---: | :---: | :---: | :--- |
| **ResNet-18** | 11.7 M | 98.24% | 15 ms | Model greu, consumă resurse mari și oferă o precizie mai scăzută. |
| **MobileNetV3-Large** | 5.4 M | 98.61% | **7 ms** | Foarte rapid, însă rata de eroare este prea mare pentru patologii agresive. |
| **EfficientNet-B0 (Selectat)** | **5.3 M** | **99.55%** | 8 ms | **Optimul Frontierei Pareto:** Cel mai mic volum de parametri, precizie maximă și viteză excelentă de rulare pe Edge. |

### 8.2 Analiză Biologică Nuanțată a Erorilor de Viziune
Raportul evidențiază o confuzie minimă între Tomato_Late_blight (Precizie 0.97, Recall 1.00) și Tomato_Early_blight (Precizie 1.00, Recall 0.98). Această variație reflectă un fenomen biologic real: în stadiile timpurii sau intermediare, leziunile necrotice induse de ciupercile Alternaria și Phytophthora prezintă tipare texturale și inele concentrice extrem de asemănătoare vizual. Acest comportament demonstrează că modelul analizează markerii biologici reali ai frunzei, nu elemente marginale din imagini.

### 8.3 Intercepția Datelor Out-of-Distribution (Mecanismul de Control OOD)
Pentru a împiedica modelul computer vision să genereze alerte fitosanitare complet eronate atunci când i se prezintă cadre non-plantă (foi de scris, texturi arbitrare sau fundaluri zgomotoase), am implementat în backend un filtru probabilist de siguranță bazat pe pragul Softmax ($\tau=0.75$). Dacă valoarea maximă a vectorului probabilistic scade sub pragul $\tau$, aplicația activează starea reziduală controlată: „Filtru OOD Activat: Cadru Non-Vegetal Detectat”, suspendând modulul optic pentru a proteja sistemul.

### 8.4 Testarea la Limită și Degradare Sintetică (Stress Testing)
Pentru a evalua stabilitatea modelului în condiții nefavorabile de captură pe teren, am rulat teste sintetice de degradare:
* **Perturbări Moderate de Câmp:** Prin aplicarea automată de rotații, zgomot Gaussian, variații de iluminare solară și blur de autofocus (simulând camera unui smartphone ieftin), modelul și-a menținut performanța, înregistrând o acuratețe remarcabilă de 99.40%.
* **Perturbări Extreme Distructive:** La adăugarea unui zgomot optic masiv și distorsiuni extreme de culoare (simulând o lentilă murdară sau obturată parțial), acuratețea de clasificare a coborât până la 7.40%. Această degradare controlată validează arhitectura decuplată: în caz de eșec vizual total, aplicația redirecționează atenția fermierului către modulul pedoclimatic pentru asistență.

---

## 9. Considerente Etice, Confidențialitate și Sustenabilitate

### Matricea Agronomică de Risc și Impact Tehnic
Predicțiile eronate comise de un algoritm în lumea reală implică pierderi financiare severe. Tabelul de mai jos detaliază abordarea critică a riscurilor din spatele sistemului hibrid:

| Tip Eroare | Scenariu Tehnologic în Câmp | Impact Agronomic Direct | Mecanism de Atenuare Implementat |
| :--- | :--- | :--- | :--- |
| **Fals Negativ (FN)** | Modelul ratează o infecție severă de *Late Blight* (Mană), clasificând frunza bolnavă drept „Sănătoasă”. | **Catastrofal:** Boala se extinde epidemic în întreaga cultură în 48 de ore, generând pierderea totală a recoltei fermierului. | Funcția de cost a rețelei optice optimizează în mod agresiv metrica de **Recall** pe clasele fitopatologice virulente pentru a elimina complet riscul ratărilor. |
| **Fals Pozitiv (FP)** | Algoritmul confundă o arsură mecanică sau solară minoră cu o patologie bacteriană distructivă. | **Moderat/Scăzut:** Fermierul aplică o stropire chimică inutilă, crescând costurile operaționale și poluând local solul. | Integrarea **Hărților de Activare Grad-CAM**. Înainte de a cumpăra substanțe chimice, fermierul verifică vizual dacă rețeaua privește leziunile biologice ale frunzei sau doar reflexia luminii, corectând decizia computerizată. |

* **XAI ca Filtru de Combatere a Bias-ului de Laborator:** Imaginile din PlantVillage sunt realizate pe fundaluri sterile. Suprapunerea hărților termice generate de Grad-CAM acționează ca o barieră de siguranță: fermierul poate valida în timp real dacă AI-ul analizează leziunile reale de pe frunză sau dacă este influențat de reflexiile de pe fundal.
* **Impactul de Mediu și Sustenabilitatea:** Prin optimizarea planurilor de rotație a culturilor în funcție de nutrienți, AgriGuard AI combate fenomenele dăunătoare de supra-fertilizare cu Azot, protejând pânza freatică și reducând emisiile de gaze cu efect de seră din agricultură.
* **Confidențialitatea Datelor:** Procesul de inferență rulează exclusiv in-memory. Platforma nu stochează imaginile încărcate de fermieri și nu salvează datele GPS ale exploatațiilor agricole private, asigurând o confidențialitate completă.

---

## 10. Structura Proiectului și Ghidul de Reproducere

### 10.1 Structura Rădăcină a Directoarelor
Proiectul este structurat modular, respectând bunele practici din industrie și integrând noile fișiere de automatizare, containerizare și regionalizare:

```plaintext
AgriGuard-AI/
│
├── data/
│   ├── raw/
│   │   ├── Crop_recommendation.csv        # Dataset sol (2200 eșantioane)
│   │   └── plantvillage/                  # Dataset imagini structurat pe clase
│   └── processed/
│       └── date_pedoclimatice_moldova.csv # Date simulate specifice cernoziomului
│
├── demo_files/                            # Capturi de ecran ale interfeței și grafice cheie
│
├── frontend/
│   └── app.py                             # Interfața Streamlit cu Filtru OOD și Fuziune
│
├── models/
│   ├── vision_model_rtx_finetuned.pth     # Modelul EfficientNet-B0 PyTorch (99.54%)
│   ├── random_forest_soil_model.pkl       # Modelul Random Forest Final Regularizat (97.60%)
│   ├── soil_scaler.pkl                    # StandardScaler pentru date structurate
│   └── soil_label_encoder.pkl             # LabelEncoder pentru etichete culturi
│
├── notebooks/
│   ├── 01_eda_soil_data.ipynb             # EDA aprofundat, grafice Violin și analiză PCA
│   ├── 02_eda_vision_data.ipynb           # EDA imagini, manual RGB mean/std și scor blur
│   ├── 03_train_vision_model.ipynb        # Antrenare optică refactorizată cu Dataset Wrapper
│   └── 04_train_tabular_model.ipynb       # Benchmarking, GridSearchCV extins și regularizare
│
├── scripts/
│   └── simulate_moldova_data.py           # Script simulator pedoclimatic (Cernoziom)
│
├── Dockerfile                             # Containerizarea Docker pentru producție izolata
├── setup.sh                               # Script bash pentru automatizarea completă a mediului
└── requirements.txt                       # Dependențe optimizate pentru cloud deployment

### 10.2 Ghidul Oficial de Reproducere a Rezultatelor

#### Metoda A: Automatizare Locală prin Script-ul Setup (Recomandat)
Pentru a asigura o reproducere completă și curată (crearea directoarelor, generarea datelor pedoclimatice din Moldova și descărcarea datelor via Kaggle API), deschide un terminal în rădăcina proiectului și rulează:

```bash
# Acordați permisiuni de execuție scriptului bash
chmod +x setup.sh

# Lansați pipeline-ul complet de configurare
./setup.sh

# Lansați interfeța Streamlit gata conectată la modelele regularizate
streamlit run frontend/app.py
```

#### Metoda B: Containerizare Complet Izolată (Docker)
Dacă dorești să rulezi platforma într-un mediu perfect izolat, eliminând complet necesitatea instalării manuale a pachetelor pe sistemul gazdă:

```bash
# Construirea imaginii Docker pe baza rețetelor de sistem optimizate
docker build -t agriguard-ai-pro .

# Lansarea containerului în fundal pe portul nativ
docker run -p 8501:8501 agriguard-ai-pro
```

Apoii, accesează în browser URL-ul local: http://localhost:8501

## 11. Referințe Bibliografice
Hughes, D. & Salathé, M. (2015). An open access image database of plant diseases on crops. arXiv:1511.08060. (Fundamentul bazei de date de laborator PlantVillage).

Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. IEEE ICCV. (Algoritmul utilizat pentru generarea matricilor de atenție vizuală și XAI).

Biroul Național de Statistică al Republicii Moldova (2025). Anuarul Statistic al Republicii Moldova: Capitolul Agricultură și Utilizarea Produselor de Uz Fitosanitar. (Sursa oficială de date pentru calibrarea profilului de simulator local).