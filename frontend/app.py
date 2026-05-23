import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import joblib
import pandas as pd
import numpy as np
import cv2

st.set_page_config(page_title="AgriGuard AI Pro", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        border: 2px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vision_model():
    num_classes = 15
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load('models/vision_model_rtx_finetuned.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_tabular_model():
    model_rf = joblib.load('models/random_forest_soil_model.pkl')
    scaler = joblib.load('models/soil_scaler.pkl')
    encoder = joblib.load('models/soil_label_encoder.pkl')
    return model_rf, scaler, encoder

vision_model = load_vision_model()
rf_model, scaler, label_encoder = load_tabular_model()

CLASE_BOLI = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

MATRICE_COMPATIBILITATE_ROTATIE = {
    'Solanaceae': {
        'optimizat': ['chickpea', 'lentil', 'kidneybeans', 'pigeonpeas', 'mungbean', 'blackgram'],
        'tolerat': ['maize', 'watermelon', 'muskmelon', 'banana', 'mango', 'grapes', 'apple', 'orange', 'papaya'],
        'anomalie_critica': ['rice', 'jute', 'coconut', 'coffee', 'cotton']
    },
    'Cucurbitaceae': {
        'optimizat': ['maize', 'beans'],
        'tolerat': ['rice'],
        'anomalie_critica': ['groundnut']
    }
}

def obtine_familie_botanica(nume_clasa):
    clasa_lower = nume_clasa.lower()
    if 'pepper' in clasa_lower or 'potato' in clasa_lower or 'tomato' in clasa_lower:
        return 'Solanaceae'
    elif 'cucumber' in clasa_lower or 'melon' in clasa_lower:
        return 'Cucurbitaceae'
    return 'Unknown'

transformare_imagine = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def genereaza_harta(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward(retain_graph=True)
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w_val in enumerate(weights):
            cam += w_val * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

def aplica_harta_peste_imagine(img_pil, heatmap):
    img_cv = np.array(img_pil.resize((224, 224)))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    heatmap_cv = np.uint8(255 * heatmap)
    heatmap_cv = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)
    superimposed_img = heatmap_cv * 0.4 + img_cv * 0.6
    return Image.fromarray(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))

cam_engine = GradCAM(vision_model, vision_model.features[-1])

st.title("🌱 AgriGuard AI Pro — Sistem Multimodal de Diagnostic și Asistență Agronomică")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910822.png", width=80) 
    st.title("Senzori Pedoclimatici")
    n_val = st.slider("Nitrogen (N)", 0, 150, 40)
    p_val = st.slider("Fosfor (P)", 0, 150, 40)
    k_val = st.slider("Potasiu (K)", 0, 200, 40)
    temp_val = st.slider("Temperatură (°C)", 0.0, 50.0, 25.0)
    hum_val = st.slider("Umiditate (%)", 0.0, 100.0, 70.0)
    ph_val = st.slider("pH Sol", 0.0, 14.0, 6.5)
    rain_val = st.slider("Precipitații (mm)", 0.0, 300.0, 120.0)

col_header, col_img = st.columns([2, 1])
with col_header:
    st.markdown("### Monitorizare Foliară")
    fisier_incarcat = st.file_uploader("Încărcați imaginea macro a frunzei afectate:", type=["jpg", "png", "jpeg"])

with col_img:
    if fisier_incarcat is not None:
        imagine = Image.open(fisier_incarcat).convert('RGB')
        st.image(imagine, caption='Cadru recepționat', width=180)

st.markdown("---")

if st.button("Execută Analiza Multimodală Hibridă", use_container_width=True):
    with st.spinner('Se rulează inferența ierarhică și verificarea consistenței logice...'):
        executa_sol = False
        executa_viziune = False
        PRAG_SIGURANTA_OOD = 0.75
        
        if fisier_incarcat is not None:
            img_tensor = transformare_imagine(imagine).unsqueeze(0)
            img_tensor.requires_grad = True
            
            output_vision = vision_model(img_tensor)
            probabilitati_vision = torch.nn.functional.softmax(output_vision[0], dim=0)
            incredere_boala, index_boala = torch.max(probabilitati_vision, dim=0)
            incredere_boala = incredere_boala.item()
            index_boala = index_boala.item()
            
            if incredere_boala >= PRAG_SIGURANTA_OOD:
                executa_viziune = True
                executa_sol = True
            else:
                st.warning("⚠️ Filtru Out-of-Distribution (OOD) Activat: Imaginea încărcată nu prezintă markeri fitosanitari recognoscibili sau aparține unui cadru non-vegetal. Modulul de viziune este dezactivat temporar conform protocolului Service Degradation, dar rulăm analiza completă de sol!")
                executa_sol = True
        else:
            st.info("ℹ️ Nu a fost încărcată o imagine foliară. Conform protocolului Service Degradation, modulul optic este suspendat, rulându-se exclusiv analiza predictivă a solului.")
            executa_sol = True
            
        if executa_sol:
            date_sol = pd.DataFrame([[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]], 
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            date_sol_scaled = scaler.transform(date_sol)
            probabilitati_sol = rf_model.predict_proba(date_sol_scaled)
            index_cultura = np.argmax(probabilitati_sol[0])
            nume_cultura = label_encoder.inverse_transform([index_cultura])[0]
            incredere_cultura = probabilitati_sol[0][index_cultura] * 100
            
            if executa_viziune:
                harta_termica = cam_engine.genereaza_harta(img_tensor, index_boala)
                imagine_explicata = aplica_harta_peste_imagine(imagine, harta_termica)
                nume_boala = CLASE_BOLI[index_boala].replace("___", " - ").replace("_", " ")
                familie_botanica = obtine_familie_botanica(CLASE_BOLI[index_boala])
                
                st.success("Fuziune ierarhică multimodală finalizată cu succes.")
                tab1, tab2, tab3 = st.tabs(["Raport Agronomic Integrat", "Explicabilitate Vizuală (Grad-CAM)", "Fuziune și Validare Backend"])
                
                with tab1:
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric(label="Patologie Detectată (Viziune)", value=nume_boala, delta=f"{incredere_boala*100:.2f}% Confidențialitate")
                    with col_res2:
                        st.metric(label="Plan de Asolament Recomandat (Sol)", value=nume_cultura.capitalize(), delta=f"{incredere_cultura:.2f}% Stabilitate")
                
                with tab2:
                    st.markdown("#### Validare Mapare Localizată (XAI)")
                    col_xai1, col_xai2, col_xai3 = st.columns([1, 2, 1])
                    with col_xai2:
                        st.image(imagine_explicata, caption='Zonele de activare neuronală asociate patologiei', use_container_width=True)
                
                with tab3:
                    st.markdown("#### Controlul Consistenței Logice Multimodale")
                    if familie_botanica in MATRICE_COMPATIBILITATE_ROTATIE:
                        if nume_cultura in MATRICE_COMPATIBILITATE_ROTATIE[familie_botanica]['anomalie_critica']:
                            st.markdown(f"""
                            <div style="background-color:#ffebee; padding:15px; border-left:6px solid #e53935; border-radius:4px;">
                                <h4 style="color:#c62828; margin:0;">⚠️ Alertă de Neconcordanță Logică Agronomică</h4>
                                <p style="color:#b71c1c; margin:5px 0 0 0;">
                                    <b>Conflict în sistemul de fuziune:</b> Parametrii pedoclimatici introduși sunt specifici unei culturi cu cerințe hidrice masive sau tropicale (<b>{nume_cultura.capitalize()}</b>), 
                                    ceea ce contrazice biologic prezența unei culturi active din familia <b>{familie_botanica}</b> identificată prin analiză foliară (<b>{nume_boala.split(' ')[0]}</b>). Solurile saturate sau inundate declanșează asfixierea radiculară a tomatelor/cartofilor. Verificați inputurile senzorilor.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif nume_cultura in MATRICE_COMPATIBILITATE_ROTATIE[familie_botanica]['optimizat']:
                            st.markdown(f"""
                            <div style="background-color:#e8f5e9; padding:15px; border-left:6px solid #43a047; border-radius:4px;">
                                <h4 style="color:#2e7d32; margin:0;">✅ Consistență Eco-Agronomică Optimă</h4>
                                <p style="color:#1b5e20; margin:5px 0 0 0;">
                                    <b>Fuziune Validată:</b> Cultura curentă din familia <b>{familie_botanica}</b> prezintă patologii foliare, însă modelul de sol recomandă corect o rotație cu o leguminoasă fixatoare avansată de azot (<b>{nume_cultura.capitalize()}</b>). 
                                    Acest plan de asolament va rupe ciclul biologic al fitopatogenilor din sol și va reface stocul nativ de Nitrogen fără fertilizare chimică excesivă.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info(f"Fuziune neutră: Rezultatele pentru familia {familie_botanica} sunt consistente din punct de vedere ecologic. Nu s-au detectat anomalii sau corelații ideale de asolament.")
                    else:
                        st.info("Fuziune neutră: Familia botanică detectată nu are reguli stricte de asolament definite.")
            else:
                st.success("Analiză pedoclimatică finalizată.")
                tab1 = st.tabs(["Plan de Asolament Optimizat"])[0]
                with tab1:
                    st.markdown("#### Rezultate Analiză Independentă de Sol")
                    st.metric(label="Plan de Asolament Recomandat (Sol)", value=nume_cultura.capitalize(), delta=f"{incredere_cultura:.2f}% Stabilitate")
                    st.info(f"Parametrii chimici actuali indică un mediu optim pentru cultivarea speciei {nume_cultura.capitalize()}. Se recomandă monitorizarea echilibrului de macro-nutrienți.")