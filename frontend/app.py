import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import cv2

st.set_page_config(page_title="AgriGuard AI", page_icon="🌱", layout="wide")

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
    num_classes = 16
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load('../models/vision_model_rtx_finetuned.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_tabular_model():
    model_xgb = joblib.load('../models/xgboost_soil_model.pkl')
    scaler = joblib.load('../models/soil_scaler.pkl')
    encoder = joblib.load('../models/soil_label_encoder.pkl')
    return model_xgb, scaler, encoder

vision_model = load_vision_model()
xgb_model, scaler, label_encoder = load_tabular_model()

CLASE_BOLI = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy', 'Necunoscut_Fantoma'
]

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
        b, c, h, w = x.size()
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
    superimposed_img = np.uint8(superimposed_img)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(superimposed_img)

cam_engine = GradCAM(vision_model, vision_model.features[-1])

st.title("AgriGuard AI - Asistent Inteligent pentru Fermieri")
st.markdown("Sistem Hibrid cu **Explicabilitate Vizuală (XAI)**. Încărcați datele pentru diagnostic.")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910822.png", width=80) 
    st.title("Parametri Sol & Mediu")
    st.markdown("Ajustează senzorii virtuali:")
    
    n_val = st.slider("Nitrogen (N)", 0, 150, 40, help="Nivelul de Azot din sol")
    p_val = st.slider("Fosfor (P)", 0, 150, 40)
    k_val = st.slider("Potasiu (K)", 0, 200, 40)
    temp_val = st.slider("Temperatură (°C)", 0.0, 50.0, 25.0)
    hum_val = st.slider("Umiditate (%)", 0.0, 100.0, 70.0)
    ph_val = st.slider("pH Sol", 0.0, 14.0, 6.5)
    rain_val = st.slider("Precipitații (mm)", 0.0, 300.0, 250.0)

col_header, col_img = st.columns([2, 1])
with col_header:
    st.markdown("### Pasul 1: Încarcă imaginea plantei")
    fisier_incarcat = st.file_uploader("", type=["jpg", "png", "jpeg"], help="Trage imaginea aici sau dă click")

with col_img:
    if fisier_incarcat is not None:
        imagine = Image.open(fisier_incarcat).convert('RGB')
        st.image(imagine, caption='Imagine Procesată', width=200)

st.markdown("---")

if st.button("Scanează și Generează Diagnostic", use_container_width=True):
    if fisier_incarcat is None:
        st.error("Sistemul necesită o imagine pentru a iniția scanarea vizuală.")
    else:
        with st.spinner('Sistemul Neural procesează datele...'):
            
            img_tensor = transformare_imagine(imagine).unsqueeze(0)
            img_tensor.requires_grad = True
            
            output_vision = vision_model(img_tensor)
            probabilitati_vision = torch.nn.functional.softmax(output_vision[0], dim=0)
            index_boala = torch.argmax(probabilitati_vision).item()
            
            harta_termica = cam_engine.genereaza_harta(img_tensor, index_boala)
            imagine_explicata = aplica_harta_peste_imagine(imagine, harta_termica)
            
            nume_boala = CLASE_BOLI[index_boala].replace("___", " - ").replace("_", " ")
            incredere_boala = probabilitati_vision[index_boala].item() * 100
                
            date_sol = pd.DataFrame([[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]], 
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            date_sol_scaled = scaler.transform(date_sol)
            probabilitati_sol = xgb_model.predict_proba(date_sol_scaled)
            index_cultura = np.argmax(probabilitati_sol[0])
            nume_cultura = label_encoder.inverse_transform([index_cultura])[0]
            incredere_cultura = probabilitati_sol[0][index_cultura] * 100

            st.success("Analiză completă finalizată.")
            
            tab1, tab2 = st.tabs(["Diagnostic Oficial", "XAI (Explicabilitate Neurală)"])
            
            with tab1:
                st.markdown("#### Rezultate Fuziune Modele")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric(label="Stare Frunza Detectata", value=nume_boala, delta=f"{incredere_boala:.2f}% Siguranta")
                    if "healthy" in nume_boala.lower():
                        st.info("Frunza pare perfect sănătoasă. Continuați planul de irigare standard.")
                    else:
                        st.warning("A fost detectată o infecție. Se recomandă izolare și tratament fungic/bacterian.")
                        
                with col_res2:
                    st.metric(label="Recomandare Cultura (Agro-Meteo)", value=nume_cultura.capitalize(), delta=f"{incredere_cultura:.2f}% Potrivire")
                    st.info(f"Parametrii actuali (N:{n_val}, Umiditate:{hum_val}%) sunt optimi pentru {nume_cultura}.")
            
            with tab2:
                st.markdown("#### Cum a gandit Inteligenta Artificiala?")
                st.markdown("Sistemul **Grad-CAM** evidentiaza cu rosu zonele celulare care au declansat alerta de boala.")
                col_xai1, col_xai2, col_xai3 = st.columns([1, 2, 1])
                with col_xai2:
                    st.image(imagine_explicata, caption='Radar AI (Zone Rosii = Focar Infectie)', use_column_width=True)