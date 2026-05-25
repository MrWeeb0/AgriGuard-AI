FROM python:3.9-slim

WORKDIR /app

# Instalare dependențe de sistem esențiale pentru OpenCV, grafică și dezarhivare
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiere requirements și instalare cache-free a pachetelor Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copierea întregii structuri a proiectului în container
COPY . .

# Expunerea portului implicit pentru aplicația Streamlit
EXPOSE 8501

# Comandă de pornire a platformei hibride
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]