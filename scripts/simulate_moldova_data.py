import pandas as pd
import numpy as np
import os

def genereaza_date_sol_moldova(mostre_per_clasa=400):
    """
    Simulează profile chimice de sol și condiții climatice specifice Republicii Moldova.
    Cernoziomurile locale prezintă valori natural ridicate de Potasiu (K), Fosfor (P) moderat,
    pH slab alcalin/neutru și un regim de precipitații temperat-continental (350-600 mm).
    Targetează exact culturile native prezente în matricea din app.py.
    """
    np.random.seed(42)
    date_sol = []
    
    # Parametri ajustați conform realității agronomice locale și claselor din app.py
    profile_climatice = {
        'maize':       {'N': (80, 130), 'P': (40, 65),  'K': (150, 200), 'temp': (20.0, 26.0), 'hum': (55.0, 70.0), 'ph': (6.2, 7.3), 'rain': (450.0, 580.0)},
        'grapes':      {'N': (45, 75),  'P': (30, 50),  'K': (160, 240), 'temp': (19.0, 27.0), 'hum': (45.0, 60.0), 'ph': (6.5, 7.5), 'rain': (360.0, 480.0)},
        'apple':       {'N': (50, 85),  'P': (35, 55),  'K': (140, 210), 'temp': (18.0, 24.0), 'hum': (55.0, 70.0), 'ph': (6.0, 7.0), 'rain': (500.0, 620.0)},
        'watermelon':  {'N': (60, 90),  'P': (35, 55),  'K': (130, 175), 'temp': (23.0, 30.0), 'hum': (45.0, 60.0), 'ph': (6.1, 7.1), 'rain': (350.0, 450.0)},
        'lentil':      {'N': (15, 35),  'P': (35, 50),  'K': (110, 145), 'temp': (18.5, 24.5), 'hum': (45.0, 60.0), 'ph': (6.3, 7.4), 'rain': (380.0, 480.0)},
        'chickpea':    {'N': (20, 40),  'P': (40, 55),  'K': (120, 150), 'temp': (19.0, 26.0), 'hum': (40.0, 55.0), 'ph': (6.4, 7.6), 'rain': (350.0, 440.0)},
        'blackgram':   {'N': (20, 45),  'P': (35, 55),  'K': (115, 145), 'temp': (22.0, 27.5), 'hum': (50.0, 65.0), 'ph': (6.2, 7.2), 'rain': (400.0, 500.0)},
        'kidneybeans': {'N': (15, 35),  'P': (40, 60),  'K': (125, 155), 'temp': (18.0, 24.0), 'hum': (50.0, 65.0), 'ph': (6.0, 7.0), 'rain': (420.0, 520.0)}
    }
    
    for cultura, param in profile_climatice.items():
        for _ in range(mostre_per_clasa):
            n = int(np.random.uniform(param['N'][0], param['N'][1]))
            p = int(np.random.uniform(param['P'][0], param['P'][1]))
            k = int(np.random.uniform(param['K'][0], param['K'][1]))
            t = round(np.random.uniform(param['temp'][0], param['temp'][1]), 4)
            h = round(np.random.uniform(param['hum'][0], param['hum'][1]), 4)
            ph = round(np.random.uniform(param['ph'][0], param['ph'][1]), 4)
            r = round(np.random.uniform(param['rain'][0], param['rain'][1]), 4)
            
            date_sol.append([n, p, k, t, h, ph, r, cultura])
            
    df_moldova = pd.DataFrame(date_sol, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    os.makedirs('data/processed', exist_ok=True)
    df_moldova.to_csv('data/processed/date_pedoclimatice_moldova.csv', index=False)
    print(f"✅ Succes: Au fost generate {len(df_moldova)} rânduri regionale în data/processed/date_pedoclimatice_moldova.csv")

if __name__ == "__main__":
    genereaza_date_sol_moldova(mostre_per_clasa=450)