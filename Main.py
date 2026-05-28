from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from PIL import Image

# Diğer dosyalardan fonksiyonları içe aktarıyoruz
from traditional_analysis import run_traditional_analysis
from ai_analysis import run_ai_analysis

##
# @mainpage Görüntü Sahteciliği Tespit Sistemi API Dökümantasyonu
# @section intro_sec Giriş
# Bu proje, dijital görüntüler üzerindeki manipülasyonları ve Copy-Move sahteciliklerini
# geleneksel bilgisayarlı görü ve derin öğrenme hibrit mimarisiyle tespit eder.
#

app = FastAPI(title="Image Forgery Detection API", version="1.0.0")

# CORS Ayarları: Frontend erişimine izin veriyoruz.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# POST endpointi: Kullanıcı görsel ve algoritma gönderiyor.
# -------------------------------
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...), algorithm: str = Form("SIFT")):
    # Sistemin desteklediği algoritmalar
    allowed_algorithms = ["SIFT", "SURF", "AKAZE", "ORB"]

    # Algoritma kontrolü: Sadece belirlenen listedekileri kabul et.
    if algorithm.upper() not in allowed_algorithms:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz algoritma! Şunları kullanabilirsiniz: {allowed_algorithms}"
        )

    # Dosya tipi kontrolü
    if file.content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı")

    # User Story-1: Dosya Okuma
    contents = await file.read()

    # Dosya boyutu kontrolü (5MB)
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Dosya çok büyük (max 5MB)")
    try:
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    except:
        raise HTTPException(status_code=400, detail="Geçersiz görsel dosyası")

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # User Story-2: Geleneksel Analiz (Modülden çağırılıyor)
    kp_count, match_count, traditional_detected, vis_image = run_traditional_analysis(img_cv, algorithm)

    if kp_count == -1:
        raise HTTPException(status_code=501, detail=f"{algorithm} algoritması bu sistemde desteklenmiyor.")

    # User Story-3: AI Analizi (Modülden çağırılıyor)
    ai_score, ai_detected = run_ai_analysis(img_pil)

    return {
        "filename": file.filename,
        "selected_algorithm": algorithm,
        "traditional_results": {
            "keypoints_found": kp_count,
            "matches_found": match_count,
            "detected": traditional_detected,
            "visualization": vis_image
        },
        "ai_results": {
            "cnn_lstm_score": f"%{ai_score * 100:.2f}",
            "detected": ai_detected
        },
        "final_status": "Şüpheli" if (traditional_detected or ai_detected) else "Temiz"
    }



# -------------------------------
# Dosya direkt çalıştırılırsa API başlat.
# -------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
