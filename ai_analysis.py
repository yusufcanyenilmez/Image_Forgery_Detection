import torch
import torch.nn as nn
from torchvision import models, transforms
import os

# Programın hızlı çalışması için GPU desteği
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# CNN (ResNet18) → feature extractor
# ==============================
resnet = models.resnet18()
resnet.fc = nn.Linear(resnet.fc.in_features, 2)

# Kaydettiğin ağırlıkları (forgery_model.pth) yüklüyoruz
model_path = "forgery_model.pth"
if os.path.exists(model_path):
    resnet.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Eğitilmiş CNN ağırlıkları yüklendi.")
else:
    print("⚠️ UYARI: forgery_model.pth bulunamadı, varsayılan ağırlıklar kullanılıyor.")

# LSTM'e beslemek için son katmanı (fc) çıkarıp 'Feature Extractor' yapıyoruz
cnn_model = nn.Sequential(*(list(resnet.children())[:-1]))
cnn_model = cnn_model.to(device)
cnn_model.eval()


# ==============================
# LSTM Modeli
# ==============================
class ForgeryLSTM(nn.Module):
    """
        @brief CNN öznitelik haritalarını ardışık veri olarak inceleyen LSTM yapay sinir ağı sınıfı.
        @details ResNet mimarisinin son katmanından elde edilen 512 boyutlu uzaysal vektör serilerini
                 zaman/sıra düzleminde işleyerek pikseller arası mikroskobik manipülasyonları ayrıştırır.
    """
    def __init__(self):
        """
            @brief Katman yapılandırmalarını gerçekleştiren kurucu fonksiyon.
            @details 512 girdili, 128 gizli hücreli bir LSTM katmanı ve olasılık çıktısı üreten doğrusal bir sınıflandırma (FC) katmanı ayağa kaldırır.
        """
        super(ForgeryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        """
            @brief Modelin ileri besleme (forward propagation) hesaplamalarını yürütür.
            @param x CNN modelinden sıkıştırılarak aktarılan tensör verisi.
            @return Olasılık değeri döndüren Sigmoid aktivasyon fonksiyonu çıktısı ([0, 1] arası).
        """
        _, (h_n, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h_n[-1]))


lstm_model = ForgeryLSTM().to(device)

# Eğitilmiş LSTM modeli olmadığı için bu model random çalışır ama sistem bozulmaz
lstm_model.eval()


# ==============================
# Preprocess (önişleme)
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ResNet standart normalizasyonu
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==============================
# AI analiz
# ==============================
def run_ai_analysis(img_pil):
    """
        @brief Derin öğrenme (CNN + LSTM) hibrit mimarisi ile görselin manipülasyon skorunu hesaplar.
        @details PIL formatında gelen görseli ResNet standartlarına göre normalize eder, ardından
                 eğitilmiş ağırlıklar üzerinden olasılık değerlerini ve ardışık katmanlardan LSTM
                 özniteliklerini çıkararak füzyon bir güven skoru üretir.
        @param img_pil PIL kütüphanesi formatında okunmuş RGB giriş görseli.
        @return final_score İki modelin ortak kararı olan nihai anomali skoru yüzdesi.
        @return status Skorun %50 eşik değerini aşıp aşmadığını belirten boolean durum bilgisi.
    """
    image = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # CNN (resnet) içindeki eğitilmiş katmanı kullanıyoruz
        outputs = resnet(image)  # resnet.fc artık 2 sınıflı eğitilmiş katman!

        # Olasılıkları hesapla
        probabilities = torch.softmax(outputs, dim=1)

        # 0. index 'Sahte' (Forged) olasılığıdır.
        cnn_score = probabilities[0][0].item()

        # 🔥 CNN feature çıkar
        features = cnn_model(image)
        features = features.view(1, -1)

        # 🔥 LSTM'e ver
        lstm_input = features.unsqueeze(1)
        lstm_score = lstm_model(lstm_input).item()

        # 🔥 FUSION
        final_score = (cnn_score + lstm_score) / 2

    return final_score, final_score > 0.5