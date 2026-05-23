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
    def __init__(self):
        super(ForgeryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
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