import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [algo, setAlgo] = useState('SIFT');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
    setResult(null);

    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Lütfen önce bir resim seçin!");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('algorithm', algo);

    try {
      const response = await fetch('http://localhost:8000/analyze/', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Bir hata oluştu.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getInterpretation = () => {
    if (!result) return "";

    const t = result.traditional_results?.detected;
    const a = result.ai_results?.detected;

    if (!t && !a) {
      return "Görüntüde belirgin bir sahtecilik tespit edilmedi.";
    }

    if (!t && a) {
      return "AI modeli görüntüde anomali tespit etti. Bu durum deepfake (yüz manipülasyonu) ihtimaline işaret edebilir.";
    }

    if (t && !a) {
      return "Görüntüde tekrar eden bölgeler bulundu. Bu durum copy-move (kopyala-yapıştır) sahteciliğe işaret edebilir.";
    }

    if (t && a) {
      return "Hem yapısal tekrarlar hem de AI anomalileri tespit edildi. Görüntü yüksek olasılıkla manipüle edilmiş.";
    }
  };

  return (
    <div className="container">

      <header>
        <h1>🔍 Görüntü Sahteciliği Tespiti</h1>
        <p className="subtitle">Copy-Move & AI Destekli Analiz Sistemi</p>
      </header>

      <main className="grid">

        {/* SOL PANEL */}
        <div className="card">

          <h3>📤 Görüntü Yükle</h3>

          <label className="upload-box">
            {preview ? "🔁 Başka bir görsel seç" : "📂 Görüntü seçmek için tıkla"}
            <input
              type="file"
              onChange={handleFileChange}
              accept="image/*"
              hidden
            />
          </label>

          {preview && (
            <img src={preview} alt="preview" className="preview" />
          )}

          <label>Algoritma Seç</label>
          <select value={algo} onChange={(e) => setAlgo(e.target.value)}>
            <option value="SIFT">SIFT</option>
            <option value="SURF">SURF</option>
            <option value="AKAZE">AKAZE</option>
            <option value="ORB">ORB</option>
          </select>

          <button onClick={handleUpload} disabled={loading}>
            {loading ? "Analiz ediliyor..." : "Analizi Başlat"}
          </button>


          <div className="info-box">
            <strong>ℹ️ Bilgi:</strong>
            <p>
              Sistem, aynı görüntü içindeki tekrar eden bölgeleri ve AI skorunu analiz eder. Bu analize göre görselin temiz veya şüpheli olduğu sonucuna ulaşılır. Eşleşen noktalar varsa görselde işaretlenerek gösterilir.
            </p>
          </div>

        </div>

        {/* SAĞ PANEL */}
        <div className="card">

          <h3>📊 Analiz Sonucu</h3>

          {loading && <div className="loader"></div>}

          {error && <div className="error">{error}</div>}

          {result && (
            <>
              <div className={`badge result-anim ${result.final_status === 'Şüpheli' ? 'danger' : 'success'}`}>
                {result.final_status}
              </div>

              <p><strong>Algoritma:</strong> {result.selected_algorithm}</p>

              <div className="stats">
                <div>
                  <span>Keypoints: </span>
                  <b>{result.traditional_results.keypoints_found}</b>
                </div>

                <div>
                  <span>Matches: </span>
                  <b>{result.traditional_results.matches_found}</b>
                </div>

                <div>
                  <span>AI Skoru: </span>
                  <b>{result.ai_results.cnn_lstm_score}</b>
                </div>
              </div>

              <div className="interpretation">
                <strong>📌 Yorum:</strong>
                <p>{getInterpretation()}</p>
              </div>

              {result?.traditional_results?.visualization && (
                <>
                  <h4>🧠 Tespit Edilen Bölgeler</h4>
                  <img
                    src={`data:image/jpeg;base64,${result.traditional_results.visualization}`}
                    alt="result"
                    className="result-img"
                  />
                </>
              )}
            </>
          )}

        </div>

      </main>
    </div>
  );
}

export default App;