import cv2
import numpy as np
import base64


# -------------------------------
# Algoritma seçimi
# -------------------------------
def get_detector(algo_name):
    algo = algo_name.upper()

    try:
        if algo == "SIFT":
            return cv2.SIFT_create()

        elif algo == "SURF":
            # SURF için opencv-contrib-python gereklidir.
            # Patent süresi dolduğu için yeni sürümlerde doğrudan erişilebilir.
            try:
                return cv2.xfeatures2d.SURF_create()
            except AttributeError:
                # Bazı versiyonlarda doğrudan cv2 altında olabilir
                return cv2.SURF_create()

        elif algo == "AKAZE":
            return cv2.AKAZE_create()

        elif algo == "ORB":
            return cv2.ORB_create(nfeatures=4000)

    except Exception as e:
        print(f"Algoritma başlatma hatası ({algo}): {e}")
        return None

    return None


# -------------------------------
# Forgery match bulma (KNN + vektör kümeleme)
# -------------------------------
def find_forgery_matches(kp, desc, algorithm):
    if desc is None or len(desc) < 20:
        return [], None

    # SIFT/SURF için L2, ORB/AKAZE için Hamming mesafesi
    norm = cv2.NORM_L2 if algorithm in ["SIFT", "SURF"] else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)

    # Her nokta için en yakın 2 komşuyu bul (k=2)
    # 1. komşu noktanın kendisidir (mesafe 0), 2. komşu ise kopyası olabilir.
    matches = bf.knnMatch(desc, desc, k=2)

    pairs = []
    vectors = []

    for m, n in matches:
        # m: kendisi, n: en yakın benzeri
        # Lowe's Ratio Test: Eğer en yakın benzeri, diğerlerinden belirgin şekilde daha yakınsa seç.
        if n.distance < 0.7 * m.distance:  # Bu oran (0.7) gürültüyü temizler
            continue  # Benzerlik yeterince güçlü değilse geç

        pt1 = np.array(kp[m.queryIdx].pt)
        pt2 = np.array(kp[n.trainIdx].pt)

        # Öklid mesafesi: Noktalar birbirine çok yakınsa (aynı obje içindeyse) sahtecilik değildir.
        dist = np.linalg.norm(pt1 - pt2)
        if dist > 50:  # En az 50 piksel uzakta olmalı
            vector = pt2 - pt1
            vectors.append(vector)
            pairs.append((pt1, pt2))

    if len(vectors) < 5:
        return [], None

    # Geometrik Tutarlılık Analizi (Vektör Kümeleme)
    # Sahtecilikte tüm kopyalanan noktalar aynı yöne ve aynı uzaklığa taşınır.
    vectors = np.array(vectors)
    median_vec = np.median(vectors, axis=0)  # Ortalama yerine medyan daha güvenlidir

    final_pairs = []
    for i, v in enumerate(vectors):
        # Eğer bir eşleşmenin yönü/uzaklığı genel gidişata uyuyorsa o 'gerçek' sahteciliktir.
        if np.linalg.norm(v - median_vec) < 30:
            final_pairs.append(pairs[i])

    return final_pairs, median_vec


# -------------------------------
# Ana analiz fonksiyonu
# -------------------------------
def run_traditional_analysis(img_cv, algorithm):
    detector = get_detector(algorithm)

    if detector is None:
        return -1, -1, False, None

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kp, desc = detector.detectAndCompute(gray, None)

    if desc is None:
        return 0, 0, False, None

    matches, vec = find_forgery_matches(kp, desc, algorithm.upper())

    # 10'dan fazla tutarlı eşleşme varsa sahtecilik olarak işaretle
    is_forged = len(matches) > 10
    vis_image = draw_matches(img_cv, matches) if len(matches) > 0 else None

    return len(kp), len(matches), is_forged, vis_image


# -------------------------------
# Görselleştirme (Kanıt Çizimi)
# -------------------------------
def draw_matches(img, matches):
    img_copy = img.copy()
    if matches is None or len(matches) == 0:
        return None

    pts1 = []
    pts2 = []

    for pt1, pt2 in matches:
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        pts1.append(pt1)
        pts2.append(pt2)

        # Kanıt çizgileri
        cv2.circle(img_copy, pt1, 4, (0, 0, 255), -1)  # Kaynak
        cv2.circle(img_copy, pt2, 4, (255, 0, 0), -1)  # Hedef
        cv2.line(img_copy, pt1, pt2, (0, 255, 0), 1)  # Bağlantı

    # Kaynak ve Hedef bölgeleri kare içine al
    if len(pts1) > 0:
        for pts, color in [(pts1, (255, 0, 255)), (pts2, (0, 255, 255))]:
            x, y, w, h = cv2.boundingRect(np.array(pts))
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 3)

    _, buffer = cv2.imencode('.jpg', img_copy)
    return base64.b64encode(buffer).decode('utf-8')