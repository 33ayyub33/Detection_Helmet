import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Helm Pengendara",
    layout="wide"
)

st.title("🚦 Deteksi Helm Pengendara (YOLO)")
st.write("Dataset: **Ga pake Helm | Helm | orang**")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # pastikan best.pt satu folder dengan app.py

model = load_model()

# =========================
# PILIH MODE
# =========================
mode = st.sidebar.selectbox(
    "Pilih Mode",
    ["Upload Gambar", "Upload Video", "Webcam Realtime"]
)

# ======================================================
# MODE 1 : UPLOAD GAMBAR
# ======================================================
if mode == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)

        results = model(frame)

        ada_orang = ada_helm = ada_tanpa_helm = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "Helm":
                    color = (0, 255, 0)
                    ada_helm = True
                elif label == "Ga pake Helm":
                    color = (255, 0, 0)
                    ada_tanpa_helm = True
                elif label == "orang":
                    color = (0, 0, 255)
                    ada_orang = True
                else:
                    color = (255, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        if ada_orang and ada_helm:
            status = "✅ MEMAKAI HELM"
        elif ada_orang and ada_tanpa_helm:
            status = "❌ TIDAK MEMAKAI HELM"
        elif ada_orang:
            status = "⚠️ STATUS HELM TIDAK JELAS"
        else:
            status = "⚠️ TIDAK ADA PENGENDARA"

        st.image(frame, caption=status, use_container_width=True)
        st.subheader(status)

# ======================================================
# MODE 2 : UPLOAD VIDEO
# ======================================================
elif mode == "Upload Video":
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        # simpan video sementara
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        frame_placeholder = st.empty()

        st.info("▶️ Video sedang diproses frame per frame...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            ada_orang = ada_helm = ada_tanpa_helm = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if label == "Helm":
                        color = (0, 255, 0)
                        ada_helm = True
                    elif label == "Ga pake Helm":
                        color = (0, 0, 255)
                        ada_tanpa_helm = True
                    elif label == "orang":
                        color = (255, 0, 0)
                        ada_orang = True
                    else:
                        color = (255, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            if ada_orang and ada_helm:
                status = "MEMAKAI HELM"
                warna = (0, 255, 0)
            elif ada_orang and ada_tanpa_helm:
                status = "TIDAK MEMAKAI HELM"
                warna = (0, 0, 255)
            elif ada_orang:
                status = "STATUS HELM TIDAK JELAS"
                warna = (0, 255, 255)
            else:
                status = "TIDAK ADA PENGENDARA"
                warna = (255, 255, 0)

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                warna,
                3
            )

            frame_placeholder.image(frame, channels="BGR")

        cap.release()
        os.remove(video_path)
        st.success("✅ Video selesai diproses")

# ======================================================
# MODE 3 : WEBCAM REALTIME
# ======================================================
else:
    st.warning("Klik **Start Webcam** lalu izinkan akses kamera di browser")

    start = st.button("▶️ Start Webcam")
    stop = st.button("⏹ Stop Webcam")

    frame_placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Webcam tidak dapat dibuka")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)

                ada_orang = ada_helm = ada_tanpa_helm = False

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        conf = float(box.conf[0])

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if label == "Helm":
                            color = (0, 255, 0)
                            ada_helm = True
                        elif label == "Ga pake Helm":
                            color = (0, 0, 255)
                            ada_tanpa_helm = True
                        elif label == "orang":
                            color = (255, 0, 0)
                            ada_orang = True
                        else:
                            color = (255, 255, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            f"{label} ({conf:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                if ada_orang and ada_helm:
                    status = "MEMAKAI HELM"
                    warna = (0, 255, 0)
                elif ada_orang and ada_tanpa_helm:
                    status = "TIDAK MEMAKAI HELM"
                    warna = (0, 0, 255)
                elif ada_orang:
                    status = "STATUS HELM TIDAK JELAS"
                    warna = (0, 255, 255)
                else:
                    status = "TIDAK ADA PENGENDARA"
                    warna = (255, 255, 0)

                cv2.putText(
                    frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    warna,
                    3
                )

                frame_placeholder.image(frame, channels="BGR")

                if stop:
                    cap.release()
                    break
