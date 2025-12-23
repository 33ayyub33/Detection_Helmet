import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from streamlit_webrtc import webrtc_streamer
import av

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
# LOAD MODEL YOLO
# =========================
@st.cache_resource
def load_model():
    return YOLO("best_2.pt")  # pastikan file ada

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

        results = model(frame, conf=0.5)

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
# MODE 2 : UPLOAD VIDEO + DOWNLOAD HASIL
# ======================================================
elif mode == "Upload Video":
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        # =========================
        # SIMPAN VIDEO INPUT
        # =========================
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        # =========================
        # INFO VIDEO
        # =========================
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25  # fallback aman

        # =========================
        # VIDEO OUTPUT
        # =========================
        output_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ).name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_placeholder = st.empty()
        st.info("▶️ Video sedang diproses...")

        # =========================
        # PROSES FRAME
        # =========================
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.5)
            ada_orang = ada_helm = ada_tanpa_helm = False

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue

                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
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

            # =========================
            # STATUS
            # =========================
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
                frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, warna, 3
            )

            # =========================
            # TULIS KE VIDEO OUTPUT
            # =========================
            out.write(frame)
            frame_placeholder.image(frame, channels="BGR")

        # =========================
        # RELEASE
        # =========================
        cap.release()
        out.release()
        os.remove(video_path)

        st.success("✅ Video selesai diproses")

        # =========================
        # DOWNLOAD BUTTON
        # =========================
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Video Hasil Deteksi",
                data=f,
                file_name="hasil_deteksi_helm.mp4",
                mime="video/mp4"
            )

# ======================================================
# MODE 3 : WEBCAM REALTIME (WEBRTC - DEPLOY SAFE)
# ======================================================
else:
    st.subheader("🎥 Webcam Realtime (WebRTC)")
    st.info("Izinkan akses kamera di browser")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=0.5)

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

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
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

        cv2.putText(img, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, warna, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

   webrtc_streamer(
    key="helm-webrtc",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    },
    media_stream_constraints={
        "video": {"width": 640, "height": 480},
        "audio": False
    }
)
