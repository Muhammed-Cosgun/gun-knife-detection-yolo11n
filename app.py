from __future__ import annotations

import base64
import os
import shutil
import subprocess
import time
from html import escape
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO


MODEL_REPO_ID = "cosgun99/gun-knife-yolo11n"
MODEL_FILENAME = "best.pt"
OUTPUT_ROOT = Path("runs") / "streamlit_outputs"
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
CLASS_LABELS = {"gun": "Silah", "knife": "Bicak"}

st.set_page_config(page_title="Silah ve Bicak Tespiti", layout="wide")


@st.cache_resource
def load_model() -> YOLO:
    token = os.environ.get("HF_TOKEN")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        token=token,
    )
    return YOLO(model_path)


def display_class_name(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name)


def get_detection_rows(result) -> list[dict[str, str | float]]:
    rows = []
    if result.boxes is None:
        return rows

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        rows.append(
            {
                "class": result.names[class_id],
                "confidence": round(confidence, 4),
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
            }
        )
    return rows


def render_css() -> None:
    st.markdown(
        """
        <style>
            .model-badge {
                border: 1px solid rgba(148, 163, 184, 0.26);
                border-radius: 8px;
                padding: 12px 14px;
                background: rgba(15, 23, 42, 0.42);
                margin-bottom: 14px;
            }
            .model-badge-label {
                color: #94a3b8;
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 4px;
            }
            .model-badge-value {
                color: #f8fafc;
                font-size: 1.05rem;
                font-weight: 800;
            }
            .det-panel {
                border: 1px solid rgba(148, 163, 184, 0.28);
                border-radius: 8px;
                background: rgba(15, 23, 42, 0.36);
                margin-top: 12px;
                overflow: hidden;
            }
            .det-panel-title {
                padding: 14px 16px;
                font-size: 0.95rem;
                font-weight: 700;
                color: #e5e7eb;
                border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            }
            .det-table {
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                font-size: 0.92rem;
            }
            .det-table th {
                color: #93c5fd;
                font-size: 0.76rem;
                font-weight: 800;
                text-transform: uppercase;
                padding: 12px 16px;
                text-align: right;
                border-bottom: 1px solid rgba(148, 163, 184, 0.22);
            }
            .det-table td {
                color: #f8fafc;
                padding: 14px 16px;
                text-align: right;
                border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            }
            .det-table th:nth-child(1), .det-table th:nth-child(2),
            .det-table td:nth-child(1), .det-table td:nth-child(2) {
                text-align: left;
            }
            .det-table th:nth-child(1), .det-table td:nth-child(1) { width: 72px; }
            .det-table th:nth-child(2), .det-table td:nth-child(2) { width: 140px; }
            .det-class {
                display: inline-block;
                min-width: 64px;
                border-radius: 999px;
                padding: 4px 12px;
                background: rgba(34, 211, 238, 0.14);
                color: #67e8f9;
                font-weight: 800;
                text-align: center;
            }
            .class-card-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 14px;
                margin-top: 10px;
                margin-bottom: 10px;
            }
            .class-card {
                border: 1px solid rgba(148, 163, 184, 0.24);
                border-radius: 8px;
                background: rgba(15, 23, 42, 0.38);
                padding: 16px;
            }
            .class-card-label {
                color: #93c5fd;
                font-size: 0.86rem;
                font-weight: 800;
                text-transform: uppercase;
                margin-bottom: 6px;
            }
            .class-card-value {
                color: #f8fafc;
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
            }
            .image-frame {
                width: 100%;
                height: min(70vh, 620px);
                min-height: 360px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                border-radius: 8px;
                background: rgba(15, 23, 42, 0.35);
                border: 1px solid rgba(148, 163, 184, 0.18);
            }
            .image-frame img {
                display: block;
                max-width: 100%;
                max-height: 100%;
                width: auto;
                height: auto;
                object-fit: contain;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_detection_table(detections: list[dict[str, str | float]]) -> None:
    rows_html = []
    for index, row in enumerate(detections, start=1):
        class_name = escape(display_class_name(str(row["class"])))
        rows_html.append(
            "<tr>"
            f"<td>#{index}</td>"
            f"<td><span class='det-class'>{class_name}</span></td>"
            f"<td>{float(row['confidence']):.2%}</td>"
            f"<td>{float(row['x1']):.1f}</td>"
            f"<td>{float(row['y1']):.1f}</td>"
            f"<td>{float(row['x2']):.1f}</td>"
            f"<td>{float(row['y2']):.1f}</td>"
            "</tr>"
        )

    st.markdown(
        "<div class='det-panel'>"
        "<div class='det-panel-title'>Tespit Edilen Nesneler</div>"
        "<table class='det-table'>"
        "<thead><tr>"
        "<th>No</th><th>Sinif</th><th>Guven</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )


def render_class_count_cards(class_counts: dict[str, int]) -> None:
    gun_count = class_counts.get("gun", 0)
    knife_count = class_counts.get("knife", 0)
    st.markdown(
        f"""
        <div class="class-card-grid">
            <div class="class-card">
                <div class="class-card-label">Silah</div>
                <div class="class-card-value">{gun_count}</div>
                <div class="class-card-caption">gun sinifi</div>
            </div>
            <div class="class-card">
                <div class="class-card-label">Bicak</div>
                <div class="class-card-value">{knife_count}</div>
                <div class="class-card-caption">knife sinifi</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_image(uploaded_file) -> Path:
    upload_dir = OUTPUT_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    image_path = upload_dir / f"image_{int(time.time() * 1000)}{suffix}"
    image_path.write_bytes(uploaded_file.getbuffer())
    return image_path


def run_image_prediction(model: YOLO, image_path: Path, confidence: float, iou: float, image_size: int) -> dict[str, object]:
    output_dir = OUTPUT_ROOT / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    start_time = time.perf_counter()
    result = model.predict(
        source=image_array,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    plotted_bgr = result.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    output_path = output_dir / f"annotated_{image_path.stem}_{int(time.time() * 1000)}.jpg"
    Image.fromarray(plotted_rgb).save(output_path, quality=95)

    return {
        "input_path": str(image_path),
        "output_path": str(output_path),
        "elapsed_ms": elapsed_ms,
        "detections": get_detection_rows(result),
    }


def get_ffmpeg_path() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_browser_playable_video(video_path: Path) -> Path:
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path is None:
        return video_path

    converted_path = video_path.with_name(f"{video_path.stem}_browser.mp4")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(converted_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, OSError):
        return video_path

    if converted_path.exists() and converted_path.stat().st_size > 0:
        return converted_path
    return video_path


def save_uploaded_video(uploaded_file) -> Path:
    upload_dir = OUTPUT_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    video_path = upload_dir / f"video_{int(time.time() * 1000)}{suffix}"
    video_path.write_bytes(uploaded_file.getbuffer())
    return video_path


def process_video(
    model: YOLO,
    source_path: Path,
    confidence: float,
    iou: float,
    image_size: int,
    frame_stride: int,
    max_frames: int,
) -> dict[str, object]:
    output_dir = OUTPUT_ROOT / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError("Video dosyasi acilamadi.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    output_fps = max(fps / max(frame_stride, 1), 1.0)

    output_path = output_dir / f"annotated_{source_path.stem}_{int(time.time() * 1000)}.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Tespit videosu olusturulamadi.")

    progress = st.progress(0, text="Video isleniyor...")
    status = st.empty()
    class_counts: dict[str, int] = {}
    max_confidence = 0.0
    processed_frames = 0
    written_frames = 0
    start_time = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            processed_frames += 1
            if processed_frames % frame_stride != 0:
                continue
            if max_frames and written_frames >= max_frames:
                break

            result = model.predict(source=frame, conf=confidence, iou=iou, imgsz=image_size, verbose=False)[0]
            writer.write(result.plot())
            written_frames += 1

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    box_conf = float(box.conf[0].item())
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    max_confidence = max(max_confidence, box_conf)

            if max_frames:
                progress_value = min(written_frames / max_frames, 1.0)
            elif total_frames:
                progress_value = min(processed_frames / total_frames, 1.0)
            else:
                progress_value = 0.0
            progress.progress(progress_value, text="Video isleniyor...")
            status.caption(f"Okunan kare: {processed_frames} | Yazilan kare: {written_frames}")
    finally:
        capture.release()
        writer.release()

    progress.progress(1.0, text="Video isleme tamamlandi.")
    return {
        "output_path": str(output_path),
        "processed_frames": processed_frames,
        "written_frames": written_frames,
        "elapsed_seconds": time.perf_counter() - start_time,
        "class_counts": class_counts,
        "max_confidence": max_confidence,
    }



def image_to_data_uri(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_image_pair(input_path: str | Path, output_path: str | Path) -> None:
    input_uri = image_to_data_uri(input_path)
    output_uri = image_to_data_uri(output_path)
    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                overflow: hidden;
                font-family: Arial, sans-serif;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                width: 100%;
            }}
            .title {{
                color: #f8fafc;
                font-size: 24px;
                line-height: 1.2;
                font-weight: 800;
                margin: 0 0 12px 0;
            }}
            .frame {{
                height: 560px;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                border-radius: 8px;
                background: rgba(15, 23, 42, 0.35);
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-sizing: border-box;
            }}
            img {{
                display: block;
                max-width: 100%;
                max-height: 100%;
                width: auto;
                height: auto;
                object-fit: contain;
            }}
            @media (max-width: 900px) {{
                .grid {{ grid-template-columns: 1fr; }}
                .frame {{ height: 420px; }}
            }}
        </style>
    </head>
    <body>
        <div class="grid">
            <section>
                <h3 class="title">Yuklenen Gorsel</h3>
                <div class="frame"><img src="{input_uri}" alt="Yuklenen gorsel" /></div>
            </section>
            <section>
                <h3 class="title">Tespit Sonucu</h3>
                <div class="frame"><img src="{output_uri}" alt="Tespit sonucu" /></div>
            </section>
        </div>
    </body>
    </html>
    """
    components.html(html, height=620, scrolling=False)

def show_image_result(result: dict[str, object], confidence: float) -> None:
    render_image_pair(result["input_path"], result["output_path"])

    detections = result["detections"]
    st.subheader("Gorsel Tespit Ozeti")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tespit sayisi", len(detections))
    col_b.metric("Cikarim suresi", f"{float(result['elapsed_ms']):.1f} ms")
    col_c.metric("Guven esigi", f"{confidence:.2f}")

    if detections:
        render_detection_table(detections)
    else:
        st.warning("Secilen guven esiginde tespit bulunamadi.")


def show_video_result(source_path: Path, display_path: Path, summary: dict[str, object]) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Yuklenen Video")
        st.video(source_path.read_bytes())
    with right:
        st.subheader("Tespit Videosu")
        st.video(display_path.read_bytes(), format="video/mp4")

    class_counts = summary["class_counts"]
    frame_level_detections = sum(class_counts.values())
    st.subheader("Video Tespit Ozeti")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Kare bazli toplam tespit", frame_level_detections)
    col_b.metric("Yazilan kare", summary["written_frames"])
    col_c.metric("Islem suresi", f"{summary['elapsed_seconds']:.1f} s")
    col_d.metric("En yuksek guven", f"{summary['max_confidence']:.2%}")

    if class_counts:
        render_class_count_cards(class_counts)
        st.caption("Not: Video tespit sayilari kare bazlidir; ayni nesne farkli karelerde tekrar sayilabilir.")
    else:
        st.warning("Islenen video karelerinde tespit bulunamadi.")

    with open(display_path, "rb") as video_file:
        st.download_button(
            "Tespit videosunu indir",
            data=video_file,
            file_name=display_path.name,
            mime="video/mp4",
        )


def main() -> None:
    render_css()
    st.title("Silah ve Bicak Tespiti")

    with st.sidebar:
        st.header("Model")
        st.markdown(
            """
            <div class="model-badge">
                <div class="model-badge-label">Aktif model</div>
                <div class="model-badge-value">YOLO11n</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        confidence = st.slider("Guven esigi", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU esigi", 0.10, 0.90, 0.45, 0.05)

        st.divider()
        st.header("Gorsel")
        image_size = st.selectbox("Gorsel boyutu", [416, 512, 640, 768], index=2)

        st.divider()
        st.header("Video")
        video_image_size = st.selectbox("Video isleme boyutu", [320, 416, 512, 640], index=1)
        frame_stride = st.selectbox("Kare atlama araligi", [5, 10, 15, 20], index=1)
        max_frames = st.number_input("Maksimum islenecek kare", min_value=30, max_value=600, value=90, step=30)
        st.caption("HF CPU demosunda video kisa onizleme olarak islenir. Yerel bilgisayarda tam video isleme daha uygundur.")

        st.divider()
        st.caption("Final YOLO11n model metrikleri")
        st.metric("mAP50", "0.964")
        st.metric("mAP50-95", "0.698")
        st.metric("Siniflar", "gun, knife")

    model = load_model()
    image_tab, video_tab = st.tabs(["Gorsel Tespiti", "Video Tespiti"])

    with image_tab:
        uploaded_image = st.file_uploader(
            "Tespit yapmak icin bir gorsel yukle",
            type=sorted(IMAGE_EXTENSIONS),
            key="image_uploader",
        )
        if uploaded_image is None:
            st.info("Gorsel tespiti icin bir dosya yukle.")
        else:
            image_key = (uploaded_image.name, uploaded_image.size, confidence, iou, image_size)
            if st.button("Gorsel tespitini baslat", type="primary"):
                image_path = save_uploaded_image(uploaded_image)
                with st.spinner("Gorsel uzerinde tespit yapiliyor..."):
                    st.session_state["image_result"] = {
                        "key": image_key,
                        "result": run_image_prediction(model, image_path, confidence, iou, image_size),
                    }
            cached = st.session_state.get("image_result")
            if cached and cached.get("key") == image_key:
                show_image_result(cached["result"], confidence)

    with video_tab:
        uploaded_video = st.file_uploader(
            "Tespit yapmak icin bir video yukle",
            type=sorted(VIDEO_EXTENSIONS),
            key="video_uploader",
        )
        if uploaded_video is None:
            st.info("Video tespiti icin bir dosya yukle.")
        else:
            video_key = (uploaded_video.name, uploaded_video.size, confidence, iou, video_image_size, frame_stride, int(max_frames))
            st.warning("Hugging Face ucretsiz CPU ortaminda video tespiti yavas olabilir. Bu demo yalnizca kisa onizleme isler.")
            if st.button("Video tespitini baslat", type="primary"):
                source_path = save_uploaded_video(uploaded_video)
                with st.spinner("Video uzerinde tespit yapiliyor..."):
                    summary = process_video(
                        model=model,
                        source_path=source_path,
                        confidence=confidence,
                        iou=iou,
                        image_size=video_image_size,
                        frame_stride=frame_stride,
                        max_frames=int(max_frames),
                    )
                    output_path = Path(summary["output_path"])
                    display_path = make_browser_playable_video(output_path)
                    st.session_state["video_result"] = {
                        "key": video_key,
                        "source_path": str(source_path),
                        "display_path": str(display_path),
                        "summary": summary,
                    }
            cached = st.session_state.get("video_result")
            if cached and cached.get("key") == video_key:
                show_video_result(Path(cached["source_path"]), Path(cached["display_path"]), cached["summary"])


if __name__ == "__main__":
    main()


