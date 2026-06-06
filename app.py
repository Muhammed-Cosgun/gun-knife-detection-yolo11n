from __future__ import annotations

import shutil
import subprocess
import time
from html import escape
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO


MODEL_REPO_ID = "cosgun99/gun-knife-yolo11n"
MODEL_FILENAME = "best.pt"
OUTPUT_ROOT = Path("runs") / "streamlit_outputs"
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
CLASS_LABELS = {"gun": "Silah", "knife": "BÄ±Ã§ak"}


st.set_page_config(
    page_title="Silah ve BÄ±Ã§ak Tespiti",
    layout="wide",
)


@st.cache_resource
def load_model() -> YOLO:
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    return YOLO(model_path)


def display_class_name(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name)


def get_detection_rows(result) -> list[dict[str, str | float]]:
    rows = []
    names = result.names

    if result.boxes is None:
        return rows

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        rows.append(
            {
                "class": names[class_id],
                "confidence": round(confidence, 4),
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
            }
        )

    return rows


def render_detection_table(detections: list[dict[str, str | float]]) -> None:
    rows_html = []
    for index, row in enumerate(detections, start=1):
        class_name = escape(display_class_name(str(row["class"])))
        rows_html.append(
            "<tr>"
            f"<td class='det-index'>#{index}</td>"
            f"<td><span class='det-class'>{class_name}</span></td>"
            f"<td class='num'>{float(row['confidence']):.2%}</td>"
            f"<td class='num'>{float(row['x1']):.1f}</td>"
            f"<td class='num'>{float(row['y1']):.1f}</td>"
            f"<td class='num'>{float(row['x2']):.1f}</td>"
            f"<td class='num'>{float(row['y2']):.1f}</td>"
            "</tr>"
        )

    table_html = (
        "<div class='det-panel'>"
        "<div class='det-panel-title'>Tespit Edilen Nesneler</div>"
        "<table class='det-table'>"
        "<thead><tr>"
        "<th>No</th><th>SÄ±nÄ±f</th><th>GÃ¼ven</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_class_count_cards(class_counts: dict[str, int]) -> None:
    gun_count = class_counts.get("gun", 0)
    knife_count = class_counts.get("knife", 0)
    st.markdown(
        f"""
        <div class="class-card-grid">
            <div class="class-card">
                <div class="class-card-label">Silah</div>
                <div class="class-card-value">{gun_count}</div>
                <div class="class-card-caption">gun sÄ±nÄ±fÄ±</div>
            </div>
            <div class="class-card">
                <div class="class-card-label">BÄ±Ã§ak</div>
                <div class="class-card-value">{knife_count}</div>
                <div class="class-card-caption">knife sÄ±nÄ±fÄ±</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            .det-table th:nth-child(1),
            .det-table th:nth-child(2),
            .det-table td:nth-child(1),
            .det-table td:nth-child(2) {
                text-align: left;
            }
            .det-table td {
                color: #f8fafc;
                padding: 14px 16px;
                text-align: right;
                border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            }
            .det-table tbody tr:last-child td {
                border-bottom: 0;
            }
            .det-table th:nth-child(1), .det-table td:nth-child(1) { width: 72px; }
            .det-table th:nth-child(2), .det-table td:nth-child(2) { width: 140px; }
            .det-index {
                color: #94a3b8 !important;
            }
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
            .num {
                font-variant-numeric: tabular-nums;
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
            .class-card-caption {
                color: #94a3b8;
                font-size: 0.84rem;
                margin-top: 6px;
            }
            div[data-testid="stImage"] {
                min-height: 320px;
            }
            div[data-testid="stImage"] img {
                display: block;
                width: 100%;
                max-height: 72vh;
                object-fit: contain;
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_image_detection(model: YOLO, image: Image.Image, confidence: float, iou: float, image_size: int) -> None:
    image_array = np.array(image)
    left, right = st.columns(2)

    with left:
        st.subheader("YÃ¼klenen GÃ¶rsel")
        st.image(image, width="stretch")

    start_time = time.perf_counter()
    results = model.predict(
        source=image_array,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    result = results[0]

    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]
    detections = get_detection_rows(result)

    with right:
        st.subheader("Tespit Sonucu")
        st.image(plotted_rgb, width="stretch")

    st.subheader("GÃ¶rsel Tespit Ã–zeti")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tespit sayÄ±sÄ±", len(detections))
    col_b.metric("Ã‡Ä±karÄ±m sÃ¼resi", f"{elapsed_ms:.1f} ms")
    col_c.metric("GÃ¼ven eÅŸiÄŸi", f"{confidence:.2f}")

    if detections:
        render_detection_table(detections)
    else:
        st.warning("SeÃ§ilen gÃ¼ven eÅŸiÄŸinde tespit bulunamadÄ±.")


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
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return video_path

    if converted_path.exists() and converted_path.stat().st_size > 0:
        return converted_path
    return video_path


def save_uploaded_video(uploaded_file) -> Path:
    upload_dir = OUTPUT_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    video_path = upload_dir / f"input_{int(time.time() * 1000)}{suffix}"
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
        raise RuntimeError("Video dosyasÄ± aÃ§Ä±lamadÄ±.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    output_fps = max(fps / max(frame_stride, 1), 1.0)

    output_path = output_dir / f"annotated_{source_path.stem}_{int(time.time() * 1000)}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Tespit videosu oluÅŸturulamadÄ±.")

    progress = st.progress(0, text="Video iÅŸleniyor...")
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

            result = model.predict(
                source=frame,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                verbose=False,
            )[0]
            annotated_frame = result.plot()
            writer.write(annotated_frame)
            written_frames += 1

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    box_conf = float(box.conf[0].item())
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    max_confidence = max(max_confidence, box_conf)

            if total_frames:
                progress_value = min(processed_frames / total_frames, 1.0)
            elif max_frames:
                progress_value = min(written_frames / max_frames, 1.0)
            else:
                progress_value = 0.0
            progress.progress(progress_value, text="Video iÅŸleniyor...")
            status.caption(f"Okunan kare: {processed_frames} | YazÄ±lan kare: {written_frames}")
    finally:
        capture.release()
        writer.release()

    progress.progress(1.0, text="Video iÅŸleme tamamlandÄ±.")
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "output_path": output_path,
        "processed_frames": processed_frames,
        "written_frames": written_frames,
        "elapsed_seconds": elapsed_seconds,
        "class_counts": class_counts,
        "max_confidence": max_confidence,
    }


def run_video_detection(
    model: YOLO,
    uploaded_file,
    confidence: float,
    iou: float,
    image_size: int,
    frame_stride: int,
    max_frames: int,
    cache_key: tuple[object, ...],
) -> None:
    cached = st.session_state.get("video_detection_cache")
    use_cached_result = False

    if cached and cached.get("key") == cache_key:
        cached_source = Path(cached["source_path"])
        cached_display = Path(cached["display_path"])
        cached_summary = cached["summary"]
        cached_output = Path(cached_summary["output_path"])
        use_cached_result = cached_source.exists() and cached_display.exists() and cached_output.exists()

    if use_cached_result:
        source_path = cached_source
        display_path = cached_display
        summary = cached_summary
        output_path = Path(summary["output_path"])
    else:
        source_path = save_uploaded_video(uploaded_file)
        summary = None
        output_path = None
        display_path = None

    left, right = st.columns(2)
    with left:
        st.subheader("YÃ¼klenen Video")
        st.video(source_path.read_bytes())

    if summary is None:
        with st.spinner("Video Ã¼zerinde tespit yapÄ±lÄ±yor..."):
            summary = process_video(
                model=model,
                source_path=source_path,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
                frame_stride=frame_stride,
                max_frames=max_frames,
            )
        output_path = Path(summary["output_path"])
        display_path = make_browser_playable_video(output_path)
        summary["output_path"] = str(output_path)
        st.session_state["video_detection_cache"] = {
            "key": cache_key,
            "source_path": str(source_path),
            "display_path": str(display_path),
            "summary": summary,
        }
    else:
        st.info("Video veya ayarlar deÄŸiÅŸmediÄŸi iÃ§in Ã¶nceki tespit sonucu kullanÄ±lÄ±yor.")

    output_path = Path(summary["output_path"])
    display_path = Path(display_path)
    with right:
        st.subheader("Tespit Videosu")
        st.video(display_path.read_bytes(), format="video/mp4")
        if display_path == output_path:
            st.caption("Ã–nizleme boÅŸ kalÄ±rsa aÅŸaÄŸÄ±daki indirme butonunu kullanabilirsin. Video dosyasÄ± yine kaydedilir.")

    st.subheader("Video Tespit Ã–zeti")
    class_counts = summary["class_counts"]
    frame_level_detections = sum(class_counts.values())
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Kare bazlÄ± toplam tespit", frame_level_detections)
    col_b.metric("Ä°ÅŸlenen kare", summary["written_frames"])
    col_c.metric("Ä°ÅŸlem sÃ¼resi", f"{summary['elapsed_seconds']:.1f} s")
    col_d.metric("En yÃ¼ksek gÃ¼ven", f"{summary['max_confidence']:.2%}")

    if class_counts:
        render_class_count_cards(class_counts)
        st.caption("Not: Video tespit sayÄ±larÄ± kare bazlÄ±dÄ±r; aynÄ± nesne farklÄ± karelerde tekrar sayÄ±labilir.")
    else:
        st.warning("Ä°ÅŸlenen video karelerinde tespit bulunamadÄ±.")

    with open(display_path, "rb") as video_file:
        st.download_button(
            "Tespit videosunu indir",
            data=video_file,
            file_name=display_path.name,
            mime="video/mp4",
        )

    st.caption(f"Tespit videosu kaydedildi: {output_path}")
    if display_path != output_path:
        st.caption(f"TarayÄ±cÄ± uyumlu Ã¶nizleme videosu: {display_path}")


def main() -> None:
    render_css()
    st.title("Silah ve BÄ±Ã§ak Tespiti")

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

        confidence = st.slider("GÃ¼ven eÅŸiÄŸi", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU eÅŸiÄŸi", 0.10, 0.90, 0.45, 0.05)
        image_size = st.selectbox("GÃ¶rÃ¼ntÃ¼ boyutu", [416, 512, 640, 768], index=2)

        st.divider()
        st.header("Video")
        frame_stride = st.selectbox("Kare atlama araligi", [1, 2, 3, 5, 10], index=3)
        max_frames = st.number_input("Maksimum islenecek kare (0 = tumu)", min_value=0, value=300, step=50)
        st.caption("Hugging Face CPU demosu icin varsayilan video ayarlari hizli moda alinmistir. Tum videoyu islemek icin kare atlama araligini 1, maksimum kare degerini 0 yapabilirsin.")

        st.divider()
        st.caption("Final YOLO11n model metrikleri")
        st.metric("mAP50", "0.964")
        st.metric("mAP50-95", "0.698")
        st.metric("SÄ±nÄ±flar", "gun, knife")

    model = load_model()

    image_tab, video_tab = st.tabs(["GÃ¶rsel Tespiti", "Video Tespiti"])

    with image_tab:
        uploaded_image = st.file_uploader(
            "Tespit yapmak iÃ§in bir gÃ¶rsel yÃ¼kle",
            type=sorted(IMAGE_EXTENSIONS),
            key="image_uploader",
        )
        if uploaded_image is None:
            st.info("GÃ¶rsel tespiti iÃ§in bir dosya yÃ¼kle.")
        else:
            image = Image.open(uploaded_image).convert("RGB")
            run_image_detection(model, image, confidence, iou, image_size)

    with video_tab:
        uploaded_video = st.file_uploader(
            "Tespit yapmak iÃ§in bir video yÃ¼kle",
            type=sorted(VIDEO_EXTENSIONS),
            key="video_uploader",
        )
        if uploaded_video is None:
            st.info("Video tespiti iÃ§in bir dosya yÃ¼kle.")
        else:
            video_size = getattr(uploaded_video, "size", len(uploaded_video.getbuffer()))
            video_cache_key = (
                uploaded_video.name,
                video_size,
                round(confidence, 4),
                round(iou, 4),
                image_size,
                frame_stride,
                int(max_frames),
                MODEL_REPO_ID,
                MODEL_FILENAME,
            )
            run_video_detection(
                model=model,
                uploaded_file=uploaded_video,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
                frame_stride=frame_stride,
                max_frames=int(max_frames),
                cache_key=video_cache_key,
            )


if __name__ == "__main__":
    main()

