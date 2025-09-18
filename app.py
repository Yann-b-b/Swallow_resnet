import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


SAMPLE_RATE_HZ = 4000
LABEL_FPS = 60
DATA_ROOT = Path(__file__).resolve().parent / "data"


@st.cache_data(show_spinner=False)
def list_signal_files(data_root: Path) -> list[Path]:
    signals_dir = data_root / "signals"
    return sorted(signals_dir.glob("*.npy"))


@st.cache_data(show_spinner=False)
def load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"start", "end"}.issubset(df.columns):
        raise ValueError("Labels CSV must contain 'start' and 'end' columns (frames @ 60fps)")
    df = df.copy()
    df["start_s"] = df["start"].astype(float) / LABEL_FPS
    df["end_s"] = df["end"].astype(float) / LABEL_FPS
    df["duration_s"] = df["end_s"] - df["start_s"]
    return df


def load_mic_channel(npy_path: Path) -> np.ndarray:
    data = np.load(str(npy_path), allow_pickle=False)
    if data.ndim == 1:
        return data.astype(np.float64)
    if data.ndim >= 2:
        return data[:, 0].astype(np.float64)
    raise ValueError("Unsupported array shape for audio conversion")


def decimate_for_plot(signal: np.ndarray, max_points: int = 400_000) -> tuple[np.ndarray, int]:
    if signal.size <= max_points:
        return signal, 1
    step = int(np.ceil(signal.size / max_points))
    return signal[::step], step


def build_waveform_figure(signal: np.ndarray, sample_rate_hz: int, events: pd.DataFrame | None) -> go.Figure:
    y_plot, step = decimate_for_plot(signal)
    x_plot = np.arange(0, y_plot.size * step, step) / float(sample_rate_hz)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", name="mic"))

    if events is not None and not events.empty:
        for _, row in events.iterrows():
            fig.add_vrect(x0=row["start_s"], x1=row["end_s"], fillcolor="#ff7f0e", opacity=0.15, line_width=0)
        # Add markers for starts/ends
        fig.add_trace(
            go.Scatter(
                x=events["start_s"], y=[0] * len(events), mode="markers", name="start", marker=dict(color="#2ca02c"),
                hovertext=[f"start={s:.2f}s" for s in events["start_s"]], hoverinfo="text"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=events["end_s"], y=[0] * len(events), mode="markers", name="end", marker=dict(color="#d62728"),
                hovertext=[f"end={e:.2f}s" for e in events["end_s"]], hoverinfo="text"
            )
        )

    fig.update_layout(
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=350,
        showlegend=True,
    )
    return fig


def make_mp4_from_signal(signal: np.ndarray, sample_rate_hz: int) -> bytes | None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return None

    # Normalize to int16 WAV in a temp file, then encode to MP4 (audio-only AAC)
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if not np.isfinite(max_abs) or max_abs == 0.0:
        pcm = np.zeros_like(signal, dtype=np.int16)
    else:
        normalized = np.clip(signal / max_abs, -1.0, 1.0)
        pcm = (normalized * 32767.0).astype(np.int16)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "audio.wav"
        tmp_mp4 = Path(tmpdir) / "audio.mp4"

        # Write WAV
        import wave
        with wave.open(str(tmp_wav), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate_hz)
            wav_file.writeframes(pcm.tobytes())

        # Encode to MP4 (AAC)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(tmp_wav),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(tmp_mp4),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not tmp_mp4.exists():
            return None
        return tmp_mp4.read_bytes()


st.set_page_config(page_title="Swallow Audio Viewer", layout="wide")
st.title("Swallow Audio Viewer")
st.write("This streamlit app allows you to view the audio signals and labels for the Swallow Resnet dataset.")
st.write("The dataset is available at https://zenodo.org/records/4539695")
st.write("The paper is available at https://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Kiyokawa_101.pdf")
st.write("average duration of the audio signals is 8.94 seconds")
signal_files = list_signal_files(DATA_ROOT)
if not signal_files:
    st.warning("No .npy files found under data/signals/")
    st.stop()

options = [p.name for p in signal_files]
selected_name = st.selectbox("Signal file", options, index=0, placeholder="Type to search…")
selected_path = next(p for p in signal_files if p.name == selected_name)
labels_path = DATA_ROOT / "labels" / (selected_path.stem + ".csv")

cols = st.columns([2, 1])
with cols[0]:
    mic = load_mic_channel(selected_path)
    events_df = None
    if labels_path.exists():
        try:
            events_df = load_labels(labels_path)
        except Exception as exc:
            st.error(f"Failed to read labels: {exc}")
    fig = build_waveform_figure(mic, SAMPLE_RATE_HZ, events_df)
    st.plotly_chart(fig, use_container_width=True)

    if events_df is not None and not events_df.empty:
        show_tbl = st.checkbox("Show event table", value=False)
        if show_tbl:
            st.dataframe(
                events_df[["start", "end", "start_s", "end_s", "duration_s"]].rename(
                    columns={"start": "start_frame", "end": "end_frame"}
                ),
                use_container_width=True,
            )

with cols[1]:
    st.subheader("Playback")
    mp4_bytes = make_mp4_from_signal(mic, SAMPLE_RATE_HZ)
    if mp4_bytes is not None:
        st.video(mp4_bytes)
    else:
        st.info("ffmpeg not found. Falling back to audio stream.")
        # Audio fallback (Streamlit supports raw bytes for audio)
        # Reuse the WAV in-memory approach
        max_abs = float(np.max(np.abs(mic))) if mic.size else 0.0
        if not np.isfinite(max_abs) or max_abs == 0.0:
            pcm = np.zeros_like(mic, dtype=np.int16)
        else:
            normalized = np.clip(mic / max_abs, -1.0, 1.0)
            pcm = (normalized * 32767.0).astype(np.int16)
        import wave
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE_HZ)
            wav_file.writeframes(pcm.tobytes())
        st.audio(wav_buf.getvalue(), format="audio/wav")

st.caption("Select a signal to view amplitude with overlaid swallow intervals; playback available as MP4 (AAC) or WAV fallback.")


