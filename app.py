"""AI Shorts Maker — Streamlit UI ("Shortify"-style)"""

import logging
import os

import streamlit as st

from backend import LENGTH_PRESETS, NUM_SHORTS_OPTIONS, ShortConfig, make_my_short, make_viral_shorts

logger = logging.getLogger("shorts_maker.app")

st.set_page_config(page_title="Shortify — AI Shorts Maker", page_icon="✂️", layout="centered")

st.markdown(
    """
    <style>
    .block-container {max-width: 760px; padding-top: 2.5rem;}
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #b9a8f5;
        border-radius: 16px;
        padding: 2.2rem 1rem;
        background: #faf9ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;margin-bottom:0;'>✂️ Shortify</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#777;margin-top:0.3rem;'>"
    "Long video → viral shorts with auto captions, apni machine par.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a video (or drag it here)", type=["mp4", "mov", "webm"],
    help="MP4 · MOV · WebM — a few minutes works best",
)

col1, col2, col3 = st.columns(3)
with col1:
    num_shorts_choice = st.selectbox("How many shorts", NUM_SHORTS_OPTIONS)
with col2:
    length_choice = st.selectbox("Short length", list(LENGTH_PRESETS.keys()))
with col3:
    enhance_choice = st.selectbox(
        "Auto-enhance", ["On — polish & clean (recommended)", "Off"],
    )
auto_enhance = enhance_choice.startswith("On")

manual_mode = False
start_time, end_time = 0, 15
whisper_size, font_size = "base", 55

with st.expander("⚙️ Advanced"):
    manual_mode = st.checkbox(
        "Manually set start/end time instead (skips auto highlight detection)"
    )
    mc1, mc2 = st.columns(2)
    with mc1:
        start_time = st.number_input("Start (seconds)", value=0, min_value=0, disabled=not manual_mode)
    with mc2:
        end_time = st.number_input("End (seconds)", value=15, min_value=1, disabled=not manual_mode)
    whisper_size = st.selectbox("Speech recognition model", ["tiny", "base", "small"], index=1)
    font_size = st.slider("Caption font size", 30, 90, 55)

st.write("")
run_clicked = st.button("💥 Make Viral Shorts Now", use_container_width=True)

if run_clicked:
    if uploaded_file is None:
        st.error("Pehle koi video select ya upload karen!")
    elif manual_mode and end_time <= start_time:
        st.error("Khatam hone ka waqt shuruat se zyada hona chahiye!")
    else:
        try:
            if manual_mode:
                with st.spinner("Aapka clip taiyar ho raha hai..."):
                    config = ShortConfig(
                        start_sec=start_time, end_sec=end_time,
                        whisper_model_size=whisper_size, font_size=font_size,
                    )
                    output_files = [make_my_short(uploaded_file, start_time, end_time, config=config)]
            else:
                with st.spinner(
                    "AI puri video scan kar raha hai aur best moments dhoond raha hai... "
                    "Is mein thoda waqt lag sakta hai."
                ):
                    output_files = make_viral_shorts(
                        uploaded_file,
                        num_shorts_choice=num_shorts_choice,
                        length_choice=length_choice,
                        auto_enhance=auto_enhance,
                        whisper_model_size=whisper_size,
                        font_size=font_size,
                    )

            if not output_files:
                st.warning("Koi acha moment nahi mila. Video ki length ya settings badal kar dobara try karen.")
            else:
                st.success(f"Mubarak ho! {len(output_files)} short(s) taiyar hain.")
                for i, path in enumerate(output_files, start=1):
                    st.markdown(f"**Short {i}**")
                    st.video(path)
                    with open(path, "rb") as f:
                        st.download_button(
                            f"📥 Download Short {i}", data=f,
                            file_name=os.path.basename(path), mime="video/mp4",
                            key=f"dl_{i}",
                        )
        except Exception as exc:
            logger.exception("Short generation failed")
            st.error(f"Kuch ghalat ho gaya: {exc}")