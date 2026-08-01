"""AI Shorts Maker — Streamlit UI"""

import logging
import os

import streamlit as st

from backend import ShortConfig, make_my_short

logger = logging.getLogger("shorts_maker.app")

st.set_page_config(page_title="AI Shorts Maker", page_icon="✂️", layout="centered")
st.title("✂️ Premium AI Shorts Maker")
st.write("Apni lambi video ko automatic viral shorts mein badlein!")

uploaded_file = st.file_uploader("Apni video yahan upload karen (MP4, MOV)", type=["mp4", "mov"])

st.subheader("⚙️ Customization Settings")
col1, col2 = st.columns(2)
with col1:
    start_time = st.number_input("Short kahan se shuru ho? (Seconds)", value=0, min_value=0)
with col2:
    end_time = st.number_input("Short kahan khatam ho? (Seconds)", value=15, min_value=1)

with st.expander("Advanced options"):
    whisper_size = st.selectbox(
        "Speech recognition model", ["tiny", "base", "small"], index=1,
        help="Bigger models are more accurate but slower.",
    )
    font_size = st.slider("Caption font size", 30, 90, 55)


def validate_inputs() -> str | None:
    if uploaded_file is None:
        return "Pehle koi video select ya upload karen!"
    if end_time <= start_time:
        return "Khatam hone ka waqt shuruat se zyada hona chahiye!"
    return None


if st.button("💥 Make Viral Shorts Now"):
    error = validate_inputs()
    if error:
        st.error(error)
    else:
        config = ShortConfig(
            start_sec=start_time,
            end_sec=end_time,
            whisper_model_size=whisper_size,
            font_size=font_size,
        )
        try:
            with st.spinner("AI Engine video aur speech detect kar raha hai... Is mein thoda waqt lag sakta hai."):
                output_file = make_my_short(uploaded_file, start_time, end_time, config=config)

            st.success("Mubarak ho! Aap ka short dynamic subtitles ke sath tayar hai.")
            st.video(output_file)
            with open(output_file, "rb") as f:
                st.download_button(
                    "📥 Download Short Video",
                    data=f,
                    file_name="ai_short.mp4",
                    mime="video/mp4",
                )
        except Exception as exc:  # surface a friendly message, keep full detail in logs
            logger.exception("Short generation failed")
            st.error(f"Kuch ghalat ho gaya: {exc}")