import streamlit as st

st.set_page_config(page_title="AI Shorts Maker", page_icon="✂️", layout="centered")

st.title("✂️ Premium AI Shorts Maker")
st.write("Apni lambi video ko automatic viral shorts mein badlein!")

# 1. Video Upload karne ka box
uploaded_file = st.file_uploader("Apni video yahan upload karen (MP4, MOV)", type=["mp4", "mov"])

# 2. Settings ke options
st.subheader("⚙️ Customization Settings")
duration = st.selectbox("Short ki length kitni honi chahiye?", ["10s-30s", "30s-60s", "60s-90s"])
shorts_count = st.slider("Kitne shorts nikalne hain?", min_value=1, max_value=5, value=3)

# 3. Main Button
if st.button("💥 Make Viral Shorts Now"):
    if uploaded_file is not None:
        st.success("AI Model Load ho raha hai... Video processing background mein chalu hai!")
        st.info("Kuch hi der mein aap ke clips download ke liye tayar ho jayenge.")
    else:
        st.error("Pehle koi video select ya upload karen!")
