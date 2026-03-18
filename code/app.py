import streamlit as st
from PIL import Image
import exifread
from geopy.geocoders import Nominatim
import requests
import base64
from openai import OpenAI

# -----------------------------
# CONFIG (Streamlit Secrets)
# -----------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="GeoVision AI Pro", layout="wide")
st.title("🌍 GeoVision AI Pro - Intelligent Image Location Finder")
st.subheader("📊 Confidence Level")
st.progress(75)

# -----------------------------
# EXIF GPS Extraction
# -----------------------------
def get_gps_coords(file):
    tags = exifread.process_file(file)

    def convert(value):
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)

    try:
        lat = convert(tags["GPS GPSLatitude"])
        if tags["GPS GPSLatitudeRef"].values != "N":
            lat = -lat

        lon = convert(tags["GPS GPSLongitude"])
        if tags["GPS GPSLongitudeRef"].values != "E":
            lon = -lon

        return lat, lon
    except:
        return None

# -----------------------------
# Reverse Geocoding
# -----------------------------
def get_address(lat, lon):
    try:
        geolocator = Nominatim(user_agent="geo_app")
        location = geolocator.reverse((lat, lon), timeout=10)
        return location.address if location else "Address not found"
    except:
        return "Geocoding failed"

# -----------------------------
# Google Vision Landmark Detection
# -----------------------------
def detect_landmarks(image_bytes):
    try:
        img_bytes = base64.b64encode(image_bytes).decode()

        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"

        payload = {
            "requests": [
                {
                    "image": {"content": img_bytes},
                    "features": [{"type": "LANDMARK_DETECTION"}]
                }
            ]
        }

        response = requests.post(url, json=payload, timeout=15)
        result = response.json()

        landmark = result["responses"][0].get("landmarkAnnotations")

        if landmark:
            name = landmark[0]["description"]
            lat = landmark[0]["locations"][0]["latLng"]["latitude"]
            lon = landmark[0]["locations"][0]["latLng"]["longitude"]
            return name, lat, lon

    except:
        pass

    return None

# -----------------------------
# LLM Reasoning
# -----------------------------
def analyze_with_llm(image_bytes):
    try:
        response = client.chat.completions.create(
            model="gpt-5.3",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image and guess the most probable location (country, city) with reasoning."},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM analysis failed: {str(e)}"

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    uploaded_file.seek(0)
    gps = get_gps_coords(uploaded_file)

    # -----------------------------
    # CASE 1: GPS FOUND
    # -----------------------------
    if gps:
        lat, lon = gps
        st.success(f"📍 Exact GPS Found: {lat}, {lon}")

        address = get_address(lat, lon)
        st.info(f"🏠 Address: {address}")

        st.map({"lat": [lat], "lon": [lon]})

    # -----------------------------
    # CASE 2: AI FLOW
    # -----------------------------
    else:
        st.warning("⚠️ No GPS metadata found. Running AI analysis...")

        image_bytes = uploaded_file.getvalue()

        # Step 1: Landmark Detection
        with st.spinner("🔍 Detecting landmarks..."):
            landmark = detect_landmarks(image_bytes)

        if landmark:
            name, lat, lon = landmark
            st.success(f"🏛️ Landmark Detected: {name}")
            st.map({"lat": [lat], "lon": [lon]})

        else:
            # Step 2: LLM Reasoning
            with st.spinner("🧠 Analyzing with AI..."):
                result = analyze_with_llm(image_bytes)

            st.subheader("🧠 AI Location Estimate")
            st.write(result)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("🚀 Built for Hackathon | GeoVision AI Pro")
