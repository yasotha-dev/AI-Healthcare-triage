import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import time
from pydantic import BaseModel, Field

# ==========================================
# MODULE 1: CONTACTLESS VITALS (rPPG)
# ==========================================
class ContactlessVitals:
    def __init__(self, fps=30):
        self.fps = fps
        self.lowcut = 0.75  # 45 BPM
        self.highcut = 3.0  # 180 BPM

    def _butter_bandpass(self):
        nyq = 0.5 * self.fps
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(1, [low, high], btype='band')
        return b, a

    def extract_heart_rate(self, rgb_signals):
        """Extracts dominant pulse frequency from Green channel temporal variance."""
        if len(rgb_signals) < self.fps * 3:  # At least 3 seconds needed
            return None
        
        # Isolate Green channel (highest light absorption by hemoglobin)
        green_channel = np.array([frame[1] for frame in rgb_signals])
        
        # Normalize signal
        normalized = (green_channel - np.mean(green_channel)) / (np.std(green_channel) + 1e-6)
        
        # Apply Butterworth bandpass filter
        b, a = self._butter_bandpass()
        filtered = filtfilt(b, a, normalized)
        
        # Fast Fourier Transform (FFT)
        fft_spectrum = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), 1.0 / self.fps)
        
        # Isolate human pulse frequency window
        valid_idx = np.where((freqs >= self.lowcut) & (freqs <= self.highcut))
        if len(valid_idx[0]) == 0:
            return None
            
        peak_freq = freqs[valid_idx][np.argmax(fft_spectrum[valid_idx])]
        bpm = peak_freq * 60.0
        return round(float(bpm), 1)

# ==========================================
# MODULE 2: CLINICAL ESI TRIAGE ENGINE
# ==========================================
def evaluate_esi_level(symptoms: str, heart_rate: float):
    """
    Evaluates Emergency Severity Index (ESI) Level 1-5 based on symptoms and vitals.
    """
    text = symptoms.lower()
    
    # Critical symptom keywords
    life_threatening_keywords = ["chest pain", "unconscious", "cardiac", "stroke", "not breathing", "severe bleeding", "seizure"]
    emergent_keywords = ["dizziness", "severe", "fever", "fracture", "shortness of breath", "fainting", "head injury"]
    
    is_critical = any(kw in text for kw in life_threatening_keywords)
    is_emergent = any(kw in text for kw in emergent_keywords)
    
    # Resource estimation based on clinical intent
    resource_keywords = ["x-ray", "blood test", "stitches", "scan", "iv", "pain", "broken"]
    resource_count = sum(1 for kw in resource_keywords if kw in text)

    # ESI Decision Logic
    if is_critical or (heart_rate and (heart_rate > 130 or heart_rate < 45)):
        return 1, "🚨 CRITICAL (Level 1)", "Immediate life-saving resuscitation required.", "RED"
    elif is_emergent or (heart_rate and heart_rate > 100):
        return 2, "🟠 EMERGENT (Level 2)", "High-risk situation or severe distress. Rapid evaluation needed.", "ORANGE"
    elif resource_count >= 2:
        return 3, "🟡 URGENT (Level 3)", "Patient stable but requires multiple diagnostic resources.", "YELLOW"
    elif resource_count == 1 or len(text) > 20:
        return 4, "🟢 LESS URGENT (Level 4)", "Patient stable, requires a single diagnostic resource or procedure.", "GREEN"
    else:
        return 5, "🔵 NON-URGENT (Level 5)", "Stable patient, minor complaint requiring routine exam or prescription.", "BLUE"

# ==========================================
# MODULE 3: STREAMLIT UI & ER DASHBOARD
# ==========================================
st.set_page_config(page_title="AI Emergency Room Triage System", layout="wide", page_icon="🏥")

# Persistent Session State for Patient Queue
if "patient_queue" not in st.session_state:
    st.session_state.patient_queue = [
        {"id": "P-101", "name": "John Doe", "symptoms": "Severe crushing chest pain and sweating", "hr": 115.0, "esi": 1, "status": "Red - Immediate"},
        {"id": "P-102", "name": "Sarah Smith", "symptoms": "Minor ankle sprain while walking", "hr": 72.0, "esi": 4, "status": "Green - Waiting"},
        {"id": "P-103", "name": "Robert Brown", "symptoms": "Dizziness and high fever for 2 days", "hr": 104.0, "esi": 2, "status": "Orange - Priority"}
    ]

st.title("🏥 AI-Powered Emergency Room Triage & Vitals Engine")
st.caption("Contactless Computer Vision (rPPG) Pulse Extraction + Emergency Severity Index (ESI) Risk Scoring")

tab1, tab2 = st.tabs(["📋 Patient Intake Portal", "🩺 Doctor ER Dashboard"])

# ------------------------------------------
# TAB 1: PATIENT INTAKE PORTAL
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Patient Information & Camera Scan")
        patient_name = st.text_input("Full Name", value="Jane Doe")
        symptoms = st.text_area("Describe your primary symptoms / reason for visit:", 
                                placeholder="e.g., I have severe chest pain, shortness of breath, and nausea.",
                                height=120)
        
        run_scan = st.button("🎥 Start 5-Second Contactless Vitals Scan", use_container_width=True)
        camera_placeholder = st.empty()
        vitals_status = st.empty()
        
        measured_hr = 75.0  # Default fallback
        
        if run_scan:
            cap = cv2.VideoCapture(0)
            vitals_engine = ContactlessVitals()
            rgb_signals = []
            
            st.info("Scanning face for micro-vascular blood flow changes... Stay still.")
            progress_bar = st.progress(0)
            
            for i in range(120):  # Approx 4-5 seconds scan
                ret, frame = cap.read()
                if not ret:
                    st.error("Unable to access webcam.")
                    break
                
                # Center Region of Interest (ROI) for facial color sampling
                h, w, _ = frame.shape
                roi = frame[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
                avg_color = cv2.mean(roi)[:3]  # BGR format
                rgb_signals.append([avg_color[2], avg_color[1], avg_color[0]])  # Convert to RGB
                
                # Draw bounding box on overlay
                cv2.rectangle(frame, (int(w*0.3), int(h*0.3)), (int(w*0.7), int(h*0.7)), (0, 255, 0), 2)
                cv2.putText(frame, "Sampling Skin Signal...", (int(w*0.3), int(h*0.28)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                camera_placeholder.image(frame, channels="BGR", use_container_width=True)
                progress_bar.progress((i + 1) / 120)
                
            cap.release()
            camera_placeholder.empty()
            
            extracted_bpm = vitals_engine.extract_heart_rate(rgb_signals)
            if extracted_bpm:
                measured_hr = extracted_bpm
                vitals_status.success(f"✓ Measured Heart Rate: **{measured_hr} BPM**")
            else:
                vitals_status.warning("Camera scan noisy. Defaulting to baseline 75 BPM.")

    with col2:
        st.subheader("2. Autonomous Triage Assessment")
        
        if st.button("Run AI Triage Evaluation", type="primary", use_container_width=True):
            if not symptoms:
                st.error("Please describe symptoms before running evaluation.")
            else:
                esi_num, esi_label, reasoning, color_code = evaluate_esi_level(symptoms, measured_hr)
                
                st.markdown("---")
                st.markdown(f"### Score: **{esi_label}**")
                
                # Display reasoning box
                if esi_num <= 2:
                    st.error(f"**Clinical Alert:** {reasoning}")
                elif esi_num == 3:
                    st.warning(f"**Clinical Alert:** {reasoning}")
                else:
                    st.success(f"**Clinical Alert:** {reasoning}")

                st.markdown("### Measured Key Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Heart Rate", f"{measured_hr} BPM")
                m2.metric("ESI Level", f"Level {esi_num}")
                m3.metric("Action Needed", "IMMEDIATE" if esi_num <= 2 else "WAITING ROOM")

                # Add Patient to Doctor Queue State
                new_id = f"P-10{len(st.session_state.patient_queue) + 1}"
                st.session_state.patient_queue.append({
                    "id": new_id,
                    "name": patient_name,
                    "symptoms": symptoms,
                    "hr": measured_hr,
                    "esi": esi_num,
                    "status": f"Level {esi_num} Priority"
                })
                st.toast(f"Patient {patient_name} added to ER Doctor Queue!", icon="✅")

# ------------------------------------------
# TAB 2: LIVE ER DOCTOR DASHBOARD
# ------------------------------------------
with tab2:
    st.subheader("🏥 Live ER Queue (Sorted by ESI Severity)")
    
    if st.session_state.patient_queue:
        # Sort queue automatically by ESI severity (Level 1 highest priority)
        sorted_queue = sorted(st.session_state.patient_queue, key=lambda x: x["esi"])
        
        for p in sorted_queue:
            badge = "🚨 CRITICAL" if p["esi"] == 1 else ("🟠 HIGH RISK" if p["esi"] == 2 else "🟢 STABLE")
            
            with st.expander(f"**[{p['id']}] {p['name']}** — ESI Level {p['esi']} ({badge})"):
                c1, c2, c3 = st.columns([1, 2, 1])
                c1.write(f"**Heart Rate:** {p['hr']} BPM")
                c2.write(f"**Symptoms:** {p['symptoms']}")
                if c3.button("Mark Checked In", key=p["id"]):
                    st.session_state.patient_queue = [item for item in st.session_state.patient_queue if item["id"] != p["id"]]
                    st.rerun()
    else:
        st.info("No patients currently in the emergency queue.")
