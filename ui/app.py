import json
import logging
import os
import time
import base64
import requests
import streamlit as st
from sarvamai import JobStatusV1Response

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DEPLOYMENT_URL = "http://localhost:2024"
ASSISTANT_ID = "agent"


# -------------------------------------------------------------------
# Multimodal helpers
# -------------------------------------------------------------------
def encode_file_to_base64(file, mime_type: str) -> str:
    """Encode uploaded file to base64 string."""
    file_bytes = file.read()
    return base64.b64encode(file_bytes).decode("utf-8")


def build_multimodal_content(
    text: str, images: list = None, audio: bytes = None
) -> list:
    """Build multimodal content array in the format expected by the graph."""
    content = [{"type": "text", "text": text}]

    # Add images
    if images:
        for image_file in images:
            mime_type = image_file.type
            b64_data = encode_file_to_base64(image_file, mime_type)
            content.append(
                {
                    "type": "image",
                    "data": b64_data,
                    "metadata": {"filename": image_file.name},
                    "source_type": "base64",
                    "mime_type": mime_type,
                }
            )

    # Add audio
    if audio:
        b64_audio = base64.b64encode(audio).decode("utf-8")
        content.append(
            {
                "type": "audio",
                "data": b64_audio,
                "source_type": "base64",
                "mime_type": "audio/webm",  # Streamlit audio input default format
            }
        )

    return content


# -------------------------------------------------------------------
# REST helpers
# -------------------------------------------------------------------
def create_thread() -> str:
    resp = requests.post(
        f"{DEPLOYMENT_URL}/threads",
        json={},  # required, otherwise 422
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["thread_id"]


def submit_run(thread_id: str, input_data: dict) -> str:
    payload = {
        "assistant_id": ASSISTANT_ID,
        "input": input_data,
        "stream_mode": "updates",
    }
    resp = requests.post(
        f"{DEPLOYMENT_URL}/threads/{thread_id}/runs",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["run_id"]


def get_run_state(thread_id: str, run_id: str) -> dict:
    resp = requests.get(
        f"{DEPLOYMENT_URL}/threads/{thread_id}/runs/{run_id}",
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def get_thread_state(thread_id: str) -> dict:
    """Fetch thread; this is where state/final_report lives in your setup."""
    resp = requests.get(
        f"{DEPLOYMENT_URL}/threads/{thread_id}",
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def _find_key_recursive(obj, key: str):
    """DFS search for key in nested dict/list structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def extract_final_report_from_thread(thread_state: dict) -> str:
    """
    Try to extract `final_report` from the thread object.
    Your State has final_report: CombinedPlan, but it's persisted by the app,
    so we search the whole thread JSON.
    """
    final = _find_key_recursive(thread_state, "final_report")
    if final is None:
        # fallback: show entire thread state
        final = thread_state

    try:
        return "```json\n" + json.dumps(final, indent=2, default=str) + "\n```"
    except TypeError:
        return str(final)


def extract_text_from_final_report(final_report_data) -> str:
    """Extract actual text content from final_report which might be an AIMessage."""
    if isinstance(final_report_data, str):
        return final_report_data
    elif isinstance(final_report_data, dict):
        # Check for AIMessage structure
        content = final_report_data.get("content", "")
        if isinstance(content, str):
            # Remove "Issue summary: " prefix if present
            if content.startswith("Issue summary: "):
                return content.replace("Issue summary: ", "", 1)
            return content
        elif isinstance(content, list):
            # Handle list content format
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
            return " ".join(text_parts)
    return str(final_report_data)


# -------------------------------------------------------------------
# Speech-to-Text helper (SarvamAI)
# -------------------------------------------------------------------
def speech_to_text(audio_bytes: bytes, file_extension: str = ".webm") -> str | None:
    """Convert audio to text using SarvamAI STT job-based API."""
    import tempfile
    import os

    try:
        st.info("🎤 Transcribing audio...")

        from sarvamai import SarvamAI

        client = SarvamAI(
            api_subscription_key=os.getenv("SARVAM_API_KEY"),
        )

        # Create STT job
        job = client.speech_to_text_job.create_job(
            language_code="en-IN",
            model="saaras:v3",
            with_timestamps=False,
            with_diarization=False,
        )

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        try:
            # Upload audio file
            job.upload_files(file_paths=[temp_file_path])

            # Start the job
            job.start()

            # Wait for completion with progress indicator
            progress_placeholder = st.empty()
            progress_placeholder.info("⏳ Processing audio...")

            final_status: JobStatusV1Response = job.wait_until_complete()

            progress_placeholder.empty()

            if job.is_failed():
                logging.info(f"failed reason: {final_status.error_message}")
                st.error("❌ STT job failed.")
                return None

            # Get transcript - parse output files
            import json

            output_dir = tempfile.mkdtemp()
            job.download_outputs(output_dir=output_dir)

            # Look for transcript JSON file
            transcript = ""
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".json"):
                        json_path = os.path.join(root, file)
                        with open(json_path, "r") as f:
                            data = json.load(f)
                            # Extract transcript from JSON structure
                            if "transcript" in data:
                                transcript = data["transcript"]
                            elif "segments" in data:
                                # Combine segments
                                transcript = " ".join(
                                    [seg.get("text", "") for seg in data["segments"]]
                                )
                            break
                if transcript:
                    break

            # Clean up temp files
            os.unlink(temp_file_path)
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for file in files:
                    os.unlink(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(output_dir)

            if not transcript:
                st.warning("⚠️ No transcript found in output")
                return None

            st.success(f"✅ Transcribed: {transcript[:100]}...")
            return transcript

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e

    except Exception as e:
        st.error(f"STT error: {e}")
        import traceback

        st.code(traceback.format_exc())
        return None


# -------------------------------------------------------------------
# Text-to-Speech helper
# -------------------------------------------------------------------
def text_to_speech(text: str) -> list | None:
    """Convert text to audio using SarvamAI TTS."""
    from sarvamai import TextToSpeechResponse

    try:
        st.info(f"🔊 Generating speech for: {text[:100]}...")
        from sarvamai import SarvamAI

        client = SarvamAI(
            api_subscription_key=os.getenv("SARVAM_API_KEY"),
        )

        response: TextToSpeechResponse = client.text_to_speech.convert(
            text=text,
            target_language_code="en-IN",
            speaker="shubh",
            pace=1.1,
            speech_sample_rate=22050,
            enable_preprocessing=True,
            model="bulbul:v3",
            temperature=0.6,
        )

        # Debug: Check what we actually get back
        logging.info(f"Response type: {type(response)}")
        logging.info(
            f"Response dict: {response.__dict__ if hasattr(response, '__dict__') else response}"
        )
        logging.info(f"Audios type: {type(response.audios)}")
        logging.info(
            f"First audio type: {type(response.audios[0]) if response.audios else 'empty'}"
        )
        logging.info(
            f"First audio length: {len(response.audios[0]) if response.audios else 'empty'}"
        )

        return response.audios
    except Exception as e:
        st.error(f"TTS error: {e}")
        import traceback

        st.code(traceback.format_exc())
        return None


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Multimodal Voice Agent", layout="wide")

st.title("🎙️ Multimodal Voice Agent")
st.markdown("Interact with the AI agent using text, images, and voice.")

# Session state initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "last_output" not in st.session_state:
    st.session_state.last_output = ""
if "last_response_text" not in st.session_state:
    st.session_state.last_response_text = ""

# Create columns for input layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Text Input")
    issue_text = st.text_area(
        "Enter your message or question",
        height=100,
        placeholder="Type your message here, or use voice input...",
        key="text_input",
    )

with col2:
    st.subheader("🎤 Voice Input")
    audio_input = st.audio_input(
        "Record your voice", help="Click the microphone to record audio input"
    )

st.markdown("---")

# File upload section
st.subheader("📷 Image Input")
uploaded_images = st.file_uploader(
    "Upload images (optional)",
    type=["png", "jpg", "jpeg", "gif", "webp"],
    accept_multiple_files=True,
    help="You can upload multiple images for analysis",
)

# Display preview of uploaded images
if uploaded_images:
    st.image([img for img in uploaded_images], width=200)

# Audio playback of recorded input
if audio_input:
    st.audio(audio_input)

# Run button
run_button = st.button("🚀 Run", type="primary", use_container_width=True)

# Process input
if run_button and (issue_text.strip() or audio_input or uploaded_images):
    # Create thread if needed
    if st.session_state.thread_id is None:
        try:
            st.session_state.thread_id = create_thread()
            st.success(f"✅ Thread created: {st.session_state.thread_id[:8]}...")
        except Exception as e:
            st.error(f"Error creating thread: {e}")
            st.stop()

    thread_id = st.session_state.thread_id

    # Build multimodal content
    text_input = (
        issue_text.strip()
        if issue_text.strip()
        else "Please analyze the uploaded content."
    )

    # Transcribe audio input if present
    if audio_input:
        audio_bytes = audio_input.getvalue()
        transcribed_text = speech_to_text(audio_bytes)
        if transcribed_text:
            # Append transcribed text to input
            if text_input and text_input != "Please analyze the uploaded content.":
                text_input = (
                    f"{text_input}\n\n[Voice input transcribed]: {transcribed_text}"
                )
                st.session_state.text_input = transcribed_text
            else:
                text_input = transcribed_text
        else:
            st.warning("⚠️ Failed to transcribe audio, continuing with text input only")

    multimodal_content = build_multimodal_content(
        text=text_input,
        images=uploaded_images,
        audio=None,  # Don't send raw audio, we send transcribed text
    )

    # Prepare input payload
    input_payload = {
        "issue": text_input,
        "messages": [
            {
                "role": "user",
                "content": multimodal_content,
            }
        ],
    }

    # Submit run
    try:
        run_id = submit_run(thread_id, input_payload)
    except Exception as e:
        st.error(f"Error submitting run: {e}")
        st.stop()

    # Poll for results
    status_placeholder = st.empty()
    step = 0
    run_state = None

    with st.spinner("🤖 Agent is thinking..."):
        while True:
            try:
                run_state = get_run_state(thread_id, run_id)
            except Exception as e:
                st.error(f"Error polling run: {e}")
                break

            status = run_state.get("status", "unknown")
            step += 1
            status_placeholder.write(f"Status: **{status}** · Poll #{step}")

            if status in ("success", "failed", "error", "cancelled"):
                break

            time.sleep(1)

    status_placeholder.empty()

    # Display results
    if run_state is not None:
        status = run_state.get("status", "unknown")
        if status == "success":
            st.success("✅ Run completed successfully!")

            try:
                thread_state = get_thread_state(thread_id)
                final_report_raw = _find_key_recursive(thread_state, "final_report")

                # Debug: show what we found
                with st.expander("🔍 Debug: Raw final_report"):
                    st.code(json.dumps(final_report_raw, indent=2, default=str))

                # Extract text from final_report
                response_text = extract_text_from_final_report(final_report_raw)

                # Also get full state for raw view
                final_report_data = extract_final_report_from_thread(thread_state)

                # Store in session state
                st.session_state.last_output = final_report_data
                st.session_state.last_response_text = response_text

            except Exception as e:
                st.error(f"Error fetching thread state: {e}")
                import traceback

                st.code(traceback.format_exc())
                response_text = "No response generated."

            # Display response
            st.markdown("### 📝 Agent Response")
            st.markdown(response_text)

            # Text-to-speech button
            if st.button("🔊 Read Aloud", key="read_aloud"):
                with st.spinner("Generating audio..."):
                    audio_list = text_to_speech(response_text)
                    if audio_list:
                        # Sarvam TTS returns list of audio data (base64 or URLs)
                        for audio_data in audio_list:
                            # Check if it's a URL or base64 data
                            if audio_data.startswith(("http://", "https://")):
                                st.audio(audio_data)
                            elif isinstance(audio_data, str):
                                # Assume base64 encoded audio
                                import base64

                                try:
                                    audio_bytes = base64.b64decode(audio_data)
                                    st.audio(audio_bytes)
                                except Exception:
                                    # If base64 decode fails, try playing as-is
                                    st.audio(audio_data)
                            else:
                                st.audio(audio_data)
                    else:
                        st.warning("No audio generated.")

        else:
            st.error(f"❌ Run finished with status: {status}")
            with st.expander("View error details"):
                st.code(json.dumps(run_state, indent=2))

# Display last output if available
if st.session_state.last_output:
    st.markdown("---")
    st.markdown("### 📋 Previous Output")
    with st.expander("View raw output"):
        st.markdown(st.session_state.last_output)

    if st.session_state.last_response_text:
        st.markdown("### 💬 Previous Response")
        st.markdown(st.session_state.last_response_text)

        # TTS for previous response
        if st.button("🔊 Read Previous Aloud", key="read_previous_aloud"):
            with st.spinner("Generating audio..."):
                audio_list = text_to_speech(st.session_state.last_response_text)
                if audio_list:
                    for audio_data in audio_list:
                        if audio_data.startswith(("http://", "https://")):
                            st.audio(audio_data)
                        elif isinstance(audio_data, str):
                            import base64

                            try:
                                audio_bytes = base64.b64decode(audio_data)
                                st.audio(audio_bytes)
                            except Exception as e:
                                logging.warning(f"Error in audio data encoding: {e}")
                                st.audio(audio_data)
                        else:
                            st.audio(audio_data)
                else:
                    st.warning("No audio generated.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Multimodal Voice Agent • Powered by LangGraph • Supports Text, Images & Audio</small>
    </div>
    """,
    unsafe_allow_html=True,
)
