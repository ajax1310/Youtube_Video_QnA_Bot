import os
import glob
import re
import streamlit as st
from workflow import create_app  
import speech_recognition as sr
from pydub import AudioSegment
import yt_dlp
from dotenv import load_dotenv
import time
import string
from langchain_core.messages import HumanMessage

load_dotenv()

def init_session_state():
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = {}
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'app' not in st.session_state:
        st.session_state.app = None  # LangGraph app instance

def get_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def sanitize_filename(name):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in name if c in valid_chars)

def get_audio_from_youtube(youtube_url):
    if not youtube_url:
        st.error("Please enter a YouTube URL")
        return None

    output_dir = "files/audio/"
    os.makedirs(output_dir, exist_ok=True)

    try:
        with yt_dlp.YoutubeDL({
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        }) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            raw_title = info['title']
            clean_title = sanitize_filename(raw_title)
            filename = os.path.join(output_dir, f"{clean_title}.mp3")

            current_path = glob.glob(os.path.join(output_dir, "*.mp3"))[-1]
            if current_path != filename:
                os.rename(current_path, filename)

            return filename
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

def transcribe_audio_chunk(chunk, recognizer):
    try:
        temp_wav = "temp_chunk.wav"
        chunk.export(temp_wav, format="wav")
        with sr.AudioFile(temp_wav) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except Exception as e:
        st.error(f"Error transcribing chunk: {str(e)}")
        return ""
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        st.error(f"Audio file not found: {audio_path}")
        return None
    try:
        st.write("Loading audio file...")
        sound = AudioSegment.from_mp3(audio_path)
        chunk_length_ms = 30000
        chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]
        recognizer = sr.Recognizer()
        full_transcript = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, chunk in enumerate(chunks):
            status_text.text(f"Transcribing chunk {i+1}/{len(chunks)}")
            transcript_chunk = transcribe_audio_chunk(chunk, recognizer)
            full_transcript.append(transcript_chunk)
            progress_bar.progress((i + 1) / len(chunks))

        status_text.text("Transcription completed!")
        progress_bar.empty()
        return " ".join(full_transcript)
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def save_transcript(audio_filename, transcript):
    if not transcript:
        return None
    output_dir = 'files/transcripts'
    os.makedirs(output_dir, exist_ok=True)
    transcript_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_filename))[0]}.txt")
    try:
        with open(transcript_filename, 'w', encoding='utf-8') as file:
            file.write(transcript)
        return transcript_filename
    except Exception as e:
        st.error(f"Error saving transcript: {str(e)}")
        return None

def main():
    st.title("YouTube Video QnA ChatBot")
    init_session_state()

    youtube_url = st.text_input("Enter YouTube video URL:")
    video_id = get_video_id(youtube_url)

    if st.button("Download and Transcribe"):
        if not video_id:
            st.error("Invalid YouTube URL format.")
            return

        if video_id in st.session_state.processed_videos:
            st.info("This video has already been processed.")
            st.session_state.transcription = st.session_state.processed_videos[video_id]['transcript']
            st.session_state.audio_processed = True
            transcript_path = st.session_state.processed_videos[video_id]['transcript_path']
        else:
            with st.spinner("Processing video..."):
                audio_filename = get_audio_from_youtube(youtube_url)
                if audio_filename:
                    st.info("Audio downloaded successfully. Starting transcription...")
                    transcription = transcribe_audio(audio_filename)
                    if transcription:
                        transcript_path = save_transcript(audio_filename, transcription)
                        st.session_state.transcription = transcription
                        st.session_state.audio_processed = True
                        st.session_state.processed_videos[video_id] = {
                            'audio': audio_filename,
                            'transcript': transcription,
                            'transcript_path': transcript_path
                        }
                        st.success("Audio processed and transcribed successfully!")
                        with st.expander("View Transcription Preview"):
                            st.write(transcription[:500] + "...")

       
        st.session_state.app = create_app(transcript_path)    

    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            role = "user" if msg.type == "human" else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

    # Chat input field
    if prompt := st.chat_input("Ask something about the video..."):
        # Display user message in chat bubble
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process AI response
        if st.session_state.app:
            with st.spinner("Processing your query..."):
                state = {"messages": st.session_state.messages}
                state = st.session_state.app.invoke(state)
            ai_response = state["messages"][-1].content
            st.session_state.messages.append(state["messages"][-1])

            # Display AI message in chat bubble
            with st.chat_message("assistant"):
                st.markdown(ai_response)

if __name__ == "__main__":
    main()
