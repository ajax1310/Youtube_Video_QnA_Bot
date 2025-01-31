import os
import glob
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import speech_recognition as sr
from pydub import AudioSegment
import yt_dlp
from dotenv import load_dotenv
import time

def init_session_state():
    """Initialize session state variables"""
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    if 'vectors' not in st.session_state:
        st.session_state.vectors = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

llm = ChatGroq(
    api_key=groq_api_key,  
    model_name='deepseek-r1-distill-llama-70b'  
)

prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only. 
        Provide the most accurate response based on the question.
        <context>
        {context}
        </context>
        Question: {input}
    """)

def get_audio_from_youtube(youtube_url):
    """Download audio from YouTube URL """
    if not youtube_url:
        st.error("Please enter a YouTube URL")
        return None
        
    output_dir = "files/audio/"
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_config) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            return os.path.join(output_dir, f"{info['title']}.mp3")
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

def transcribe_audio_chunk(chunk, recognizer):
    """Transcribe a single chunk of audio"""
    try:
        temp_wav = "temp_chunk.wav"
        chunk.export(temp_wav, format="wav")
        
        with sr.AudioFile(temp_wav) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        st.error(f"Error transcribing chunk: {str(e)}")
        return ""
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def transcribe_audio(audio_path):
    """Transcribe audio file with chunking and progress tracking"""
    if not os.path.exists(audio_path):
        st.error(f"Audio file not found: {audio_path}")
        return None
        
    try:
        st.write("Loading audio file...")
        sound = AudioSegment.from_mp3(audio_path)
        
        # Split audio into 30-second chunks
        chunk_length_ms = 30000  # 30 seconds
        chunks = [sound[i:i + chunk_length_ms] 
                 for i in range(0, len(sound), chunk_length_ms)]
        
        recognizer = sr.Recognizer()
        full_transcript = []
        
        # Create a progress bar
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
    """Save transcript to file"""
    if not transcript:
        return None
        
    output_dir = 'files/transcripts'
    os.makedirs(output_dir, exist_ok=True)
    
    transcript_filename = os.path.join(
        output_dir, 
        f"{os.path.splitext(os.path.basename(audio_filename))[0]}.txt"
    )
    
    try:
        with open(transcript_filename, 'w', encoding='utf-8') as file:
            file.write(transcript)
        return transcript_filename
    except Exception as e:
        st.error(f"Error saving transcript: {str(e)}")
        return None

def create_vector_embeddings(transcription):
    """Create vector embeddings from transcription"""
    if not transcription:
        st.error("No transcription available to create embeddings")
        return None
        
    try:
        embeddings = OllamaEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        final_documents = text_splitter.split_text(transcription)
        return FAISS.from_texts(final_documents, embeddings)
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def main():
    st.title("YouTube Video QA System")
    init_session_state()    
    
    # Download & Transcribe
    youtube_url = st.text_input("Enter YouTube video URL:")
    
    if st.button("Download and Transcribe"):
        with st.spinner("Processing video..."):
            audio_filename = get_audio_from_youtube(youtube_url)
            if audio_filename:
                st.info("Audio downloaded successfully. Starting transcription...")
                st.session_state.transcription = transcribe_audio(audio_filename)
                if st.session_state.transcription:
                    save_transcript(audio_filename, st.session_state.transcription)
                    st.session_state.audio_processed = True
                    st.success("Audio processed and transcribed successfully!")
                    # Display a preview of the transcription
                    with st.expander("View Transcription Preview"):
                        st.write(st.session_state.transcription[:500] + "...")
    
    # Create embeddings
    if st.session_state.audio_processed and st.button('Create Document Embeddings'):
        with st.spinner("Creating embeddings..."):
            st.session_state.vectors = create_vector_embeddings(st.session_state.transcription)
            if st.session_state.vectors:
                st.success("Vector database is ready!")
    
    # Query handling
    user_prompt = st.text_input("Enter your query about the video content:")
    
    if user_prompt and st.session_state.vectors:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            with st.spinner("Processing your query..."):
                start_time = time.process_time()
                response = retrieval_chain.invoke({'input': user_prompt})
                process_time = time.process_time() - start_time
                
                st.write("Answer:", response['answer'])
                st.info(f"Processing time: {process_time:.2f} seconds")
                
                with st.expander("View Related Context"):
                    for i, doc in enumerate(response['context'], 1):
                        st.write(f"Context {i}:", doc.page_content)
                        st.divider()
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":  
    main()