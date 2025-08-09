# YouTube Video QnA Chatbot

## Overview
This project is a YouTube Video QnA Chatbot that allows users to extract and analyze content from YouTube videos. The system downloads audio from a given YouTube video, transcribes it, stores it in the context, and enables users to query the extracted content using a selected LLM.

## Features
- **Download Audio**: Extracts audio from a YouTube video.
- **Transcription**: Uses Google Speech Recognition to transcribe audio in chunks.
- **Question Answering**: Allows users to query transcribed content using AI models.
- **Model Selection**: Provides multiple LLMs options for answering questions.

## Tech Stack
- **Python**
- **Streamlit** (for UI)
- **LangChain** (for text processing & retrieval)
- **Langgraph** (for the workflow)
- **SpeechRecognition** (for audio transcription)
- **Pydub** (for audio processing)
- **yt-dlp** (for YouTube audio extraction)
- **Groq API** (for LLM responses)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ajax1310/Youtube_Video_QnA_Bot.git

    ```

2. Create a virtual environment:
    ```bash
    conda create -n venv python=3.10
    conda activate venv
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add:
    ```plaintext
    GROQ_API_KEY=your_api_key_here
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run bot.py
    ```

2. Enter a YouTube video URL and click **Download and Transcribe**.
4. Enter a query to retrieve relevant information from the transcribed text.

## File Structure
```
├── bot.py
├── main.py              # Main app alongwith Streamlit
├── workflow.py          # Contains the langgraph workflow
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
├── files/
│   ├── audio/          # Downloaded audio files
│   ├── transcripts/    # Saved transcripts
└── README.md            # Project documentation
```

## Future Improvements
- Add support for long-duration YouTube videos.
- Improve retrieval accuracy using advanced embedding techniques.
- Enable multilingual transcription and QA support.


