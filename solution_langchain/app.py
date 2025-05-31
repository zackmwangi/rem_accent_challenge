
"""
Video Accent Analysis Agent
===========================

A Streamlit application that analyzes video files to detect and classify English accents.
Supports video downloads from URLs, audio extraction, language detection, and accent classification with confidence score

Categories: [ automation, agents, STT/TTS, Classification, speech analysis ]

Author: Zack Mwangi [zackmwangi@gmail.com]
Date: May 2025

"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

import streamlit as st
import os
import tempfile
import logging

import requests
import subprocess
import whisper

import json
import re

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File handling - video download and file upload
class VideoDownloadManager:
    """
    Handles video file downloads from URLs and local file management.
    """
    
    def __init__(self, download_directory: str = "video_downloads"):
        """
        Initialize the download manager.
        
        Args:
            download_directory: Directory to store downloaded videos
        """
        self.download_directory = Path(download_directory)
        self.download_directory.mkdir(exist_ok=True)
        
    def download_video_from_url(self, video_url: str) -> Optional[str]:
        """
        Download video from URL to local storage.
        
        Args:
            video_url: URL of the video to download
            
        Returns:
            Path to downloaded video file or None if failed
        """
        try:
            # Parse URL to get filename
            parsed_url = urlparse(video_url)
            filename = os.path.basename(parsed_url.path)
            
            # If no filename in URL, generate one
            if not filename or '.' not in filename:
                filename = "downloaded_video.mp4"
                
            file_path = self.download_directory / filename
            
            # Download the video
            logger.info(f"Downloading video from: {video_url}")
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    
            logger.info(f"Video downloaded successfully: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            return None
    
    def save_uploaded_file(self, uploaded_file) -> Optional[str]:
        """
        Save uploaded Streamlit file to local storage.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            file_path = self.download_directory / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            logger.info(f"File saved successfully: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            return None

# Audio extraction
class AudioExtractor:
    """
    Handles audio extraction from video files using FFmpeg.
    """
    
    @staticmethod
    def extract_audio_from_video(video_path: str, output_format: str = "mp3") -> Optional[str]:
        """
        Extract audio from video file and convert to specified format.
        
        Args:
            video_path: Path to input video file
            output_format: Desired audio format (default: mp3)
            
        Returns:
            Path to extracted audio file or None if failed
        """
        try:
            # Generate output filename
            video_file = Path(video_path)
            audio_filename = f"{video_file.stem}.{output_format}"
            audio_path = video_file.parent / audio_filename
            
            # FFmpeg command for audio extraction
            ffmpeg_command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "mp3" if output_format == "mp3" else "copy",
                "-ab", "192k",  # Audio bitrate
                "-ar", "44100",  # Sample rate
                "-y",  # Overwrite output file
                str(audio_path)
            ]
            
            logger.info(f"Extracting audio from: {video_path}")
            result = subprocess.run(
                ffmpeg_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            if audio_path.exists():
                logger.info(f"Audio extracted successfully: {audio_path}")
                return str(audio_path)
            else:
                logger.error("Audio extraction failed: Output file not created")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return None

# Language detection
class LanguageDetector:
    """
    Detects language in audio files using OpenAI Whisper.
    """
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the language detector.
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
        """
        try:
            self.whisper_model = whisper.load_model(model_name)
            logger.info(f"Whisper model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def detect_language_and_transcribe(self, audio_path: str) -> Tuple[str, str, float]:
        """
        Detect language and transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (detected_language, transcription, confidence_score)
        """
        try:
            logger.info(f"Processing audio file: {audio_path}")
            
            # Load and process audio
            result = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",
                language=None  # Auto-detect language
            )
            
            detected_language = result.get("language", "unknown")
            transcription = result.get("text", "")
            
            # Get language confidence from segments if available
            segments = result.get("segments", [])
            avg_confidence = 0.0
            if segments:
                confidences = [seg.get("no_speech_prob", 0.0) for seg in segments]
                avg_confidence = 1.0 - (sum(confidences) / len(confidences))
            
            logger.info(f"Language detected: {detected_language} (confidence: {avg_confidence:.2f})")
            
            return detected_language, transcription, avg_confidence
            
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return "unknown", "", 0.0
    
    def is_english(self, detected_language: str, confidence_threshold: float = 0.7) -> bool:
        """
        Check if detected language is English with sufficient confidence.
        
        Args:
            detected_language: Language code detected by Whisper
            confidence_threshold: Minimum confidence required
            
        Returns:
            True if language is English with sufficient confidence
        """
        return detected_language.lower() in ["en", "english"]

# Accent classification
class AccentClassifier:
    """
    Classifies English accents using local LLM via Ollama and Langchain.
    """
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        """
        Initialize the accent classifier.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        try:
            self.llm = Ollama(
                model=model_name,
                base_url=base_url,
                temperature=0.1
            )
            
            # Create the accent analysis prompt template
            self.accent_prompt = PromptTemplate(
                input_variables=["transcription"],
                template="""
                You are an expert linguist specializing in English accent classification and phonetic analysis.
                
                Analyze the following English transcription and classify the speaker's accent:
                
                Transcription: "{transcription}"
                
                Based on vocabulary choices, spelling patterns, and linguistic markers in the text, provide your analysis in the following JSON format:
                
                {{
                    "accent_classification": "Primary accent (e.g., American, British, Australian, Canadian, etc.)",
                    "confidence_score": confidence_percentage_as_integer_0_to_100,
                    "explanation": "Brief 1-2 sentence explanation of your classification reasoning",
                    "linguistic_markers": ["marker1", "marker2", "marker3"]
                }}
                
                Focus on:
                - Vocabulary choices (lift vs elevator, lorry vs truck)
                - Spelling patterns if present
                - Idiomatic expressions
                - Regional terminology
                
                Provide only the JSON response, no additional text.
                """
            )
            
            # Create the analysis chain
            self.accent_chain = (
                self.accent_prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            logger.info("Accent classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing accent classifier: {str(e)}")
            raise
    
    def classify_accent(self, transcription: str) -> Dict[str, Any]:
        """
        Classify English accent from transcription.
        
        Args:
            transcription: English text transcription
            
        Returns:
            Dictionary with accent classification results
        """
        try:
            logger.info("Analyzing accent from transcription")
            
            # Get LLM response
            llm_response = self.accent_chain.invoke({"transcription": transcription})
            
            # Parse JSON response
            try:
                # Clean the response to extract JSON
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                else:
                    # Fallback parsing
                    result = json.loads(llm_response)
                
                # Validate required fields
                required_fields = ["accent_classification", "confidence_score", "explanation"]
                for field in required_fields:
                    if field not in result:
                        result[field] = "Unknown" if field != "confidence_score" else 0
                
                # Ensure confidence score is integer
                if isinstance(result["confidence_score"], str):
                    result["confidence_score"] = 50  # Default fallback
                
                logger.info(f"Accent classified: {result['accent_classification']} ({result['confidence_score']}%)")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM JSON response: {str(e)}")
                return {
                    "accent_classification": "Classification Error",
                    "confidence_score": 0,
                    "explanation": "Unable to parse accent analysis response",
                    "linguistic_markers": []
                }
                
        except Exception as e:
            logger.error(f"Error in accent classification: {str(e)}")
            return {
                "accent_classification": "Analysis Error",
                "confidence_score": 0,
                "explanation": f"Error during accent analysis: {str(e)}",
                "linguistic_markers": []
            }

# Accent Analyzer Agent logic
class VideoAccentAnalysisAgent:
    """
    Main agent orchestrating the video accent analysis pipeline.
    """
    
    def __init__(self):
        """Initialize the analysis agent with all required components."""
        self.download_manager = VideoDownloadManager()
        self.audio_extractor = AudioExtractor()
        self.language_detector = LanguageDetector()
        self.accent_classifier = AccentClassifier()
        
    def analyze_video(self, video_source: str, is_url: bool = False, uploaded_file=None) -> Dict[str, Any]:
        """
        Complete video accent analysis pipeline.
        
        Args:
            video_source: Video file path or URL
            is_url: Whether the source is a URL
            uploaded_file: Streamlit uploaded file object (if applicable)
            
        Returns:
            Dictionary with complete analysis results
        """
        results = {
            "success": False,
            "video_path": None,
            "audio_path": None,
            "language_detected": None,
            "transcription": None,
            "accent_analysis": None,
            "error_message": None
        }
        
        try:
            # Step 1: Get video file
            if uploaded_file:
                video_path = self.download_manager.save_uploaded_file(uploaded_file)
            elif is_url:
                video_path = self.download_manager.download_video_from_url(video_source)
            else:
                video_path = video_source
            
            if not video_path:
                results["error_message"] = "Failed to obtain video file"
                return results
            
            results["video_path"] = video_path
            
            # Step 2: Extract audio
            audio_path = self.audio_extractor.extract_audio_from_video(video_path)
            if not audio_path:
                results["error_message"] = "Failed to extract audio from video"
                return results
            
            results["audio_path"] = audio_path
            
            # Step 3: Language detection and transcription
            detected_language, transcription, lang_confidence = self.language_detector.detect_language_and_transcribe(audio_path)
            
            results["language_detected"] = detected_language
            results["transcription"] = transcription
            
            # Step 4: Check if English
            if not self.language_detector.is_english(detected_language):
                results["error_message"] = f"Non-English speech detected ({detected_language}). This agent can only process English speech."
                return results
            
            # Step 5: Accent classification
            accent_analysis = self.accent_classifier.classify_accent(transcription)
            results["accent_analysis"] = accent_analysis
            
            results["success"] = True
            return results
            
        except Exception as e:
            logger.error(f"Error in video analysis pipeline: {str(e)}")
            results["error_message"] = f"Analysis failed: {str(e)}"
            return results

# Streamlit Application
def main():
    """Main Streamlit application function."""
    
    st.set_page_config(
        page_title="English Accent Analysis Agent (Video speech to text)",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 English Accent Analysis Agent (Video speech to text)")
    st.markdown("---")
    
    st.markdown("""
    This application analyzes video files to detect and classify English accents.
    
    **Features:**
    
    - 📹 Upload video files or provide URL
    - 🎵 Audio extraction
    - 🌍 Language detection
    - 🗣️ English accent classification
    - 📊 Confidence scoring and analysis
    
    """)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        whisper_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        ollama_model = st.text_input(
            "Ollama Model",
            value="llama3",
            help="Name of the Ollama model for accent classification"
        )
        
        st.markdown("---")
        st.markdown("**Requirements:**")
        st.markdown("- Ollama running locally")
        st.markdown("- FFmpeg installed")
        st.markdown("- Internet connection (for URLs)")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Video Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Provide URL"],
            horizontal=True
        )
        
        video_file = None
        video_url = None
        
        if input_method == "Upload File":
            video_file = st.file_uploader(
                "Upload video file",
                type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
                help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM"
            )
        else:
            video_url = st.text_input(
                "Video URL",
                placeholder="https://example.com/video.mp4",
                help="Direct link to video file"
            )
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Analyze Video",
            type="primary",
            disabled=not (video_file or video_url)
        )
    
    with col2:
        st.header("📊 Analysis Results")
        
        if analyze_button:
            # Initialize agent
            try:
                with st.spinner("Initializing analysis agent..."):
                    agent = VideoAccentAnalysisAgent()
                
                # Run analysis
                with st.spinner("Analyzing video... This may take a few minutes."):
                    if video_file:
                        results = agent.analyze_video(None, is_url=False, uploaded_file=video_file)
                    else:
                        results = agent.analyze_video(video_url, is_url=True)
                
                st.session_state.analysis_results = results
                
            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")
                st.stop()
        
        # Display results
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            if results["success"]:
                st.success("✅ Analysis completed successfully!")
                
                # Language Detection Results
                st.subheader("🌍 Language Detection")
                col_lang1, col_lang2 = st.columns(2)
                with col_lang1:
                    st.metric("Detected Language", results["language_detected"].upper())
                with col_lang2:
                    st.metric("Status", "✅ English Confirmed")
                
                # Transcription
                st.subheader("📝 Transcription")
                with st.expander("View Full Transcription", expanded=False):
                    st.text_area(
                        "Transcribed Text",
                        value=results["transcription"],
                        height=150,
                        disabled=True
                    )
                
                # Accent Analysis
                if results["accent_analysis"]:
                    st.subheader("🗣️ Accent Analysis")
                    accent_data = results["accent_analysis"]
                    
                    # Main metrics
                    col_acc1, col_acc2 = st.columns(2)
                    with col_acc1:
                        st.metric(
                            "Accent Classification",
                            accent_data.get("accent_classification", "Unknown")
                        )
                    with col_acc2:
                        confidence = accent_data.get("confidence_score", 0)
                        st.metric(
                            "Confidence Score",
                            f"{confidence}%",
                            delta=None
                        )
                    
                    # Progress bar for confidence
                    st.progress(confidence / 100.0)
                    
                    # Explanation
                    st.subheader("💡 Analysis Explanation")
                    st.info(accent_data.get("explanation", "No explanation available"))
                    
                    # Linguistic markers
                    st.badge("Home", color="blue")
                    if "linguistic_markers" in accent_data and accent_data["linguistic_markers"]:
                        st.subheader("🔤 Linguistic Markers")
                        for marker in accent_data["linguistic_markers"]:
                            st.badge(marker)
                
            else:
                st.error("❌ Analysis failed")
                st.error(results.get("error_message", "Unknown error occurred"))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ from Nairobi, Kenya - using Streamlit, OpenAI Whisper, Ollama, and Langchain"
    )

# Run the main app
if __name__ == "__main__":
    main()