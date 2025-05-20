import json
import queue
import threading
import asyncio
import logging
from openai import OpenAI
from openai.helpers import LocalAudioPlayer
import base64
import io
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAITranscriber:
    def __init__(self, OPENAI_API_KEY):
        """Initialize the OpenAI client"""
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not configured")
        logger.info("Initializing OpenAI Whisper transcription service...")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.audio_queue = queue.Queue()
        self.is_active = False
        self.current_transcript = ""
        self._setup_streaming()
        logger.info("OpenAI Whisper transcription service initialized successfully!")

    def _setup_streaming(self):
        """Setup the streaming recognition thread"""
        def streaming_recognize():
            while self.is_active:
                try:
                    chunk = self.audio_queue.get(timeout=2.0)
                    if not chunk:
                        continue
                        
                    # Create a file-like object from the audio chunk
                    audio_file = io.BytesIO(chunk)
                    
                    logger.info("Starting OpenAI Whisper transcription streaming...")
                    # Create streaming transcription
                    stream = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        stream=True
                    )
                    
                    # Process streaming events
                    for event in stream:
                        if event:
                            logger.info(f"OpenAI Whisper transcription received: {event}")
                            self.current_transcript = event

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in OpenAI Whisper transcription streaming: {e}")
                    if self.is_active:  # Only retry if still active
                        self._setup_streaming()

        self.stream_thread = threading.Thread(target=streaming_recognize)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def connect(self):
        """Start the transcription service"""
        self.is_active = True
        logger.info("OpenAI Whisper transcription service connected")

    def stream(self, audio_chunk):
        """Stream audio data to OpenAI"""
        if self.is_active and audio_chunk:
            try:
                self.audio_queue.put(audio_chunk)
            except Exception as e:
                logger.error(f"Error queueing audio chunk: {e}")

    def close(self):
        """Close the transcription service"""
        self.is_active = False
        logger.info("OpenAI Whisper transcription service disconnected")

# class OpenAITTS:
#     def __init__(self, OPENAI_API_KEY):
#         """Initialize the OpenAI client"""
#         if not OPENAI_API_KEY:
#             raise ValueError("OpenAI API key is not configured")
#         logger.info("Initializing OpenAI TTS service...")
#         self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
#         logger.info("OpenAI TTS service initialized successfully!")

#     async def synthesize_speech(self, text):
#         """Convert text to speech using OpenAI's TTS with streaming response"""
#         try:
#             logger.info("Starting OpenAI TTS synthesis...")
#             async with self.client.audio.speech.with_streaming_response.create(
#                 model="tts-1",
#                 voice="alloy",
#                 input=text,
#                 response_format="mp3"
#             ) as response:
#                 # Get the audio content from the streaming response
#                 audio_content = await response.read()
#                 logger.info("OpenAI TTS synthesis completed successfully!")
#                 return audio_content
                
#         except Exception as e:
#             logger.error(f"Error in OpenAI TTS synthesis: {e}")
#             return None 
class OpenAITTS:
    def __init__(self, OPENAI_API_KEY: str):
        """
        Initialize the OpenAI client.

        Args:
            OPENAI_API_KEY (str): Your OpenAI API key.
        
        Raises:
            ValueError: If the OpenAI API key is not configured.
        """
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not configured")
        logger.info("Initializing OpenAI TTS service...")
        # Initialize AsyncOpenAI for asynchronous operations
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI TTS service initialized successfully!")

    async def synthesize_speech(self, text: str) -> bytes | None:
        """
        Convert text to speech using OpenAI's TTS with streaming response.
        Uses the 'gpt-4o-mini-tts' model with an 'alloy' voice and an
        'eternal optimist' tone.

        Args:
            text (str): The text to convert to speech.

        Returns:
            bytes | None: The audio content in MP3 format as bytes, or None if an error occurs.
        """
        try:
            logger.info("Starting OpenAI TTS synthesis...")
            # Initiate the streaming Text-to-Speech request
            # Using 'async with' ensures the response stream is properly handled and closed.
            async with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",  # Changed model to gpt-4o-mini-tts
                voice="alloy",            # Using the 'alloy' voice
                input=text,               # The text to be synthesized
                response_format="mp3",    # Requesting the audio in MP3 format
                # Added instructions parameter to guide the tone of voice
                instructions="Speak in a cheerful and positive tone, like an eternal optimist."
            ) as response:
                # Read the entire audio content from the streaming response.
                # 'await' is used because response.read() is an asynchronous operation.
                audio_content = await response.read()
                logger.info("OpenAI TTS synthesis completed successfully with gpt-4o-mini-tts and optimist tone!")
                return audio_content

        except Exception as e:
            logger.error(f"Error in OpenAI TTS synthesis: {e}")
            return None
