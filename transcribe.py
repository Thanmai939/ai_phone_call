from google.cloud import speech
import json
import queue
import threading
from google.cloud import texttospeech

class GoogleTranscriber:
    def __init__(self, credentials_path):
        """Initialize the Google Cloud Speech-to-Text client"""
        self.client = speech.SpeechClient.from_service_account_json(credentials_path)
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
                sample_rate_hertz=8000,
                language_code="en-US",
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )
        self.audio_queue = queue.Queue()
        self.is_active = False
        self.current_transcript = ""
        self._setup_streaming()

    def _setup_streaming(self):
        """Setup the streaming recognition thread"""
        def streaming_recognize():
            while self.is_active:
                # The audio stream to send to Google Cloud Speech-to-Text
                def audio_generator():
                    while self.is_active:
                        try:
                            chunk = self.audio_queue.get(timeout=2.0)
                            yield speech.StreamingRecognizeRequest(audio_content=chunk)
                        except queue.Empty:
                            continue

                try:
                    requests = audio_generator()
                    responses = self.client.streaming_recognize(
                        config=self.streaming_config,
                        requests=requests
                    )

                    for response in responses:
                        if not response.results:
                            continue

                        result = response.results[0]
                        if result.is_final:
                            self.current_transcript = result.alternatives[0].transcript
                            print(f"Final transcript: {self.current_transcript}")

                except Exception as e:
                    print(f"Error in streaming recognition: {e}")
                    if self.is_active:  # Only retry if still active
                        self._setup_streaming()

        self.stream_thread = threading.Thread(target=streaming_recognize)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def connect(self):
        """Start the transcription service"""
        self.is_active = True
        print("Google Speech-to-Text service connected")

    def stream(self, audio_chunk):
        """Stream audio data to Google Cloud Speech-to-Text"""
        if self.is_active:
            self.audio_queue.put(audio_chunk)

    def close(self):
        """Close the transcription service"""
        self.is_active = False
        print("Google Speech-to-Text service disconnected")

class GoogleTTS:
    def __init__(self, credentials_path):
        """Initialize the Google Cloud Text-to-Speech client"""
        self.client = texttospeech.TextToSpeechClient.from_service_account_json(credentials_path)

    def synthesize_speech(self, text):
        """Convert text to speech"""
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Standard-I",  # You can change this to other available voices
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )

        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        return response.audio_content