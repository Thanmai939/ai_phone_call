import os
import json
import datetime
import requests
import time
import logging
import pytz
import re
import base64
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify, send_file
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Gather, Connect
from twilio.rest import Client
from transcribe import OpenAITranscriber, OpenAITTS
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify required environment variables
required_vars = ['OPENAI_API_KEY', 'COHERE_API_KEY', 'TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_PHONE_NUMBER']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not configured")
print("OpenAI API key is configured")

app = Flask(__name__)
sock = Sock(app)

# Initialize OpenAI services for Speech and TTS
print("Initializing OpenAI services...")
transcriber = None
tts_client = OpenAITTS(OPENAI_API_KEY)
print("OpenAI services initialized successfully!")

# Google Calendar Authentication Setup
SCOPES = ['https://www.googleapis.com/auth/calendar']
calendar_service = None
LOCAL_TIMEZONE = pytz.timezone('Asia/Kolkata')

# Key environment variables
REQUIRED_ENV_VARS = {
    'COHERE_API_KEY': os.getenv('COHERE_API_KEY'),
    'TWILIO_ACCOUNT_SID': os.getenv('TWILIO_ACCOUNT_SID'),
    'TWILIO_AUTH_TOKEN': os.getenv('TWILIO_AUTH_TOKEN'),
    'TWILIO_PHONE_NUMBER': os.getenv('TWILIO_PHONE_NUMBER'),
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
}

# Initialize Twilio client
twilio_client = Client(REQUIRED_ENV_VARS['TWILIO_ACCOUNT_SID'], REQUIRED_ENV_VARS['TWILIO_AUTH_TOKEN'])

# Global variables for call management
active_calls = {}
conversation_histories = {}

def get_calendar_service():
    """Get or refresh Google Calendar service using OAuth credentials"""
    global calendar_service
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    calendar_service = build('calendar', 'v3', credentials=creds)
    return calendar_service

class InMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self._messages = []

    def add_message(self, message):
        self._messages.append(message)

    def add_messages(self, messages):
        self._messages.extend(messages)

    def clear(self):
        self._messages = []

    @property
    def messages(self):
        return self._messages

def get_session_history(call_sid: str) -> BaseChatMessageHistory:
    if call_sid not in conversation_histories:
        conversation_histories[call_sid] = InMemoryChatMessageHistory()
    return conversation_histories[call_sid]

@app.route('/', methods=['GET', 'POST'])
async def receive_call():
    if request.method == 'POST':
        try:
            call_sid = request.values.get('CallSid')
            if not call_sid:
                response = VoiceResponse()
                response.say("Error. Please try again.", voice="Polly.Matthew")
                return str(response)
            
            response = VoiceResponse()
            chain = create_conversation_chain(call_sid)
            
            active_calls[call_sid] = {
                'conversation_chain': chain,
                'responses': [],
                'turn_count': 0,
                'start_time': datetime.datetime.now()
            }
            
            # Generate greeting audio using OpenAI TTS
            print("Generating greeting using OpenAI GPT-4o-mini-tts...")
            greeting = "Hi, how can I help you?"
            greeting_audio = await tts_client.synthesize_speech(greeting)
            if not greeting_audio:
                raise Exception("Failed to generate greeting audio")
            
            # Save the audio to a temporary file
            temp_audio_path = f"temp_audio_{call_sid}.mp3"
            with open(temp_audio_path, "wb") as f:
                f.write(greeting_audio)
            
            # Create a URL for the audio file
            audio_url = f"https://{request.host}/audio/{temp_audio_path}"
            
            # Add the audio to the response
            response.play(audio_url)
            
            # Connect to WebSocket for streaming transcription
            connect = Connect()
            connect.stream(url=f'wss://{request.host}/transcribe')
            response.append(connect)

            # Add Gather for speech input
            gather = Gather(
                input='speech',
                action='/process_speech',
                timeout=5,
                speechTimeout='auto',
                language='en-US',
                bargeIn=True
            )
            response.append(gather)

            # Add fallback message
            response.say("Didn't catch that. Goodbye.", voice="Polly.Matthew")
            response.hangup()

            # Log the TwiML response for debugging
            twiml_response = str(response)
            logger.info(f"Generated TwiML response: {twiml_response}")

            return Response(twiml_response, mimetype='text/xml')
            
        except Exception as e:
            logger.error(f"Error in receive_call: {e}")
            response = VoiceResponse()
            response.say("Sorry, there was an error. Please try again.", voice="Polly.Matthew")
            return str(response)
    else:
        return 'Server Running Successfully'

@sock.route('/transcribe')
def transcribe_websocket(ws):
    global transcriber
    try:
        while True:
            message = ws.receive()
            data = json.loads(message)
            
            if data['event'] == 'connected':
                logger.info('Twilio Connected')
                print("Initializing OpenAI GPT-4o-mini-transcribe for new connection...")
                transcriber = OpenAITranscriber(OPENAI_API_KEY)
                transcriber.connect()
                logger.info('OpenAI GPT-4o-mini-transcribe Connection Established')

            elif data['event'] == 'media':
                if transcriber:
                    payload = base64.b64decode(data['media']['payload'])
                    transcriber.stream(payload)

            elif data['event'] == 'stop':
                logger.info('Twilio Disconnected')
                if transcriber:
                    transcriber.close()
                    logger.info('OpenAI GPT-4o-mini-transcribe Connection Closed')
                break

    except Exception as e:
        logger.error(f"Error in transcribe_websocket: {e}")
        if transcriber:
            transcriber.close()

def create_conversation_chain(call_sid):
    llm = ChatCohere(
        cohere_api_key=REQUIRED_ENV_VARS['COHERE_API_KEY'],
        model_name="command"
    )
    
    system_prompt = """
    You're a concise AI scheduling assistant. Help users book OR check appointments.
    Keep responses VERY short (5-15 words). Ask clarifying questions one at a time.
    
    RULES:
    - Be extremely brief.
    - If booking: Ask for purpose, date/time. Confirm details before booking.
    - If checking: Ask clarifying questions if needed (e.g., "For what day?"). State findings clearly.
    - Don't offer times, wait for user.
    - Use keywords like "schedule", "book", "confirm" when ready to call the booking function.
    - Use keywords like "check", "view", "look up" when ready to call the retrieval function.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{text}")
    ])
    
    chain = prompt | llm
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="text",
        history_messages_key="history"
    )
    
    return chain_with_history

def create_google_calendar_event(summary, start_time, duration_minutes=30, description=None):
    """Create a calendar event using OAuth credentials"""
    try:
        service = get_calendar_service()
        end_time = start_time + datetime.timedelta(minutes=duration_minutes)
        
        if start_time.tzinfo is None:
            start_time = LOCAL_TIMEZONE.localize(start_time)
        if end_time.tzinfo is None:
            end_time = LOCAL_TIMEZONE.localize(end_time)
        
        event = {
            'summary': summary,
            'description': description or f'Appointment scheduled via AI Assistant',
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': LOCAL_TIMEZONE.zone,
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': LOCAL_TIMEZONE.zone,
            },
            'reminders': {
                'useDefault': True
            },
        }
        
        created_event = service.events().insert(
            calendarId='primary',
            body=event,
            conferenceDataVersion=1
        ).execute()
        
        return {
            "success": True,
            "event_id": created_event['id'],
            "link": created_event.get('htmlLink'),
            "event_time": created_event.get('start', {}).get('dateTime')
        }
            
    except Exception as e:
        print(f"Calendar insert error: {e}")
        return {"success": False, "error": str(e)}

def get_google_calendar_events(time_min=None, time_max=None, max_results=5):
    """Get calendar events using OAuth credentials"""
    try:
        service = get_calendar_service()
        
        if time_min is None:
            time_min = datetime.datetime.now(pytz.utc).isoformat()
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        return events_result.get('items', [])
        
    except Exception as e:
        print(f"Error fetching calendar events: {e}")
        raise ConnectionError(f"Failed to fetch calendar events: {e}")

# Import the rest of the helper functions from server_local.py
from server_local import parse_date_time, extract_purpose, extract_appointment_duration, process_user_input

@app.route('/process_speech', methods=['POST'])
@async_route
async def process_speech():
    try:
        call_sid = request.values.get('CallSid')
        speech_result = request.values.get('SpeechResult')
        transcription = request.values.get('TranscriptionText')  # Get transcription from WebSocket

        if not call_sid or (not speech_result and not transcription) or call_sid not in active_calls:
            response = VoiceResponse()
            response.say("Sorry, connection error. Try again.", voice="Polly.Matthew")
            response.hangup()
            return Response(str(response), mimetype='text/xml')

        # Use transcription if available, otherwise fall back to speech_result
        user_input = transcription if transcription else speech_result
        logger.info(f"User: '{user_input}'")

        ai_response = process_user_input(call_sid, user_input)
        logger.info(f"AI: '{ai_response}'")

        active_calls[call_sid]['responses'].append({
            'user': user_input,
            'ai': ai_response,
            'timestamp': datetime.datetime.now().isoformat()
        })
        active_calls[call_sid]['turn_count'] += 1

        response = VoiceResponse()

        sentences = re.split(r'(?<=[.!?])\s+', ai_response.strip())
        if len(sentences) > 2:
            sentences = sentences[:2]
            ai_response = " ".join(sentences) + "."

        # Generate response audio using OpenAI TTS
        print("Generating response using OpenAI GPT-4o-mini-tts...")
        response_audio = await tts_client.synthesize_speech(ai_response)
        if not response_audio:
            raise Exception("Failed to generate response audio")
            
        # Save the audio to a temporary file
        temp_audio_path = f"temp_audio_{call_sid}_response.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response_audio)
        
        # Create a URL for the audio file
        audio_url = f"https://{request.host}/audio/{temp_audio_path}"
        
        # Add the audio to the response
        response.play(audio_url)

        end_keywords = ["booked", "confirmed", "appointment is", "no appointments", "goodbye"]
        should_end_call = any(keyword in ai_response.lower() for keyword in end_keywords) or active_calls[call_sid]['turn_count'] >= 8

        if should_end_call:
            # Generate goodbye audio
            goodbye = "Thanks. Goodbye."
            goodbye_audio = await tts_client.synthesize_speech(goodbye)
            if goodbye_audio:
                goodbye_path = f"temp_audio_{call_sid}_goodbye.mp3"
                with open(goodbye_path, "wb") as f:
                    f.write(goodbye_audio)
                goodbye_url = f"https://{request.host}/audio/{goodbye_path}"
                response.play(goodbye_url)
            
            response.hangup()
            if call_sid in active_calls: del active_calls[call_sid]
            if call_sid in conversation_histories: del conversation_histories[call_sid]
        else:
            # Connect to WebSocket for streaming transcription
            connect = Connect()
            connect.stream(url=f'wss://{request.host}/transcribe')
            response.append(connect)

            gather = Gather(
                input='speech',
                action='/process_speech',
                timeout=5,
                speechTimeout='auto',
                language='en-US',
                bargeIn=True
            )
            response.append(gather)
            
            # Generate followup audio
            followup = "Is there anything else?"
            followup_audio = await tts_client.synthesize_speech(followup)
            if followup_audio:
                followup_path = f"temp_audio_{call_sid}_followup.mp3"
                with open(followup_path, "wb") as f:
                    f.write(followup_audio)
                followup_url = f"https://{request.host}/audio/{followup_path}"
                response.play(followup_url)
            
            response.hangup()

        # Log the TwiML response for debugging
        twiml_response = str(response)
        logger.info(f"Generated TwiML response: {twiml_response}")

        return Response(twiml_response, mimetype='text/xml')

    except Exception as e:
        logger.error(f"Error in process_speech: {e}")
        response = VoiceResponse()
        response.say("Error processing request. Try again.", voice="Polly.Matthew")
        response.hangup()
        return Response(str(response), mimetype='text/xml')

# Add a route to serve audio files
@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_file(filename, mimetype='audio/mpeg')
