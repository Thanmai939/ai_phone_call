import os
import json
import datetime
import requests
import time
import logging
import pytz
import re
import base64
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from services.transcribe import GoogleTranscriber, GoogleTTS
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

app = Flask(__name__)
sock = Sock(app)

# Initialize Google Cloud services for Speech and TTS
credentials_path = "gcpkeys.json"
transcriber = None
tts_client = GoogleTTS(credentials_path)

# Google Calendar Authentication Setup
SCOPES = ['https://www.googleapis.com/auth/calendar']
calendar_service = None
LOCAL_TIMEZONE = pytz.timezone('Asia/Kolkata')

# Key environment variables
REQUIRED_ENV_VARS = {
    'COHERE_API_KEY': os.getenv('COHERE_API_KEY'),
    'TWILIO_ACCOUNT_SID': os.getenv('TWILIO_ACCOUNT_SID'),
    'TWILIO_AUTH_TOKEN': os.getenv('TWILIO_AUTH_TOKEN'),
    'TWILIO_PHONE_NUMBER': os.getenv('TWILIO_PHONE_NUMBER')
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
def receive_call():
    if request.method == 'POST':
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
        
        # Use Google TTS for greeting
        greeting = "Hi, how can I help you?"
        greeting_audio = tts_client.synthesize_speech(greeting)
        xml = f"""
        <Response>
            <Play>{base64.b64encode(greeting_audio).decode()}</Play>
            <Connect>
                <Stream url='wss://{request.host}/transcribe'/>
            </Connect>
        </Response>
        """.strip()
        
        return Response(xml, mimetype='text/xml')
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
                print('Twilio Connected')
                transcriber = GoogleTranscriber(credentials_path)
                transcriber.connect()
                print('Google Speech-to-Text Connection Established')

            elif data['event'] == 'media':
                if transcriber:
                    payload = base64.b64decode(data['media']['payload'])
                    transcriber.stream(payload)

            elif data['event'] == 'stop':
                print('Twilio Disconnected')
                if transcriber:
                    transcriber.close()
                    print('Google Speech-to-Text Connection Closed')
                break

    except Exception as e:
        print(f"Error: {str(e)}")
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
