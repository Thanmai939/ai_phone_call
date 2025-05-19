import os
import json
import datetime
import requests
import time
import logging
import subprocess
import signal
import pytz
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from utils import parse_date_time, extract_purpose, extract_appointment_duration
from openai import OpenAI

# Configure logging - simplified
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Key environment variables
REQUIRED_ENV_VARS = {
    'COHERE_API_KEY': os.getenv('COHERE_API_KEY'),
    'TWILIO_ACCOUNT_SID': os.getenv('TWILIO_ACCOUNT_SID'),
    'TWILIO_AUTH_TOKEN': os.getenv('TWILIO_AUTH_TOKEN'),
    'TWILIO_PHONE_NUMBER': os.getenv('TWILIO_PHONE_NUMBER')
}
missing_vars = [var for var, value in REQUIRED_ENV_VARS.items() if not value]
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
PORT = int(os.getenv("PORT", 5000))

# Initialize Twilio client
twilio_client = Client(REQUIRED_ENV_VARS['TWILIO_ACCOUNT_SID'], REQUIRED_ENV_VARS['TWILIO_AUTH_TOKEN'])

# Global variables
SCOPES = ['https://www.googleapis.com/auth/calendar']
creds = None
calendar_service = None
active_calls = {}
conversation_histories = {}
ngrok_process = None
LOCAL_TIMEZONE = pytz.timezone('Asia/Kolkata') # Define a global timezone

@app.route('/', methods=['GET'])
def index():
    ngrok_url = get_ngrok_url()
    if ngrok_url:
        return f"""
        <html>
            <body>
                <h1>AI Appointment Scheduler is running</h1>
                <h2>Your ngrok URL is:</h2>
                <p style="font-size: 20px; color: blue;">{ngrok_url}</p>
                <h2>Twilio Webhook URL:</h2>
                <p style="font-size: 20px; color: green;">{ngrok_url}/voice</p>
                <p>Use this URL to configure your Twilio phone number's webhook.</p>
            </body>
        </html>
        """
    return "AI Appointment Scheduler running"

@app.route('/status', methods=['GET'])
def status():
    active_call_count = len(active_calls)
    ngrok_url = get_ngrok_url()
    return jsonify({
        "status": "running",
        "active_calls": active_call_count,
        "ngrok_url": ngrok_url,
        "webhook_url": f"{ngrok_url}/voice" if ngrok_url else None
    })

def get_ngrok_url():
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        if response.status_code == 200:
            tunnels = response.json().get('tunnels', [])
            if tunnels and len(tunnels) > 0:
                return tunnels[0].get('public_url')
    except Exception as e:
        logger.error(f"Error getting ngrok URL: {e}")
    return None

@app.route('/voice', methods=['POST'])
def voice():
    try:
        call_sid = request.values.get('CallSid')
        logger.info(f"Received voice webhook request. CallSid: {call_sid}")
        logger.info(f"Request values: {request.values}")
        
        if not call_sid:
            logger.error("No CallSid received in request")
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

        greeting = "Hi, I'm your scheduling assistant. How can I help book or check appointments?"
        logger.info(f"Sending greeting for call {call_sid}")
        response.say(greeting, voice="Polly.Matthew", prosody={'rate': '95%'})

        gather = Gather(
            input='speech',
            action='/process_speech',
            timeout=5,
            speechTimeout='auto',
            language='en-US',
            bargeIn=True
        )

        response.append(gather)
        response.say("Didn't catch that. Goodbye.", voice="Polly.Matthew")
        response.hangup()

        logger.info(f"Returning TwiML response for call {call_sid}")
        return str(response)

    except Exception as e:
        logger.error(f"Error in voice endpoint: {e}", exc_info=True)
        response = VoiceResponse()
        response.say("Error. Try again later.", voice="Polly.Matthew")
        return str(response)

@app.route('/process_speech', methods=['POST'])
def process_speech():
    try:
        call_sid = request.values.get('CallSid')
        speech_result = request.values.get('SpeechResult')

        if not call_sid or not speech_result or call_sid not in active_calls:
            response = VoiceResponse()
            response.say("Sorry, connection error. Try again.", voice="Polly.Matthew")
            response.hangup()
            return str(response)

        logger.info(f"User: '{speech_result}'")

        ai_response = process_user_input(call_sid, speech_result)
        logger.info(f"AI: '{ai_response}'")

        active_calls[call_sid]['responses'].append({
            'user': speech_result,
            'ai': ai_response,
            'timestamp': datetime.datetime.now().isoformat()
        })
        active_calls[call_sid]['turn_count'] += 1

        response = VoiceResponse()

        sentences = re.split(r'(?<=[.!?])\s+', ai_response.strip())
        if len(sentences) > 2:
            sentences = sentences[:2]
            ai_response = " ".join(sentences) + "."

        response.say(ai_response, voice="Polly.Matthew", prosody={'rate': '95%'})

        end_keywords = ["booked", "confirmed", "appointment is", "no appointments", "goodbye"]
        should_end_call = any(keyword in ai_response.lower() for keyword in end_keywords) or active_calls[call_sid]['turn_count'] >= 8

        if should_end_call:
            response.say("Thanks. Goodbye.", voice="Polly.Matthew")
            response.hangup()
            if call_sid in active_calls: del active_calls[call_sid]
            if call_sid in conversation_histories: del conversation_histories[call_sid]
        else:
            gather = Gather(
                input='speech',
                action='/process_speech',
                timeout=5,
                speechTimeout='auto',
                language='en-US',
                bargeIn=True
            )
            response.append(gather)
            response.say("Is there anything else?", voice="Polly.Matthew", prosody={'rate': '95%'})
            response.hangup()

        return str(response)

    except Exception as e:
        logger.error(f"Error in process_speech: {e}")
        response = VoiceResponse()
        response.say("Error processing request. Try again.", voice="Polly.Matthew")
        response.hangup()
        return str(response)

def process_user_input(call_sid, text):
    try:
        if call_sid not in active_calls:
            return "Sorry, connection issue. Try again."

        call_data = active_calls[call_sid]
        conversation_chain = call_data['conversation_chain']
        history = get_session_history(call_sid).messages
        full_conversation_text = "\n".join([msg.content for msg in history] + [f"User: {text}"])

        retrieval_keywords = ["check", "when is", "what is", "do i have", "my appointment", "view appointment"]
        booking_keywords = ["schedule", "book", "make", "new appointment", "set up"]

        is_retrieval_intent = any(keyword in text.lower() for keyword in retrieval_keywords)
        is_booking_intent = any(keyword in text.lower() for keyword in booking_keywords)

        if is_retrieval_intent and not is_booking_intent:
            logger.info("Detected RETRIEVAL intent.")
            search_date = parse_date_time(text)
            time_min, time_max = None, None
            if search_date:
                start_of_day = search_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = start_of_day + datetime.timedelta(days=1)
                time_min = LOCAL_TIMEZONE.localize(start_of_day).astimezone(pytz.utc).isoformat()
                time_max = LOCAL_TIMEZONE.localize(end_of_day).astimezone(pytz.utc).isoformat()

            try:
                events = get_google_calendar_events(time_min=time_min, time_max=time_max)
                if events:
                    response_parts = []
                    for event in events[:2]:
                        event_summary = event.get('summary', 'Appointment')
                        start = event.get('start', {}).get('dateTime')
                        if start:
                            dt_local = datetime.datetime.fromisoformat(start).astimezone(LOCAL_TIMEZONE)
                            day = dt_local.strftime('%A')
                            time_str = dt_local.strftime('%I:%M %p').lstrip('0')
                            response_parts.append(f"{event_summary} on {day} at {time_str}")

                    if len(events) > 2:
                        response_parts.append("and more.")

                    return f"Found: {'; '.join(response_parts)}." if response_parts else "Couldn't find specific time for appointments."
                else:
                    date_str = f" for {search_date.strftime('%A, %B %d')}" if search_date else " soon"
                    return f"No upcoming appointments found{date_str}."
            except Exception as e:
                logger.error(f"Error retrieving calendar events: {e}")
                return "Error checking calendar. Try later?"

        else:
            logger.info("Detected BOOKING or GENERAL query intent.")

            response = conversation_chain.invoke(
                {"text": text},
                config={"configurable": {"session_id": call_sid}}
            )
            response_content = response.content

            booking_confirmation_keywords = ["schedule", "book", "appointment", "confirm"]
            potential_booking_response = any(term in response_content.lower() for term in booking_confirmation_keywords)

            if potential_booking_response or is_booking_intent:
                appointment_datetime = parse_date_time(full_conversation_text)
                purpose = extract_purpose(full_conversation_text)

                if appointment_datetime and purpose:
                    logger.info(f"Attempting to book based on conversation: {purpose} at {appointment_datetime}")
                    try:
                        duration = extract_appointment_duration(full_conversation_text)
                        calendar_result = create_google_calendar_event(purpose, appointment_datetime, duration)

                        if calendar_result and calendar_result.get('success'):
                            # Convert the time to local timezone for display
                            if appointment_datetime.tzinfo is None:
                                appointment_datetime = LOCAL_TIMEZONE.localize(appointment_datetime)
                            else:
                                appointment_datetime = appointment_datetime.astimezone(LOCAL_TIMEZONE)
                                
                            day_name = appointment_datetime.strftime('%A')
                            month_name = appointment_datetime.strftime('%B')
                            day_num = appointment_datetime.strftime('%d').lstrip('0')
                            time_str = appointment_datetime.strftime('%I:%M %p').lstrip('0')
                            return f"Booked appointment for {day_name}, {month_name} {day_num} at {time_str}."
                        else:
                            error_detail = calendar_result.get("error", "unknown error")
                            logger.error(f"Calendar booking failed: {error_detail}")
                            if "conflict" in error_detail.lower():
                                return "Time conflicts. Try another?"
                            else:
                                return "Calendar error. Try again?"
                    except Exception as e:
                        logger.error(f"Calendar booking process error: {e}")
                        return "Booking failed. Try again?"
                
                elif not appointment_datetime and purpose:
                    return "Okay, what date and time for that?"
                elif not purpose and appointment_datetime:
                    return "Got the time. What is this appointment for?"

            words = response_content.split()
            if len(words) > 15:
                response_content = " ".join(words[:15])
                if response_content[-1] not in ".!?":
                    response_content += "."

            return response_content
    
    except Exception as e:
        logger.error(f"Error in process_user_input: {e}")
        return "Sorry, internal error. Try again."

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

# GOOGLE CALENDAR INTEGRATION

def create_google_calendar_event(summary, start_time, duration_minutes=30, description=None):
    if not refresh_google_credentials():
        return {"success": False, "error": "Google Auth Failed"}

    global calendar_service
    if not calendar_service:
        try:
            calendar_service = build('calendar', 'v3', credentials=creds)
        except Exception as e:
            logger.error(f"Failed to build calendar service: {e}")
            return {"success": False, "error": f"Failed to build calendar service: {e}"}

    end_time = start_time + datetime.timedelta(minutes=duration_minutes)

    # Ensure times are in local timezone
    if start_time.tzinfo is None:
        start_time = LOCAL_TIMEZONE.localize(start_time)
    else:
        start_time = start_time.astimezone(LOCAL_TIMEZONE)
        
    if end_time.tzinfo is None:
        end_time = LOCAL_TIMEZONE.localize(end_time)
    else:
        end_time = end_time.astimezone(LOCAL_TIMEZONE)

    event = {
        'summary': 'Booked Appointment',  # Fixed summary for all appointments
        'description': f'Purpose: {summary}',  # Original summary becomes the description
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

    for attempt in range(2):
        try:
            logger.info(f"Creating event: {json.dumps(event, default=str)}")
            created_event = calendar_service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1
            ).execute()
            logger.info(f"Event created: {created_event.get('htmlLink')}")
            return {
                "success": True,
                "event_id": created_event['id'],
                "link": created_event.get('htmlLink'),
                "event_time": created_event.get('start', {}).get('dateTime')
            }

        except Exception as e:
            logger.error(f"Calendar insert error (attempt {attempt+1}): {e}")
            error_details = getattr(e, 'content', str(e))
            error_msg = str(error_details).lower()

            if 'unauthorized' in error_msg or 'auth' in error_msg or 'token' in error_msg or '401' in error_msg:
                if not refresh_google_credentials(force=True):
                    return {"success": False, "error": "Google Auth Refresh Failed"}
                try:
                    calendar_service = build('calendar', 'v3', credentials=creds)
                except Exception as build_e:
                    logger.error(f"Failed to rebuild calendar service after refresh: {build_e}")
                    return {"success": False, "error": "Failed to rebuild calendar service after refresh"}
                continue
            elif 'conflict' in error_msg or '409' in error_msg:
                return {"success": False, "error": "Time conflict detected"}
            else:
                time.sleep(1.5)

    return {"success": False, "error": "Failed to create event after retries"}

def get_google_calendar_events(time_min=None, time_max=None, max_results=5):
    if not refresh_google_credentials():
        raise ConnectionError("Google Auth Failed")

    global calendar_service
    if not calendar_service:
        try:
            calendar_service = build('calendar', 'v3', credentials=creds)
        except Exception as e:
            logger.error(f"Failed to build calendar service: {e}")
            raise ConnectionError(f"Failed to build calendar service: {e}")

    if time_min is None:
        time_min = datetime.datetime.now(pytz.utc).isoformat()

    try:
        logger.info(f"Fetching calendar events: timeMin={time_min}, timeMax={time_max}, maxResults={max_results}")
        events_result = calendar_service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        logger.info(f"Found {len(events)} events.")
        return events

    except Exception as e:
        logger.error(f"Error fetching Google Calendar events: {e}")
        error_details = getattr(e, 'content', str(e))
        error_msg = str(error_details).lower()
        if 'unauthorized' in error_msg or 'auth' in error_msg or 'token' in error_msg or '401' in error_msg:
            if refresh_google_credentials(force=True):
                try:
                    calendar_service = build('calendar', 'v3', credentials=creds)
                    events_result = calendar_service.events().list(
                        calendarId='primary', timeMin=time_min, timeMax=time_max,
                        maxResults=max_results, singleEvents=True, orderBy='startTime'
                    ).execute()
                    return events_result.get('items', [])
                except Exception as retry_e:
                    logger.error(f"Error fetching events after refresh: {retry_e}")
                    raise ConnectionError("Failed to fetch calendar events after auth refresh")
            else:
                raise ConnectionError("Google Auth Refresh Failed during event fetch")
        else:
            raise ConnectionError(f"Failed to fetch calendar events: {e}")

def refresh_google_credentials(force=False):
    global creds, calendar_service

    if creds and creds.valid and not force:
        return True

    try:
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)

            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open('token.json', 'w') as f:
                    f.write(creds.to_json())
                return True

        if os.path.exists('credentials.json'):
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as f:
                f.write(creds.to_json())
            return True
        else:
            logger.error("No credentials.json file found")
            return False

    except Exception as e:
        logger.error(f"Error refreshing credentials: {e}")
        return False

# CONVERSATION CHAIN

def create_conversation_chain(call_sid):
    client = OpenAI()

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
    - If user says "stop" or "wait", acknowledge and pause.

    EXAMPLES (Booking):
    User: "I need an appointment."
    AI: "Okay, what is the appointment for?"
    User: "A checkup."
    AI: "Got it. What day and time works?"
    User: "Tomorrow at 2pm."
    AI: "OK. Scheduling checkup for tomorrow 2pm." (Triggers booking function)

    EXAMPLES (Retrieval):
    User: "Do I have an appointment on Friday?"
    AI: "Checking for Friday appointments..." (Triggers retrieval function)
    User: "When is my next appointment?"
    AI: "Let me check your upcoming appointments..." (Triggers retrieval function)
    """

    def get_ai_response(text, history):
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        for msg in history:
            messages.append({
                "role": "assistant" if msg.type == "ai" else "user",
                "content": msg.content
            })
        
        # Add current message
        messages.append({"role": "user", "content": text})
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
                max_tokens=50,
                presence_penalty=1.0,
                frequency_penalty=1.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "Sorry, I'm having trouble processing that. Could you repeat?"

    class CustomChain:
        def __init__(self, call_sid):
            self.call_sid = call_sid
            self.history = get_session_history(call_sid)

        def invoke(self, input_data, config=None):
            text = input_data.get("text", "")
            response = get_ai_response(text, self.history.messages)
            
            # Add to history
            self.history.add_message({"type": "user", "content": text})
            self.history.add_message({"type": "ai", "content": response})
            
            return type('Response', (), {'content': response})()

    return CustomChain(call_sid)

def init_cohere_llm():
    if not REQUIRED_ENV_VARS['COHERE_API_KEY']:
        logger.error("Cohere API key not found")
        exit(1)

    llm = ChatCohere(
        model="command-r-plus",
        cohere_api_key=REQUIRED_ENV_VARS['COHERE_API_KEY'],
        temperature=0.4,
        max_tokens=50,
        presence_penalty=1.0,
        frequency_penalty=1.0
    )
    return llm

# NGROK HANDLING

def start_ngrok():
    global ngrok_process

    if NGROK_AUTH_TOKEN:
        subprocess.run(["ngrok", "config", "add-authtoken", NGROK_AUTH_TOKEN], check=True)

    ngrok_process = subprocess.Popen(
        ["ngrok", "http", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    logger.info("Starting ngrok...")
    time.sleep(3)

    try:
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            logger.info(f"ngrok URL: {ngrok_url}")
            logger.info(f"Twilio webhook URL: {ngrok_url}/voice")
            configure_twilio_webhook(ngrok_url)
        else:
            logger.error("Failed to get ngrok URL")
    except Exception as e:
        logger.error(f"Error getting ngrok URL: {e}")

def configure_twilio_webhook(ngrok_url):
    if not all([REQUIRED_ENV_VARS['TWILIO_ACCOUNT_SID'], REQUIRED_ENV_VARS['TWILIO_AUTH_TOKEN'], REQUIRED_ENV_VARS['TWILIO_PHONE_NUMBER']]):
        logger.warning(f"Twilio credentials missing. Please manually configure webhook to: {ngrok_url}/voice")
        return

    try:
        webhook_url = f"{ngrok_url}/voice"
        logger.info(f"Configuring Twilio webhook to: {webhook_url}")
        
        # First, validate the phone number format
        phone_number = REQUIRED_ENV_VARS['TWILIO_PHONE_NUMBER']
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number
            logger.info(f"Added + prefix to phone number: {phone_number}")
        
        # Get all phone numbers
        update_url = f"https://api.twilio.com/2010-04-01/Accounts/{REQUIRED_ENV_VARS['TWILIO_ACCOUNT_SID']}/IncomingPhoneNumbers.json"
        auth = (REQUIRED_ENV_VARS['TWILIO_ACCOUNT_SID'], REQUIRED_ENV_VARS['TWILIO_AUTH_TOKEN'])
        
        response = requests.get(update_url, auth=auth)
        logger.info(f"Twilio API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Failed to get phone numbers: {response.text}")
            return
            
        phone_numbers = response.json().get('incoming_phone_numbers', [])
        logger.info(f"Found {len(phone_numbers)} phone numbers")
        
        if not phone_numbers:
            logger.error("No phone numbers found in your Twilio account. Please purchase a phone number first.")
            logger.error("Go to: https://console.twilio.com > Phone Numbers > Get a phone number")
            return

        # Try to find the phone number
        found_number = None
        for phone in phone_numbers:
            if phone['phone_number'] == phone_number or phone['phone_number'] == f"+{phone_number}":
                found_number = phone
                break

        if not found_number:
            logger.error(f"Phone number {phone_number} not found in your Twilio account.")
            logger.error("Available numbers:")
            for phone in phone_numbers:
                logger.error(f"- {phone['phone_number']}")
            return

        # Update the webhook for the found number
        phone_sid = found_number['sid']
        logger.info(f"Updating webhook for phone number: {found_number['phone_number']}")
        
        update_response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{REQUIRED_ENV_VARS['TWILIO_ACCOUNT_SID']}/IncomingPhoneNumbers/{phone_sid}.json",
            data={"VoiceUrl": webhook_url},
            auth=auth
        )

        if update_response.status_code == 200:
            logger.info(f"Successfully configured Twilio webhook to: {webhook_url}")
        else:
            logger.error(f"Failed to update webhook: {update_response.text}")
            
    except Exception as e:
        logger.error(f"Error configuring Twilio webhook: {e}", exc_info=True)

def check_ngrok_installation():
    try:
        subprocess.run(["ngrok", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ngrok is not installed or not in PATH")
        return False

def cleanup():
    global ngrok_process
    if ngrok_process:
        logger.info("Stopping ngrok process...")
        ngrok_process.send_signal(signal.SIGTERM)
        try:
            ngrok_process.wait(timeout=5)
            logger.info("Ngrok process stopped.")
        except subprocess.TimeoutExpired:
            logger.warning("Ngrok process did not terminate gracefully, killing.")
            ngrok_process.kill()

def setup():
    try:
        if not check_ngrok_installation():
            raise ValueError("ngrok is not properly installed")

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Validate Twilio credentials
        if not REQUIRED_ENV_VARS['TWILIO_ACCOUNT_SID'] or not REQUIRED_ENV_VARS['TWILIO_AUTH_TOKEN']:
            raise ValueError("Twilio Account SID and Auth Token are required")
        
        if not REQUIRED_ENV_VARS['TWILIO_PHONE_NUMBER']:
            raise ValueError("Twilio Phone Number is required")
            
        # Test Twilio credentials
        try:
            account = twilio_client.api.accounts(REQUIRED_ENV_VARS['TWILIO_ACCOUNT_SID']).fetch()
            logger.info(f"Successfully connected to Twilio account: {account.friendly_name}")
        except Exception as e:
            raise ValueError(f"Invalid Twilio credentials: {str(e)}")

        refresh_google_credentials()
        start_ngrok()
        signal.signal(signal.SIGINT, lambda s, f: cleanup())

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        cleanup()
        raise

if __name__ == '__main__':
    try:
        setup()
        app.run(host='0.0.0.0', port=PORT)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        cleanup()
        raise
