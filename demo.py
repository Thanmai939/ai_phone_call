# Assuming you have an instance of OpenAITTS and an async environment
import asyncio
import os
import subprocess
import time
import requests
from transcribe import OpenAITTS
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from dotenv import load_dotenv
import base64
from flask import Flask, request, Response, send_file
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

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

def start_ngrok():
    try:
        # Check if ngrok is already running
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            logger.info(f"ngrok is already running at: {ngrok_url}")
            return ngrok_url

        # Start ngrok
        logger.info("Starting ngrok...")
        subprocess.Popen(
            ["ngrok", "http", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for ngrok to start
        time.sleep(3)
        
        # Get the ngrok URL
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            logger.info(f"ngrok started at: {ngrok_url}")
            return ngrok_url
        else:
            raise Exception("Failed to get ngrok URL")
            
    except Exception as e:
        logger.error(f"Error starting ngrok: {e}")
        return None

# Initialize Twilio client
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    logger.error("Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER environment variables.")
    exit()

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize OpenAI TTS
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Please set the OPENAI_API_KEY environment variable.")
    exit()

tts_service = OpenAITTS(api_key)

@app.route('/voice', methods=['POST'])
async def voice():
    try:
        # Generate greeting audio
        greeting = "Hello, thank you for calling. This is a test message. Goodbye."
        logger.info(f"Generating greeting audio for: '{greeting}'")
        
        audio_content = await tts_service.synthesize_speech(greeting)
        if not audio_content:
            raise Exception("Failed to generate greeting audio")

        # Save the audio to a temporary file
        temp_audio_path = "greeting_audio.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_content)
        logger.info(f"Saved audio to {temp_audio_path}")

        # Create TwiML response
        response = VoiceResponse()
        response.play(f"{request.host_url}audio/{temp_audio_path}")
        response.hangup()

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        logger.error(f"Error in voice endpoint: {e}")
        response = VoiceResponse()
        response.say("Sorry, there was an error. Please try again.", voice="Polly.Matthew")
        return Response(str(response), mimetype='text/xml')

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    try:
        return send_file(filename, mimetype='audio/mpeg')
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {e}")
        return Response("Error serving audio file", status=500)

def configure_twilio_webhook(ngrok_url):
    try:
        webhook_url = f"{ngrok_url}/voice"
        logger.info(f"Configuring Twilio webhook to: {webhook_url}")
        
        # Get all phone numbers
        update_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/IncomingPhoneNumbers.json"
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        response = requests.get(update_url, auth=auth)
        logger.info(f"Twilio API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Failed to get phone numbers: {response.text}")
            return
            
        phone_numbers = response.json().get('incoming_phone_numbers', [])
        logger.info(f"Found {len(phone_numbers)} phone numbers")
        
        if not phone_numbers:
            logger.error("No phone numbers found in your Twilio account.")
            return

        # Try to find the phone number
        found_number = None
        for phone in phone_numbers:
            if phone['phone_number'] == TWILIO_PHONE_NUMBER or phone['phone_number'] == f"+{TWILIO_PHONE_NUMBER}":
                found_number = phone
                break

        if not found_number:
            logger.error(f"Phone number {TWILIO_PHONE_NUMBER} not found in your Twilio account.")
            return

        # Update the webhook for the found number
        phone_sid = found_number['sid']
        logger.info(f"Updating webhook for phone number: {found_number['phone_number']}")
        
        update_response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/IncomingPhoneNumbers/{phone_sid}.json",
            data={"VoiceUrl": webhook_url},
            auth=auth
        )

        if update_response.status_code == 200:
            logger.info(f"Successfully configured Twilio webhook to: {webhook_url}")
        else:
            logger.error(f"Failed to update webhook: {update_response.text}")
            
    except Exception as e:
        logger.error(f"Error configuring Twilio webhook: {e}")

if __name__ == "__main__":
    # Start ngrok and get URL
    ngrok_url = start_ngrok()
    if not ngrok_url:
        logger.error("Failed to start ngrok")
        exit()

    # Configure Twilio webhook
    configure_twilio_webhook(ngrok_url)
    
    # Start Flask app
    logger.info(f"Starting Flask app on port 5000...")
    app.run(port=5000, debug=True)
