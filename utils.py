import datetime
import re
import logging
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_TIMEZONE = pytz.timezone('Asia/Kolkata')

def parse_date_time(text):
    """Extract date and time from text using regex patterns"""
    text = text.lower()
    now = datetime.datetime.now()
    
    # Common date patterns
    today_patterns = ["today", "tonight"]
    tomorrow_patterns = ["tomorrow"]
    day_patterns = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    
    # Time patterns
    time_pattern = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
    
    # Check for specific dates
    target_date = now
    
    if any(pattern in text for pattern in today_patterns):
        target_date = now
    elif any(pattern in text for pattern in tomorrow_patterns):
        target_date = now + datetime.timedelta(days=1)
    else:
        for day, offset in day_patterns.items():
            if day in text:
                days_ahead = (offset - now.weekday()) % 7
                if days_ahead == 0:  # If mentioned day is today, assume next week
                    days_ahead = 7
                target_date = now + datetime.timedelta(days=days_ahead)
                break
    
    # Extract time
    time_match = re.search(time_pattern, text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        meridian = time_match.group(3)
        
        # Handle PM times
        if meridian and meridian.lower() == "pm":
            if hour != 12:  # Only add 12 if it's not already 12 PM
                hour += 12
        # Handle AM times
        elif meridian and meridian.lower() == "am":
            if hour == 12:  # 12 AM should be 0
                hour = 0
        # No meridian specified but context suggests evening
        elif hour < 12 and ("evening" in text or "night" in text or "p.m." in text or "pm" in text):
            hour += 12
        
        target_date = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return target_date
    
    return None

def extract_purpose(text):
    """Extract appointment purpose from text"""
    text = text.lower()
    
    # Common purpose indicators
    purpose_indicators = [
        "for", "about", "regarding", "schedule", "book", "appointment"
    ]
    
    # Try to find purpose after common indicators
    for indicator in purpose_indicators:
        match = re.search(fr"{indicator}\s+(\w+(?:\s+\w+)*)", text)
        if match:
            purpose = match.group(1).strip()
            # Clean up common words that might be captured
            purpose = re.sub(r'\b(an?|the|at|on|for|about)\b', '', purpose).strip()
            if purpose:
                return purpose.title()
    
    # If no clear indicator, try to find the first noun phrase
    words = text.split()
    if len(words) >= 2:
        for i in range(len(words)-1):
            phrase = f"{words[i]} {words[i+1]}"
            if not any(word in phrase for word in ["today", "tomorrow", "am", "pm", "schedule", "book"]):
                return phrase.title()
    
    return None

def extract_appointment_duration(text):
    """Extract appointment duration in minutes from text"""
    text = text.lower()
    
    # Duration patterns
    hour_pattern = r"(\d+)\s*(?:hour|hr)"
    minute_pattern = r"(\d+)\s*(?:minute|min)"
    
    duration = 30  # Default duration
    
    # Check for hours
    hour_match = re.search(hour_pattern, text)
    if hour_match:
        hours = int(hour_match.group(1))
        duration = hours * 60
    
    # Check for minutes
    minute_match = re.search(minute_pattern, text)
    if minute_match:
        minutes = int(minute_match.group(1))
        if hour_match:
            duration += minutes
        else:
            duration = minutes
    
    return duration 
