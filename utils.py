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
    
    # Month patterns
    month_patterns = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    # Time patterns - updated to handle more formats including p.m. with periods
    time_pattern = r"(\d{1,2})(?::(\d{2}))?\s*(?:a\.?m\.?|p\.?m\.?|am|pm)?"
    
    # Check for specific dates
    target_date = None
    
    # Try to match explicit date format (e.g., "May 29th, 2025" or "May 29, 2025")
    date_pattern = r"(?:(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4}))"
    date_match = re.search(date_pattern, text)
    
    if date_match:
        month_name = date_match.group(1).lower()
        day = int(date_match.group(2).replace(",", ""))
        year = int(date_match.group(3))
        
        if month_name in month_patterns:
            month = month_patterns[month_name]
            try:
                target_date = datetime.datetime(year, month, day)
            except ValueError:
                logger.error(f"Invalid date: {year}-{month}-{day}")
                return None
    elif any(pattern in text for pattern in today_patterns):
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
    
    if target_date is None:
        return None
    
    # Extract time
    time_match = re.search(time_pattern, text)
    if time_match:
        try:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            
            # Check for PM in the text (including variations)
            is_pm = any(pm in text for pm in ["p.m.", "pm", "p.m"])
            is_am = any(am in text for am in ["a.m.", "am", "a.m"])
            
            # Handle PM times
            if is_pm:
                if hour != 12:  # Only add 12 if it's not already 12 PM
                    hour = (hour + 12) % 24
            # Handle AM times
            elif is_am:
                if hour == 12:  # 12 AM should be 0
                    hour = 0
            
            # Validate hour is in valid range
            if not (0 <= hour <= 23):
                logger.error(f"Invalid hour value: {hour}")
                return None
                
            if not (0 <= minute <= 59):
                logger.error(f"Invalid minute value: {minute}")
                return None
            
            # Create datetime with the correct hour and minute
            target_date = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Ensure the time is in the local timezone
            if target_date.tzinfo is None:
                target_date = LOCAL_TIMEZONE.localize(target_date)
            
            return target_date
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing time: {e}")
            return None
    
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
            purpose = re.sub(r'\b(an?|the|at|on|for|about|book|appointment|schedule)\b', '', purpose).strip()
            if purpose:
                return purpose.title()
    
    # If no clear indicator, try to find the first noun phrase
    words = text.split()
    if len(words) >= 2:
        for i in range(len(words)-1):
            phrase = f"{words[i]} {words[i+1]}"
            if not any(word in phrase.lower() for word in ["today", "tomorrow", "am", "pm", "schedule", "book", "appointment", "may", "june", "july", "august", "september", "october", "november", "december"]):
                return phrase.title()
    
    return "General Appointment"  # Default purpose if none found

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
