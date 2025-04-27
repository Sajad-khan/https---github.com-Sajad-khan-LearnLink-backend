import os
import json
import requests
import re
from dotenv import load_dotenv
import logging
import pandas as pd
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

def read_file_content(filepath):
    """
    Read file content based on file type
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    try:
        if file_extension == '.xlsx':
            logger.info(f"Reading Excel file: {filepath}")
            df = pd.read_excel(filepath, engine='openpyxl')
            
            # Log basic dataframe info
            logger.info(f"Excel file shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df.to_csv(index=False)
        
        elif file_extension == '.csv':
            logger.info(f"Reading CSV file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"CSV file size: {len(content)} characters")
                return content
        
        elif file_extension == '.txt':
            logger.info(f"Reading Text file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Text file size: {len(content)} characters")
                return content
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        logger.error(traceback.format_exc())
        raise

def clean_json_response(json_response):
    """
    Clean and extract JSON from potentially formatted response
    """
    # Remove markdown code block markers
    json_response = json_response.replace('```json', '').replace('```', '').strip()
    
    # Try different cleaning strategies
    cleaning_attempts = [
        json_response,  # Original response
        json_response.strip('[]'),  # Remove outer brackets if present
        re.sub(r'^\s*[\[\{]', '', json_response),  # Remove leading whitespace and first bracket
    ]
    
    for attempt in cleaning_attempts:
        try:
            # Try parsing the cleaned response
            parsed_json = json.loads(attempt)
            logger.info("Successfully parsed JSON")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing attempt failed: {e}")
    
    # If all attempts fail, raise an error
    raise ValueError("Could not parse JSON response")

def convert_to_embedable_json(filepath):
    """
    Convert various file types to a standardized JSON format using OpenAI
    """
    try:
        # Read file content
        file_content = read_file_content(filepath)
        
        # Prepare prompt for OpenAI
        prompt = f"""
        Convert the following data into a clean, structured JSON array where each object represents a row.
        
        Key requirements:
        1. Create a JSON array of objects
        2. Each object should have meaningful keys from the original data
        3. Preserve all important information
        4. Ensure valid JSON formatting
        5. Remove any unnecessary whitespace or formatting

        Sample Data (first 5000 characters):
        ```
        {file_content[:5000]}
        ```

        Return ONLY the JSON array. No additional text or formatting.
        """
        
        logger.info("Sending request to OpenAI for JSON conversion")
        
        # Call OpenAI API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert data transformer. Convert input data to a clean, embeddable JSON format."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 4000
            }
        )
        
        response.raise_for_status()
        
        # Extract JSON response
        json_response = response.json()['choices'][0]['message']['content']
        
        logger.info("Received response from OpenAI")
        logger.info(f"Response length: {len(json_response)} characters")
        
        # Clean and parse JSON
        parsed_json = clean_json_response(json_response)
        
        # Validate parsed JSON
        if not isinstance(parsed_json, list):
            logger.warning("Parsed JSON is not a list, converting to list")
            parsed_json = [parsed_json]
        
        logger.info(f"Successfully parsed JSON. Number of entries: {len(parsed_json)}")
        
        # Save parsed JSON
        output_filepath = os.path.splitext(filepath)[0] + '_embedable.json'
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=2)
        
        logger.info(f"Converted {filepath} to embeddable JSON: {output_filepath}")
        return output_filepath
    
    except Exception as e:
        logger.error(f"Unexpected error during JSON conversion: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    # Prompt user for file path
    filepath = "data.xlsx"
    
    try:
        # Convert to embedable JSON
        embedable_json_path = convert_to_embedable_json(filepath)
        print(f"Embedable JSON created at: {embedable_json_path}")
    
    except Exception as e:
        print(f"Error processing file: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()