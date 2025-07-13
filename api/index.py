import os
import json
import asyncio
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from typing import Dict
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode
import requests 
from flask import Flask, jsonify
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

webhook_url = os.getenv("WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class BountyItem(BaseModel):
    url: str = Field(..., description="url of the bounty")
    amount: str = Field(..., description="amount of te bounty")
    date: str = Field(..., description="date since when bounty is opened")
    heading: str = Field(..., description="heading of bounty")

async def extract_structured_data_using_llm(
    provider: str = "gemini/gemini-1.5-flash", api_token: str = GEMINI_API_KEY, extra_headers: Dict[str, str] = None
):
    logger.info(f"Extracting Structured Data with {provider}")
    
    try:
        browser_config = BrowserConfig(headless=True)
        
        extra_args = {"temperature": 0.1, "top_p": 0.9, "max_tokens": 20000}
        if extra_headers:
            extra_args["extra_headers"] = extra_headers

        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=1,
            page_timeout=20000,
            extraction_strategy=LLMExtractionStrategy(
                llm_config=LLMConfig(provider=provider, api_token=api_token),
                schema=BountyItem.model_json_schema(),
                extraction_type="schema",
                instruction="""From the crawled web page content, identify and extract ALL individual bounty listings. Each listing should be represented as an object with its 'url', 'amount', 'heading', 'bounty_description'. For the 'date' field, extract the date or relative time string (e.g., 'X days ago', 'Y hours from now ') if present; otherwise, leave it null. Compile all extracted bounty objects into a single JSON array under the key 'bounties', as defined by the provided schema. Ensure you capture every distinct bounty on the page.""",
                extra_args=extra_args,
            ),
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url="https://replit.com/bounties?status=open&order=creationDateDescending", 
                config=crawler_config
            )
            logger.info("Successfully extracted content")
            return result.extracted_content
            
    except Exception as e:
        logger.error(f"Error in extract_structured_data_using_llm: {str(e)}")
        raise

def writing_email(bounties_data):
    try:
        from google import genai
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""You are provided with bounty data. You must ONLY use the data provided below and nothing else. Do not invent or add any information not present in the data.
                        Data: {bounties_data}
                        Write a simple ,professional message(containing url ,amount,description of bounty/bounties,time since when its posted) about the highest paid bounty which is posted within 24 hours/1 day using ONLY the information provided above. If the data shows price ranges, mention the range exactly as given.If more than one bounty is having the same price range than mention them all in your message from the provided information ONLY.""",
        )
        logger.info("Successfully generated email content")
        return response.text
        
    except Exception as e:
        logger.error(f"Error in writing_email: {str(e)}")
        raise

def send_slack_message(message):
    try:
        payload = '{"text":"%s"}' % message.replace('"', '\\"')  # Escape quotes
        final_output = requests.post(webhook_url, data=payload)
        logger.info(f"Slack message sent with status: {final_output.status_code}")
        return final_output.text
        
    except Exception as e:
        logger.error(f"Error in send_slack_message: {str(e)}")
        raise

app = Flask(__name__)

@app.route('/', methods=['GET'])
def driver_func():
    try:
        # Validate environment variables
        if not webhook_url:
            logger.error("WEBHOOK_URL not found in environment variables")
            return jsonify({"error": "WEBHOOK_URL not configured"}), 500
            
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return jsonify({"error": "GEMINI_API_KEY not configured"}), 500
        
        logger.info("Starting bounty extraction...")
        
        # Extract bounties
        result = asyncio.run(
            extract_structured_data_using_llm(
                provider="gemini/gemini-1.5-flash", api_token=GEMINI_API_KEY
            )
        )
        
        bounties_data = json.dumps(result, indent=1)
        logger.info(f"Extracted bounties data: {bounties_data}")
        
        # Generate message
        message = writing_email(bounties_data)
        logger.info(f"Generated message: {message}")
        
        # Send to Slack
        final_output = send_slack_message(message)
        
        return jsonify({
            "status": "success",
            "message": "Bounty data processed and sent to Slack",
            "slack_response": final_output
        })
        
    except Exception as e:
        logger.error(f"Error in driver_func: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
