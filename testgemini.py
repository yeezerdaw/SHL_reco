import google.generativeai as genai
from google.api_core import client_options
import os
from dotenv import load_dotenv

load_dotenv()

def test_gemini_connection():
    try:
        opts = client_options.ClientOptions(
            api_endpoint="generativelanguage.googleapis.com"
        )
        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY"),
            client_options=opts
        )
        
        print("\n=== Testing Gemini Connection ===")
        print("Available models:", [m.name for m in genai.list_models()])
        
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content("Hello")
        print("Test response:", response.text)
        
    except Exception as e:
        print(f"\n❌ Gemini connection failed: {e}")
        print("Possible fixes:")
        print("1. Verify your API key in .env")
        print("2. Check billing is enabled")
        print("3. Ensure 'generativelanguage.googleapis.com' is accessible")
        print("4. Use correct model name (gemini-1.0-pro)")

if __name__ == "__main__":
    test_gemini_connection()