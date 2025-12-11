
import os
import sys
from pyneuphonic import Neuphonic

# Mock key from user env
API_KEY = "8999e98ca9370e30d6742fe614bbeb549bfb7132c8d808077ce81cdd6c7da4a1.3155bb0d-6e2f-46f2-af40-be217cd8b40d"

def test_neuphonic():
    print(f"Testing Neuphonic with key: {API_KEY[:10]}...")
    try:
        client = Neuphonic(api_key=API_KEY)
        sse = client.tts.SSEClient()
        sse.speed = 1.0
        
        # Mimic the actual enhanced text with emojis and formatting
        text = """Here's a summary of your persuasive practice analysis.
            
            You did a good job minimizing filler words. Keep it up! Your vocabulary was exceptionally diverse and rich. Your ideas flowed together excellently, creating a cohesive narrative.
            
            Your persuasive speech is developing well. Continue using evidence and addressing counterarguments.
            
            You said 150 total words, and about 2.5% of them were filler words.
            
            Keep practicing to improve your persuasive skills! 🎯 ✨ 🌊 💪"""
            
        print(f"Sending request with text length {len(text)} containing emojis...")
        response = sse.send(text)
        print("Response received.")
        
        chunk_count = 0
        for i, chunk in enumerate(response):
            chunk_count += 1
            if i == 0:
                print(f"First chunk type: {type(chunk)}")
                
        print(f"Stream data chunks received: {chunk_count}")
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_neuphonic()
