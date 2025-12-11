
import requests
import os

# Create a dummy audio file for testing
dummy_audio = "test_audio.wav"
with open(dummy_audio, "wb") as f:
    # Minimal WAV header + some data
    f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')

url = "http://localhost:5328/api/enhance-audio"
files = {'file': open(dummy_audio, 'rb')}
data = {'category': 'persuasive'}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success! Headers:")
        for k, v in response.headers.items():
            if k.startswith("X-"):
                print(f"{k}: {v[:100]}...")
        # Check if audio was returned
        if response.headers.get('Content-Type') == 'audio/wav':
            print(f"Audio received, size: {len(response.content)} bytes")
        else:
            print("Response is not audio/wav")
    else:
        print("Request Failed!")
        print(response.text)
except Exception as e:
    print(f"Connection error: {e}")
    
# Cleanup
try:
    os.remove(dummy_audio)
except:
    pass
