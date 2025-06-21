import requests
import json

# The URL of the FastAPI server's streaming endpoint
url = "http://localhost:8008/stream"

# The prompt you want to send to the model
prompt = "How many r's in strawberry?"

# The data payload for the POST request
payload = {
    "prompt": prompt
}

# Set the headers for the request
headers = {
    "Content-Type": "application/json"
}

print(f"Sending prompt: '{prompt}'")
print("Response from server:")

try:
    # Make the POST request with streaming enabled
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        # Check if the request was successful
        if response.status_code == 200:
            # Iterate over the response content chunks
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    print(chunk, end='', flush=True)
        else:
            print(f"\nError: Received status code {response.status_code}")
            print(response.text)

except requests.exceptions.ConnectionError as e:
    print(f"\nError: Could not connect to the server at {url}.")
    print("Please make sure the inference server is running.")

print() # for a final newline 