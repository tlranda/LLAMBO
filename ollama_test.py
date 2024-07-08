import os
import requests

# Set your Ollama API key and base URL
OLLAMA_API_KEY = os.environ["OPENAI_API_KEY"]
OLLAMA_API_URL = os.environ["OPENAI_API_BASE"] + "chat/completions"
print(OLLAMA_API_URL)

# Define the messages
messages = [
    {"role": "system", "content": "You are an AI assistant that helps people find information."},
    {"role": "user", "content": "Say hi!"}
]

def talk_with_llm(messages):
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3",  # Adjust this to the appropriate engine for Ollama
        "messages": messages, # The usual JSON of conversation history to observe
        "temperature": 0.7, # Between 0 and 2, with >0.8 being more random
        "max_tokens": 8, # Truncate response early once this many tokens are generated
        "top_p": 0.95, # Alternative to temperature called nucleus sampling, where only tokens with top probability mass are sampled (excluding bottom 5% least likely things)
        "n": 10 # N is not supported by Ollama's OpenAI stuff yet -- according to "https://platform.openai.com/docs/api-reference/chat/create", this is how many choices to generate for each input message
    }

    response = requests.post(OLLAMA_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None

# Call the function and print the response
resp = talk_with_llm(messages)

if resp is None:
    print("Uh nope, that did not work")
else:
    print(resp)

