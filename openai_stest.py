import os
import openai

openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_base = os.environ["OPENAI_API_BASE"]# + "chat/completions"
openai.api_key = os.environ["OPENAI_API_KEY"]

message = [
    {"role": "system", "content": "You are an AI assistant that helps people find information."},
    {"role": "user", "content": "Say hi!"}
]

def talk_with_llm(messages):
    try:
        resp = openai.ChatCompletion.create(
            #engine="llama3",  # Value of the script argument --engine
            #headers = {"Content-Type": "application/json"},
            model="llama3",
            messages=messages,
            temperature=0.7,
            max_tokens=8,
            top_p=0.95,
            n=10,  # max(self.n_gens==10 if _ else self.n_gens==10/(?int), 3)
            request_timeout=10
        )
        return resp
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

resp = talk_with_llm(message)

if resp is None:
    print("Uh nope that did not work")
else:
    print(resp)

