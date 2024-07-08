# Builtins
import asyncio
import os
# I think these are dependent, not sure about aiohttp
import openai
from aiohttp import ClientSession

openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_base = os.environ["OPENAI_API_BASE"] + "chat/completions"
openai.api_key = os.environ["OPENAI_API_KEY"]

message = []
message.append({"role": "system", "content": "You are an AI assistant that helps people find information."})
message.append({"role": "user", "content": "Say hi!"})

async def talk_with_llm(messages):
    async with ClientSession(trust_env=True) as session:
        openai.aiosession.set(session)
        resp = None
        resp = await openai.ChatCompletion.acreate(engine="llama3", # Value of the script argument --engine
                            messages=messages,
                            temperature=0.7,
                            max_tokens=8,
                            top_p=0.95,
                            n=10, # max(self.n_gens==10 if _ else self.n_gens==10/(?int), 3)
                            request_timeout=10)
    await openai.aiosession.get().close()
    return resp

resp = asyncio.run(talk_with_llm(message))

if resp is None:
    print("Uh nope that did not work")
else:
    print(resp)

