from openai import OpenAI
from typing import List, Dict
from config import LLM_API_KEY, LLM_BASE_URL, EMBED_API_KEY, EMBED_BASE_URL, LLM_MODEL, EMBED_MODEL

llm_client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

embedding_client = OpenAI(
    api_key=EMBED_API_KEY,
    base_url=EMBED_BASE_URL,
)

def get_response(messages: List[Dict[str, str]]):
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=16384,
        timeout=60,
        temperature=0.0,
    )
    return response.choices[0].message.content

def get_embedding(text):
    response = embedding_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding

# if __name__ == "__main__":
#     prompt = "你好"
#     response = get_response(prompt)
#     print(response)
#     embedding = get_embedding(prompt)
#     print(embedding)