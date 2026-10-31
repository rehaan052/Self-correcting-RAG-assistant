from app.config import settings
from app.llm import get_embeddings, get_llm

print("1) Groq LLM test...")
reply = get_llm().invoke("Say 'setup ok' in two words.")
print("   LLM reply:", reply.content)

print("2) Embedding model test (pehli baar download hoga)...")
vec = get_embeddings().embed_query("hello world")
print("   Embedding size:", len(vec))

print("3) Bonus config check...")
print("   Tavily key set:", bool(settings.tavily_api_key))
print("   Web search enabled:", settings.enable_web_search)
print("   Hallucination check enabled:", settings.enable_hallucination_check)

print("\nSetup sahi hai. Ab ingestion step pe jaa sakte ho.")