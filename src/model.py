from langchain.chat_models import init_chat_model

def get_gemini():
    return init_chat_model(model="gemini-2.5-flash-preview-04-17", model_provider="google_genai")

def get_perplexity():
    return init_chat_model("sonar", model_provider="perplexity")

