from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

import config
config.load()

model = init_chat_model("sonar", model_provider="perplexity")

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "korean", "text": "Hi!"})
result = model.invoke(prompt)
print(result.content)
