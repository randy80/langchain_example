import config
config.load()

import model
model = model.get_gemini()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "korean", "text": "Hi!"})
result = model.invoke(prompt)
print(result.content)
