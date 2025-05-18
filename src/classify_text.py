import config
config.load()

import model
model = model.get_gemini()

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(description="Sentiment of the text")
    aggressiveness: int = Field(description="How aggressive the text is on a scale from 1 to 10")
    language: str = Field(description="The language the text is written in")


structured_llm = model.with_structured_output(Classification)

inp = "너의 태도가 마음에 안들어."
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

print(response)
