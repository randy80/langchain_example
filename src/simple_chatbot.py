import config
config.load()

import model
model = model.get_gemini()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = { "configurable": {"thread_id": "abc123"}}
query = "Hi! I'm Bob."
input_message = [HumanMessage(query)]
output = app.invoke({"messages": input_message}, config)
print(output["messages"][-1].pretty_print())

query = "What's my name?"
input_message = [HumanMessage(query)]
output = app.invoke({"messages": input_message}, config)
print(output["messages"][-1].pretty_print())
