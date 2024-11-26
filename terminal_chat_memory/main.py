from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import (
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    FileChatMessageHistory,
)
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)

# Does 2 things:
# 1) Keep the input and output (messages) in memory  and build up a history of the conversation
# 2) Also write this history to a file to continue conversations
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
)

# Since the history can get very long and there is a cap to how much you can send to ChatGTP
# the trick is to let a LLM summarize the conversation and keep that summary in memory
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
)
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])
