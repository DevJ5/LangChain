from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Api Key (Obviously keep this in environment variables)
api_key = "sk-abcdef"

# Which LLM to use?
llm = OpenAI(openai_api_key=api_key)

# Prompt template
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

# Set up a chain with the LLM and the prompt template
code_chain = LLMChain(llm=llm, prompt=code_prompt)

# Call the chain to execute with the input parameters
result = code_chain({"language": "python", "task": "return a list of numbers"})

# The result is always an object with the input parameters and a key called text that contains the output
print(result["text"])
