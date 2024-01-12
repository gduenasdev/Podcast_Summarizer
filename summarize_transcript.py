import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Load variables from .env into the environment
load_dotenv()

def splitText(transcript: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    texts = text_splitter.split_text(transcript)
    
    return texts

def connectToApi():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", \
                    temperature=0.7, openai_api_key=os.getenv("API_KEY"))

def mapChain():
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

def reduceChain():
    # Reduce
    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

def mapReduceChain():
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )