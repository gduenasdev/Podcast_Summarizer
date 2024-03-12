import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

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

def convertToLangchainDocuments(documents):
    langchain_documents = []
    metadata = {"source": "youtube"}
    for context in documents:
        document = Document(page_content=context, metadata=metadata)
        langchain_documents.append(document)
    return langchain_documents

def connectToApi():
    return OpenAI(model_name="gpt-3.5-turbo-instruct", \
                    temperature=0.7, openai_api_key=os.getenv("API_KEY"))

def createSummary(llm, transcript_chunks):
    # Map
    map_template = """The following text is from the transcript of a podcast
    {text}
    Based on this text, please summarize and highlight topics and examples in the text clearly and concisely in less than 100 words. 
    Summary:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries:
    {text}
    Take these and distill it into a final, consolidated summary of the main topics discussed in the podcast. 
    Podcast Summary:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="text"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="text",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    final_summary = map_reduce_chain.run(transcript_chunks)
    return final_summary
