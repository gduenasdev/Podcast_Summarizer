import streamlit as st
from langchain.llms import OpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from app_blurb import blurb
from summarize_transcript import splitText, connectToApi, createSummary, convertToLangchainDocuments

st.title('🎙️⏳ Podcast Summarizer')
st.markdown(blurb)
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(transcript_chunks):
    llm = connectToApi()
    summary = createSummary(llm, transcript_chunks)
    return summary

def load_youtube_transcript(url: str):
    loader = YoutubeLoader.from_youtube_url(
    url, add_video_info=True
    )
    transcript = loader.load()
    title = transcript[0].metadata['title']
    transcript_text = transcript[0].page_content
    split_transcript = splitText(transcript=transcript_text)
    transcript_chunks = convertToLangchainDocuments(split_transcript)
    return transcript_chunks, title

user_input = st.text_input('Enter a youtube podcast url to summarize')

if user_input:  # Check if user_input is not empty
    if openai_api_key.startswith('sk-'):
        #generate_response(user_input)  # Call your response generation function
        transcript_chunks, title = load_youtube_transcript(user_input)
        print('Calling API')
        final_summary = generate_response(transcript_chunks)
        if final_summary:
            st.header(f'Summary of {title}')
            st.write(final_summary)
    else:
        st.warning('Please enter a valid OpenAI API key!', icon='⚠️')
else:
    st.info('Please enter a YouTube URL to get started.')