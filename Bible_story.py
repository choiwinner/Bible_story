#streamlit run Bible_story.py
#실행 streamlit run Lagnchain_with_bible_250329.py --server.address=0.0.0.0 --server.port=8501
import streamlit as st

from langchain.schema.runnable import RunnableMap
from langchain.retrievers.multi_query import MultiQueryRetriever

#import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever  #TF-IDF 계열의 검색 알고리즘
from langchain.retrievers import EnsembleRetriever # 여러 retriever를 입력으로 받아 처리
from langchain_community.vectorstores.faiss import DistanceStrategy #vectorstores의 거리 계산
from langchain.memory import ConversationBufferWindowMemory

import pickle
import time


from google import genai
from google.genai import types
from io import BytesIO
from PIL import Image
import PIL

from gtts import gTTS
import edge_tts
import io
import wave
from sentence_transformers import SentenceTransformer

def get_conversation_chain(vectorstore,data_list,query,st_memory):
   
    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0)
    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0)

    template = """당신은 인공지능 ChatBOT으로 Question 내용에 대해서 대답합니다.
    Context에 있는 내용을 참조해서만 대답합니다.
    대답은 마크다운으로 출력합니다.
    대답은 context의 있는 original source도 같이 출력합니다.
    #Chat history: 
    {chat_history}
    #Context: 
    {context}
    #Question:
    {question}

    #Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    faiss_retriever=vectorstore.as_retriever(search_type="mmr",
    search_kwargs={'k':10, 'fetch_k': 30})

    # initialize the bm25 retriever(10개)
    bm25_retriever = BM25Retriever.from_documents(data_list)
    bm25_retriever.k = 10

    f_ratio = 0.7
    # initialize the ensemble retriever
    #retriever 가중치 설정(bm25:30% + faiss:70%)
    # 문서 결합 방식 설정(default setting:combine_documents-결합된 문서들을 합치는 방식으로 동작)
    ensemble_retriever_combine = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[1-f_ratio, f_ratio] 
        ,retriever_type="combine_documents")
    
    multiqueryretriever = MultiQueryRetriever.from_llm(ensemble_retriever_combine, llm=llm)


    memory = st_memory

    chain = (
      RunnableMap({
        "context": lambda x: multiqueryretriever.invoke(x['question']),
        "question": lambda x: x['question'],
        'chat_history' : lambda x: x['chat_history']
    }) 
    | prompt | llm | StrOutputParser())

    full_response = '' 
    for chunk in chain.stream({'question': query,
                            'chat_history': memory.load_memory_variables({})['chat_history']}):
        full_response += chunk


    return full_response

def make_story(response,name="크리스"):
   
    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0)
    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp", temperature=0)
    #llm = ChatGoogleGenerativeAI(
        #model="gemini-2.5-flash-preview-04-17-thinking", temperature=0.5)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.5)

    template = """"당신은 세계 최고의 애니메이션 동화책 작가입니다.
    Context를 기반으로 이야기를 만들고 다음 규칙을 지켜서 작성해야 합니다.

    [규칙]
    1. Context의 내용은 성경의 내용을 요약한 것입니다.        
    2. Context 전체 내용을 파악하고 동화 이야기 주제를 설정합니다.  
    3. 동화는 {name}이라는 아이가 애니메이션 슈퍼북처럼 슈퍼북이라는 타임머신을 타고 성경 속 장면을 엿보는 형태로 만듭니다.
    4. 동화는 전체 이야기를 사건과 장면을 중심으로 챕터로 나누어서 세부 내용을 생성합니다.
    5. 동화는 이야기를 듣는 사람이 미취학 어린이부터 초등학교 3학년 이하 수준으로 쉽고 재미있게 작성합니다.
    6. 자연스러운 이야기기를 위해 대화체를 포함하도록 합니다.
    7. 각 챕터에 어울리는 동화 일러스트 생성을 위한 이미지 prompt를 만듭니다. prompt는 <image> 키워드로 분리 생성합니다. 이미지 prompt는 문장 사이에 띄어 쓰기를 하지 않습니다. prompt는 챕터마다 1개만 생성합니다.
    8. 동화 이야기의 줄거리는 <summary> 키워드로 분리 생성합니다.

    #Context: 
    {context}

    #Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (prompt | llm | StrOutputParser())

    #response = chain.invoke({'question': query,
    #                         'chat_history': memory.load_memory_variables({})['chat_history']})
    
    #story_message_placeholder = st.empty() # DeltaGenerator 반환
    
    full_response = '' 
    for chunk in chain.stream({"context": response,
                               "name": name}):
        full_response += chunk
        #story_message_placeholder.markdown(full_response)

    return full_response

def print_story(story):
    st.session_state.story_list = story.split('\n')

    full_tts_response = ''

    for i in range(len(st.session_state.story_list)):
        if "<summary>" in st.session_state.story_list[i]:
            st.session_state.story_summary = st.session_state.story_list[i]

    for i in range(len(st.session_state.story_list)):
        if ("<image>" in st.session_state.story_list[i]) or ("</image>" in st.session_state.story_list[i]) or ("<summary>" in st.session_state.story_list[i]) or ("</summary>" in st.session_state.story_list[i]):
            if ("<image>" in st.session_state.story_list[i]) or ("</image>" in st.session_state.story_list[i]):
                if st.session_state.image_option == '이미지 생성':
                    try:
                        image = make_image3(st.session_state.story_list[i],
                                           st.session_state.story_summary)

                        if image: # 이미지 생성이 성공했는지 확인
                            st.image(image)
                            time.sleep(6) #gemini api limit
                            st.session_state.full_story_list.append(image) 
                        else:
                            # 선택 사항: 이 부분에 대한 이미지 생성 실패 알림
                            # st.markdown("_이미지를 생성하지 못했습니다._")
                            pass
                    except Exception as e:
                        # st.error(f"이미지 생성 중 오류 발생: {e}") # 디버깅용
                        continue
            else:
                continue
        else:
            st.markdown(st.session_state.story_list[i])
            st.session_state.full_story_list.append(st.session_state.story_list[i])  
            full_tts_response += st.session_state.story_list[i]
            time.sleep(0.1)

    return full_tts_response

def make_image(response,summary):
    # API 키 설정
    client = genai.Client(api_key=st.session_state.gemini_api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp-image-generation", temperature=0.3)
    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-preview-image-generation", temperature=0.3)

    prompt_template = PromptTemplate(
        template="""
        The section content is a story from the Bible. Based on the following section content, create a prompt to generate a Ghibli Studio-style image to represent this section. Your prompt should be less than 500 characters. Be consistent in your image creation and refer to the Summary for full details. Write your prompt in English. 
        
        Section content:
        
        {section_content}

        Summary:
        
        {summary}
        
        Prompt:""",
        input_variables=["section_content", "summary"],
    )
    
    image_prompt = llm.invoke(prompt_template.format(section_content=response, summary=summary))

    contents = image_prompt.content

    # Gemini 2.0 Flash Experimental 모델 사용
    response = client.models.generate_content(
        model="models/gemini-2.0-flash-exp",
        #model="gemini-2.0-flash-preview-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image'],
            temperature=0.7,
        )
    )

    # 응답 처리
    for part in response.candidates[0].content.parts:
        try:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                image = Image.open(BytesIO(part.inline_data.data))
                return image
        
        except Exception as e:
            continue
    


def make_image2(response,summary):
    # API 키 설정
    client = genai.Client(api_key=st.session_state.gemini_api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp-image-generation", temperature=0.3)
    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-preview-image-generation", temperature=0.3)

    prompt_template = PromptTemplate(
        template="""
        The section content is a story from the Bible. Based on the following section content, create a prompt to generate a japanese fairy tale animation style image to represent this section. Your prompt should be less than 500 characters. Be consistent in your image creation and refer to the Summary for full details. Write your prompt in English. 
        
        Section content:
        
        {section_content}

        Summary:
        
        {summary}
        
        Prompt:""",
        input_variables=["section_content", "summary"],
    )
    
    image_prompt = llm.invoke(prompt_template.format(section_content=response, summary=summary))

    contents = image_prompt.content

    # Gemini 2.0 Flash Experimental 모델 사용
    response = client.models.generate_content(
        model="models/gemini-2.0-flash-exp",
        #model="gemini-2.0-flash-preview-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image'],
            temperature=0.7,
        )
    )

    # 응답 처리
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_bytes = part.inline_data.data
            image = Image.open(BytesIO(part.inline_data.data))
            st.image(image)

def make_image3(response,summary):
    # API 키 설정
    client = genai.Client(api_key=st.session_state.gemini_api_key)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.3)

    prompt_template = PromptTemplate(
        template="""
        The section content is a story from the Bible. Based on the following section content, create a prompt to generate a Ghibli Studio-style image to represent this section. Your prompt should be less than 500 characters. Be consistent in your image creation and refer to the Summary for full details. Write your prompt in English. 
        
        Section content:
        
        {section_content}

        Summary:
        
        {summary}
        
        Prompt:""",
        input_variables=["section_content", "summary"],
    )
    
    image_prompt = llm.invoke(prompt_template.format(section_content=response, summary=summary))

    contents = image_prompt.content

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE']))

    # 응답 처리
    for part in response.candidates[0].content.parts:
        try:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                image = Image.open(BytesIO(part.inline_data.data))
                return image
        
        except Exception as e:
            continue

def print_response(response):

    #실시간 출력(Stream)

    sentence = ''
    
    if '\n' not in response:
        st.write(response)

    else: 
        for chunk in response:
            #st.write(chunk)
            if chunk in ['\n','\n\n', '\n\n\n']:
                st.write(sentence)
                time.sleep(0.1)
                sentence = ''
            else:
                sentence = sentence + chunk

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"Human: {message.content}")
        else:
            st.write(f"AI: {message.content}")

# 대화 히스토리를 문자열로 변환하는 함수
def get_chat_history_str(chat_history):
    return "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in chat_history])

def file_read():
    folder_path = "C:\\python\\Bible-Rag\\new_data"
    file_list = os.listdir(folder_path)

    documents = []
    pattern2 = r'[가-힣]+'

    for file_name in file_list:
        loader = TextLoader(folder_path+'\\'+file_name, encoding='utf-8')
        document = loader.load()
        result2 = re.search(pattern2, file_name)
        if result2:
            book_name = result2.group()
        document[0].metadata = {'source':book_name}
        documents.append(document)

    document_list = []
    for index,i in enumerate(documents):
        document_list.append(i[0])

    return document_list

def text_splitter(document_list):
    # 2000자(overlap=50로 청킹하기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000,chunk_overlap = 50)
    #splitter를 이용한 문서 청킹
    data = text_splitter.split_documents(document_list)

    return data

def make_vectorstore(data):

    #임베딩 모델
    embeddings = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

    vectorstore = FAISS.from_documents(documents=data, embedding=embeddings)

    st.success(f"총 {len(data)}개의 페이지를 성공적으로 로드했습니다.")
    
    return vectorstore

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key, model="gemini-embedding-exp-03-07", task_type="SEMANTIC_SIMILARITY"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.task_type = task_type
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type=self.task_type)
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings
    
    def embed_query(self, text):
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type=self.task_type)
        )
        return result.embeddings[0].values

def load_bible():
    with st.spinner("파일 불러오는 중..."):
        
        #임베딩 모델 불로오기
        #embeddings = SentenceTransformer('jhgan/ko-sroberta-multitask')
        embeddings = GeminiEmbeddings(api_key=st.session_state.gemini_api_key)

        # 저장된 인덱스 로드(allow_dangerous_deserialization=True 필요)
        
        vectorstore = FAISS.load_local("Rag_data/bible_embed_gemini", 
        embeddings,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE, 
        allow_dangerous_deserialization=True)

        with open("Rag_data/bible_data2.pkl", 'rb') as f:
            data_load = pickle.load(f)
    
    st.success(f"총 {len(data_load)}개의 페이지를 성공적으로 로드했습니다.")

    return data_load, vectorstore

def text_to_speech(text, language='ko'):

    text_new = re.sub('[^a-zA-Z가-힣0-9.,:()?!]', ' ', text)

    # gTTS로 텍스트를 음성으로 변환
    tts = gTTS(text=text_new, lang=language)
    
    # 메모리에 오디오 저장
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)

    # 오디오 재생
    st.audio(audio_data, format="audio/mp3")

    #audio_data를 history에 저장
    st.session_state.full_story_list.append({'mp3':audio_data})
    
    # 다운로드 버튼 추가
    st.download_button(
        label="오디오 다운로드",
        data=audio_data,
        file_name="output.mp3",
        mime="audio/mp3"
    )

def text_to_speech2(text, voice_option):

    text_new = re.sub('[^a-zA-Z가-힣0-9.,:()?!]', ' ', text)

    if voice_option == '여성':
        voice = "ko-KR-SunHiNeural" #여성

    elif voice_option == '남성2':
        voice = "ko-KR-InJoonNeural" #남성2

    else:
        voice = "ko-KR-HyunsuNeural" #남성3

    # 음성 생성
    tts = edge_tts.Communicate(text_new, voice)

    # 메모리에 오디오 저장
    tts.save_sync('output.mp3')

    with open("output.mp3", "rb") as f:
        audio_data = f.read()
    
    st.audio(audio_data, format="audio/mp3")

    #audio_data를 history에 저장
    st.session_state.full_story_list.append({'mp3':audio_data})
    
    st.download_button(
        label="오디오 다운로드",
        data=audio_data,
        file_name="output.mp3",
        mime="audio/mp3"
    )

def text_to_speech3(text):

    text_new = re.sub('[^a-zA-Z가-힣0-9.,:()?!]', ' ', text)

    # 음성 메시지 분할(2000자 이내)

    #strings = text_new.splitlines() # 문자열을 개행문자(\n)를 기준으로 여러 개의 문자열로 나눈 list
    strings = text_new.split(".") # 문자열을 .를 기준으로 여러 개의 문자열로 나눈 list

    new_string = []
    for s in strings:
        if len(s) > 0:                         # 공백 문자 제거
            new_string.append(s.strip() + ".") # 공백 문자 제거 및 문자열 끝에 . 추가

    MAX_LENGTH = 2000                           # Max 문자 설정 : 2000자
    text_list = []
    long_string = ''
    for s in new_string:
        if len(long_string) + len(s) > MAX_LENGTH:
            text_list.append(long_string)
            long_string = s
        else:
            long_string = long_string.strip() + ' ' + s.strip()
    if long_string:
        text_list.append(long_string)

    wave_list = []

    # API 키 설정
    client = genai.Client(api_key=st.session_state.gemini_api_key)

    # 감정이 담긴 한국어 프롬프트
    for index,text in enumerate(text_list):
        emotional_prompt = f"""다음은 어린이 성경 동화입니다. 실감나게 말해 주세요.:
        {text}"""
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=emotional_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore'  )
                        )
                    )
                )
            )

        try:
            # 응답에서 음성 데이터 추출 및 저장
            if response.candidates and response.candidates[0].content.parts:
                audio_data = response.candidates[0].content.parts[0].inline_data.data

            #음성 데이터를 WAV 파일로 저장
            with wave.open(f"output{index}.wav", 'wb') as wav_file:
                wav_file.setnchannels(1)  # 모노
                wav_file.setsampwidth(2)  # 16비트
                wav_file.setframerate(24000)  # 24kHz 샘플링 레이트
                wav_file.writeframes(audio_data)

            wave_list.append(f"output{index}.wav")

        except Exception as e:
            st.info("음성 파일 생성을 실패했습니다. 다른 음성 모델을 사용해 보세요.")

    outfile = "output.wav"

    # 첫 파일의 파라미터를 기준으로 설정
    with wave.open(wave_list[0], 'rb') as w:
        params = w.getparams()

    output = wave.open(outfile, 'wb')
    output.setparams(params)

    for infile in wave_list:
        with wave.open(infile, 'rb') as w:
            output.writeframes(w.readframes(w.getnframes()))

    output.close()

    with open("output.wav", "rb") as f:
        audio_data = f.read()
        
    st.audio(audio_data, format='audio/wav')

    #audio_data를 history에 저장
    st.session_state.full_story_list.append({'wav':audio_data})

    st.download_button(
        label="오디오 다운로드",
        data=audio_data,
        file_name="output.wav",
        mime="audio/wav"
    )

def main():

    st.set_page_config(page_title="Lagnchain_with_bible", page_icon="📖")
    st.title("📖 성경 동화 만들기")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "document_list" not in st.session_state:
        st.session_state.document_list = []
    if "vector_option" not in st.session_state:
        st.session_state.vector_option = None
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = None
    if "response" not in st.session_state:
        st.session_state.response = None
    if "voice_option" not in st.session_state:
        st.session_state.voice_option = None
    if "image_option" not in st.session_state:
        st.session_state.image_option = None
    if "story" not in st.session_state:
        st.session_state.story = None
    if "story_list" not in st.session_state:
        st.session_state.story_list = []
    if "tts_story" not in st.session_state:
        st.session_state.tts_story = None
    if "story_summary" not in st.session_state:
        st.session_state.story_summary = None
    if "name" not in st.session_state:
        st.session_state.name = None
    if "full_story_list" not in st.session_state:
        st.session_state.full_story_list = []
    if "user_query" not in st.session_state:
        st.session_state.user_query = None

    #윈도우 크기 k를 지정하면 최근 k개의 대화만 기억하고 이전 대화는 삭제
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4,return_messages=True) 

    with st.sidebar:
        st.session_state.gemini_api_key = st.text_input('Gemini_API_KEY를 입력하세요.', key="langchain_search_api_gemini", type="password")
        "[Gemini API Key 만들기](https://aistudio.google.com/apikey)"
        "[Gemini API Key 만드는 방법(설명)](https://booknbeyondinsights.tistory.com/entry/gemini-api-key-guide)"

        if (st.session_state.gemini_api_key[0:2] != 'AI') or (len(st.session_state.gemini_api_key) != 39):
            st.warning('잘못된 key 입력', icon='⚠️')
        else:
            st.success('정상 key 입력', icon='👉')

        if process :=st.button("Process"):
            if (st.session_state.gemini_api_key[0:2] != 'AI') or (len(st.session_state.gemini_api_key) != 39):
                st.error("잘못된 key 입력입니다. 다시 입력해 주세요.")
                st.stop()

        if data_clear :=st.button("대화 클리어"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4,return_messages=True)
            st.session_state.response = None
            st.session_state.story = None
            st.session_state.story_list = []
            st.session_state.tts_story = None
            st.session_state.story_summary = None
            st.session_state.full_story_list = []
            st.session_state.user_query = None 
            st.rerun() # 초기화된 상태를 즉시 반영하기 위해

        st.session_state.name = st.text_input("이름", value="", placeholder="아이 이름을 입력하세요.")

        st.session_state.voice_option = st.radio(label='음성 생성 기능',
                          options=['유치원선생님','여성','남성1','남성2','남성3', '음성 미생성'],
                          index=0  # 기본 선택값은 여성
                          )
        
        st.session_state.image_option = st.radio(label='이미지 생성 기능',
                          options=['이미지 생성', '이미지 미생성'],
                          index=0  # 기본 선택값은 생성
                          )
            
    #0. gemini api key Setting
    if not st.session_state.gemini_api_key:
        st.warning("Gemini API Key를 입력해 주세요.")
        st.stop()

    #genai.configure(api_key=gemini_api_key)

    #0. gemini api key Setting
    os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_api_key


    # 파일이 업로드되면 처리
    if st.session_state.vectorstore == None:

        st.session_state.document_list, st.session_state.vectorstore = load_bible()

    st.chat_message("assistant").write("안녕하세요. 무엇을 도와드릴까요?")

    #2. 이전 대화 내용을 출력
    # st.session_state['chat_history']가 있으면 실행
    
    #if ("chat_history" in st.session_state) and (len(st.session_state['chat_history'])>0):
    #    #st.session_state['messages']는 tuple 형태로 저장되어 있음.
    #    for role, message in st.session_state['chat_history']:
    #        if role == 'user':
    #            st.chat_message(role).write(message)
    #        elif role == 'assistant':
    #            with st.chat_message(role):
    #                for content in message:
    #                    if type(content) == str:
    #                        st.markdown(content)
    #                    elif type(content) == PIL.Image.Image:
    #                        st.write("[Image]")
    #                        st.image(content)

    #2. 이전 대화 내용을 출력
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for role, message_content in st.session_state.chat_history:
            if role == 'user':
                st.chat_message(role).write(message_content) # 사용자의 경우, message_content는 질의 문자열
            elif role == 'assistant':
                with st.chat_message(role):
                    # 어시스턴트의 경우, message_content는 문자열 및/또는 PIL.Image.Image 객체의 리스트여야 함
                    if isinstance(message_content, list):
                        for item in message_content:
                            if isinstance(item, str):
                                st.markdown(item)
                            elif isinstance(item, PIL.Image.Image):
                                st.image(item) # 이제 이미지가 표시되어야 합니다
                            elif isinstance(item, dict):
                                index = 0
                                if 'wav' in item:
                                    audio_data = item.get('wav')
                                    st.audio(audio_data, format='audio/wav')
                                    st.download_button(
                                        label="오디오 다운로드",
                                        data=audio_data,
                                        file_name="output.wav",
                                        mime="audio/wav")
                                elif 'mp3' in item:
                                    audio_data = item.get('mp3')
                                    st.audio(audio_data, format='audio/mp3')
                                    st.download_button(
                                        label="오디오 다운로드",
                                        data=audio_data,
                                        file_name="output.mp3",
                                        mime="audio/mp3")
                    else:
                        # 어시스턴트 메시지가 단일 문자열인 경우 (예: 오류 메시지 또는 이전 형식)
                        st.markdown(str(message_content))

    #3. query를 입력받는다.
    #st.session_state.user_query = st.chat_input("질문을 입력해주세요.", key="input_box")
#
    ##3.5 query가 없을 때
#
    #questions = [
    #    "창세기 1장 말씀을 알려주세요.",
    #    "예수님과 삭개오가 만난 이야기를 해 주세요.", 
    #    "성령의 열매를 알려주세요.",
    #    "시편 41장 1절부터 13절 말씀을 알려주세요."]
#
    #if not st.session_state.user_query:
    #    sample = st.selectbox("또는 예시 질문을 선택해주세요:", questions, key="sample_box")
    #    query = sample
    #else:
    #    query = st.session_state.user_query
 
    #3. query를 입력받는다.
    if query := st.chat_input("말씀 주제나 알고 싶은 성경 구절을 알려주세요. \n ex1)창세기 1장을 알려주세요. \n ex2)예수님과 삭개오가 만난 이야기를 해 주세요.", key="input_box"):
        #4.'user' icon으로 query를 출력한다.
        st.chat_message("user").write(query)
        #5. query를 session_state 'user'에 append 한다.
        st.session_state['chat_history'].append(('user',query))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                # chain 호출
                st.session_state.response = get_conversation_chain(
                    st.session_state.vectorstore,
                    st.session_state.document_list,
                    query,
                    st.session_state.memory)

                #story 만들기
                st.session_state.story = make_story(st.session_state.response,
                                                    st.session_state.name)
                
                # 중요: 현재 어시스턴트 메시지를 위해 full_story_list 초기화
                st.session_state.full_story_list = []

                #story 출력하기    
                st.session_state.tts_story = print_story(st.session_state.story)
                #답변 음성 생성하기
                with st.spinner("음성 파일 생성 중입니다."):
                    if st.session_state.voice_option == '유치원선생님':
                        text_to_speech3(text=st.session_state.tts_story)
                    elif len(st.session_state.tts_story) > 5000:
                        st.warning('답변 길이가 너무 길어서 음성 파일을 생성할 수 없습니다.')
                    elif st.session_state.voice_option == '남성1':
                        text_to_speech(text=st.session_state.tts_story,language='ko')
                    elif st.session_state.voice_option != '음성 미생성':
                        text_to_speech2(text=st.session_state.tts_story,
                                        voice_option=st.session_state.voice_option)
                end_time = time.time()
                total_time = (end_time - start_time)
                st.info(f"검색 소요 시간: {total_time}초")
                if st.session_state.full_story_list != None:
                    st.session_state['chat_history'].append(
                        ('assistant',st.session_state.full_story_list))
    
    else:
        #st.warning("질문을 입력해주세요.")
        pass

if __name__ == '__main__':
    main()