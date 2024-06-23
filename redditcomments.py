import praw
import json
import os
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain_groq import ChatGroq
import gradio as gr
from fuzzywuzzy import fuzz, process
from transformers import pipeline
from io import StringIO
import sys
import re
import datetime

# Step 1: Fetch Reddit Comments
def fetch_reddit_comments():
    client_id = 'qtn3PRoAXu1o4tZ8BEsmTw'
    client_secret = 'KjN6Tsk5_Oa-n3BCdH-eziyRDpB2Yw'
    user_agent = 'my_reddit_app by /u/First_Acanthaceae_23'
    subreddit_name = 'Forex'

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    comments = []
    subreddit = reddit.subreddit(subreddit_name)
    for comment in subreddit.comments(limit=10000):
        if 'strategy' in comment.body.lower():
            comments.append({
                'author': comment.author.name,
                'body': comment.body,
                'created_utc': comment.created_utc
            })

    with open("forex_strategy_comments.json", "w") as f:
        json.dump(comments, f, indent=4)

    return comments

# Step 2: Data Preparation
def prepare_data():
    with open("forex_strategy_comments.json", "r") as f:
        comments = json.load(f)
    
    directory_path = "Docs"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    for i, comment in enumerate(comments):
        txt_filename = f"comment_{i}.txt"
        txt_filepath = os.path.join(directory_path, txt_filename)
        with open(txt_filepath, "w") as txt_file:
            txt_file.write(comment['body'])

# Step 3: Store Data in Qdrant Vector Database
def store_data_in_qdrant():
    directory_path = "Docs"
    txt_files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]
    all_documents = {}
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    for txt_file in txt_files:
        loader = TextLoader(os.path.join(directory_path, txt_file))
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
        docs = text_splitter.split_documents(documents)
        for doc in docs:
            doc.metadata["source"] = txt_file
        all_documents[txt_file] = docs

    qdrant_collections = {}
    for txt_file in txt_files:
        qdrant_collections[txt_file] = Qdrant.from_documents(
            all_documents[txt_file], embeddings, location=":memory:", collection_name=txt_file)

    return qdrant_collections

# Step 4: Create ReAct Agents
def create_react_agents(qdrant_collections):
    retriever = {}
    for txt_file in qdrant_collections:
        retriever[txt_file] = qdrant_collections[txt_file].as_retriever()

    def get_relevant_document(name: str) -> str:
        search_name = name
        best_match = process.extractOne(search_name, list(qdrant_collections.keys()), scorer=fuzz.ratio)
        selected_file = best_match[0]
        selected_retriever = retriever[selected_file]
        global query
        results = selected_retriever.get_relevant_documents(query)
        global retrieved_text
        total_content = "\n\nBelow are the related document's content: \n\n"
        chunk_count = 0
        for result in results:
            chunk_count += 1
            if chunk_count > 4:
                break
            total_content += result.page_content + "\n"
        retrieved_text = total_content
        return total_content

    def get_summarized_text(name: str) -> str:
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        global retrieved_text
        article = retrieved_text
        return summarizer(article, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']

    def get_today_date(input: str) -> str:
        return f"\n {datetime.date.today()} \n"

    def get_age(name: str, person_database: dict) -> int:
        if name in person_database:
            return person_database[name]["Age"]
        else:
            return None

    def get_age_info(name: str) -> str:
        person_database = {
            "Sam": {"Age": 21, "Nationality": "US"},
            "Alice": {"Age": 25, "Nationality": "UK"},
            "Bob": {"Age": 11, "Nationality": "US"}
        }
        age = get_age(name, person_database)
        if age is not None:
            return f"\nAge: {age}\n"
        else:
            return f"\nAge Information for {name} not found.\n"

    tools = [
        Tool(name="Get Age", func=get_age_info, description="Useful for getting age information for any person. Input should be the name of the person."),
        Tool(name="Get Todays Date", func=get_today_date, description="Useful for getting today's date"),
        Tool(name="Get Relevant document", func=get_relevant_document, description="Useful for getting relevant document that we need."),
        Tool(name="Get Summarized Text", func=get_summarized_text, description="Useful for getting summarized text for any document.")
    ]
    
    retrieved_text = ""
    prompt_react = hub.pull("hwchase17/react")
    model = ChatGroq(model_name="llama3-70b-8192", groq_api_key="YOUR_GROQ_API_KEY", temperature=0)
    react_agent = create_react_agent(model, tools=tools, prompt=prompt_react)
    react_agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return react_agent_executor

# Step 5: Implement a User Interface
def generate_response(question):
    tools = [get_age_info_tool, get_health_info_tool]
    model = ChatGroq(model_name="llama3-70b-8192", groq_api_key="YOUR_GROQ_API_KEY", temperature=0)
    react_agent = create_react_agent(model, tools=tools, prompt=prompt_react)
    react_agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    with StringIO() as text_output:
        sys.stdout = text_output
        completion = react_agent_executor.invoke({"input": question})
        sys.stdout = sys.__stdout__
        text_output_str = text_output.getvalue()
        text_output_str = re.sub(r'\x1b\[[0-9;]*m', '', text_output_str)
        return text_output_str

iface = gr.Interface(fn=generate_response, inputs=[gr.Textbox(label="Question")], outputs=[gr.Textbox(label="Generated Response")], title="Intelligent RAG with Qdrant, LangChain ReAct and Llama3 from Groq Endpoint", description="Enter a question and get a generated response based on the retrieved text.")
iface.launch()

# Running the entire pipeline
comments = fetch_reddit_comments()
prepare_data()
qdrant_collections = store_data_in_qdrant()
react_agent_executor = create_react_agents(qdrant_collections)
