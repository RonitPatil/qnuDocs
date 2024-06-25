import os
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor
from ingestData import initialize_vector_store
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

def create_agent_executor(astra_vector_store, openai_api_key, stream_handler):
    retriever = astra_vector_store.as_retriever(search_kwargs={"k": 3})
    # retriever=astra_vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={
    #                           'score_threshold': 0.5})

    retriever_tool = create_retriever_tool(
        retriever,
        "data_search",
        "Search for information about diseases in the vector database and return the most relevant information."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="""You are a very powerful assistant with access to tools that can 
        help you retrieve data from the vector database, answer questions about any questions 
        asked. You are also required to give citations about each answer ALONG with the name 
        of the pdf AND that is given as well as the page number in a NEW PARAGRAPH. ALWAYS PROVIDE THE PDF NAME
        As an AI assistant you provide answers based on the given context, ensuring accuracy and brifness. 

        You always follow these guidelines:

        -If the answer isn't available within the context, state that fact
        -Otherwise, answer to your best capability, refering to source of documents provided
        -Only use examples if explicitly requested
        -Do not introduce examples outside of the context
        -Do not answer if context is absent"""),

            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    chat_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.5,
        streaming=True,
        callbacks=[stream_handler],
    )

    tools = [retriever_tool]
    llm_with_tools = chat_llm.bind_tools(tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

def main():
    st.title("Document QA with Citations")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    db_token = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    db_id = st.secrets["ASTRA_DB_ID"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    vector_store = initialize_vector_store(db_token, db_id)

    for message in st.session_state.agent_messages:
        if isinstance(message, SystemMessage):
            continue
        role = None
        content = None
        if isinstance(message, HumanMessage):
            role = "user"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "assistant"
            content = message.content

        with st.chat_message(role):
            st.markdown(content)

    query = st.chat_input("Ask me anything:", key="user_input")
    if query:
        st.session_state.agent_messages.append(HumanMessage(content=query))
        st.chat_message("user").markdown(query)
        stream_handler = StreamHandler(st.empty())
        agent_executor = create_agent_executor(vector_store, st.secrets['OPENAI_API_KEY'],stream_handler)
        response = agent_executor.invoke({"input": query, "chat_history": st.session_state.agent_messages})
        # st.chat_message("assistant").markdown(response['output'])
        st.session_state.agent_messages.append(AIMessage(content=response['output']))
        if len(st.session_state.agent_messages) > 25:
            st.session_state.agent_messages = st.session_state.agent_messages[-25:]
        


    # Scroll to the bottom to show the latest message
    st.write("<script>window.scrollTo(0,document.body.scrollHeight);</script>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
