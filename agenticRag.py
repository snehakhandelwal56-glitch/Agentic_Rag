import re
import logging
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_userchat(index_name, user_input, thread_id):
    try:
        # Validate inputs
        if not index_name or not user_input or not thread_id:
            raise ValueError("collection_name, user_input, and thread_id are required")

        # Prepare collection name (assuming this function exists)
        collection_withoutSpace = updated_collection_name(index_name)

        # Check if collection exists (adapted for Chroma)
        if not utility.has_collection(collection_withoutSpace):
            logger.info(f"Collection '{collection_withoutSpace}' does not exist.")
            return {"message": f"Collection '{collection_withoutSpace}' does not exist."}

        # Connect to Chroma vector store
        embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma(
            collection_name=collection_withoutSpace,
            persist_directory="./chroma_db",  # Path to your persisted Chroma DB
            embedding_function=embedding_fn
        )

        # Create retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Answer from Knowledge Base (file)
        def file_search(query: str) -> str:
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            return qa.run(query)

        # Web Search Tool
        def local_web_search(query: str) -> str:
            try:
                response = websearch(query)
                if isinstance(response, list):
                    snippets = " ".join([item.get("snippet", "") for item in response])
                else:
                    snippets = str(response)

                cleaned_snippets = re.sub(r"http[s]?://\S+|www\.\S+", "", snippets)
                cleaned_snippets = re.sub(r"\s+", " ", cleaned_snippets).strip()

                system_prompt = (
                    "You are a helpful assistant. Based on the provided web search snippets, "
                    "answer the user's question as accurately as possible. Do not include links, citations, or source names. "
                    "Only return the answer in plain language."
                )

                user_prompt = f"Question: {query}\nWeb Search Results: {cleaned_snippets}"
                messages = [("system", system_prompt), ("human", user_prompt)]
                web_answer = llm.invoke(messages)
                return getattr(web_answer, "content", str(web_answer))

            except Exception as e:
                logger.error(f"Error in local_web_search: {e}")
                return "Error during web search."

        # Define tools
        file_tool = Tool(
            name="File Search",
            func= file_search,
            description=(
                "Use this first for all technical, company, or product questions. "
                "It retrieves from a domain-specific knowledge base. "
                "Only use other tools if this doesn't help."
            )
        )

        web_tool = Tool(
            name="Web Search",
            func=local_web_search,
            description="Use this to search the internet for up-to-date information."
        )

        # Initialize Agent
        agent = initialize_agent(
            tools=[file_tool, web_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )

        # Run agent
        agent_response = agent.run(user_input)
        logger.info(f"Agent response: {agent_response}")

        return {"response": agent_response}

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        return {"error": "An error occurred while processing your request."}
