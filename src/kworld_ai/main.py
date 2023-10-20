import os
import sys
import argparse
from pathlib import Path
from time import perf_counter
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv()

# PATHS
ROOT_DIR = Path(__file__).parents[2]
DATA_PATH = os.path.join(ROOT_DIR, "data")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", None)

# OPENAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

parser = argparse.ArgumentParser(
    prog="KWorld's IR Chatbot",
    description="Provides answers to queries based on the KPMG website",
)

def query(input_query: str, chat_history: list) -> str:
    """
    Takes a query from the user and calls the Conversational Agent
    to provide a response to it.

    Args:
        input_query (str): Query provided by the user through the terminal.
        chat_history (list): List of tuples with past queries and provided answers.

    Returns:
        str: Answer provided by the Conversational Agent.
    """
    response = qa_chain({'question': input_query, 'chat_history': chat_history})
    return response


if __name__ == "__main__":
    parser.add_argument("--temperature", type=float, default=0.1, help="Sets the OpenAI model's temperature.")
    parser.add_argument("--openai_api_key", type=str, default=OPENAI_API_KEY, help="Overrides default OPENAI_API_KEY.")
    parser.add_argument("--embedding_model", type=str, default="thenlper/gte-base", help="Overrides default Embedding model. Must be a valid HF model with dim=768.")
    parser.add_argument("--nearest_k", type=int, default=6, help="Overrides default nearest K neighbours to return when performing semantic search in the vector space.")

    args = parser.parse_args()

    # API KEY validation
    if not args.openai_api_key:
        print("The System needs a valid OPENAI API KEY!")
        sys.exit()
    
    # Insantiates Embedding model
    embedder = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
    )

    # Loads DB from disk
    db_path = os.path.join(DATA_PATH, VECTOR_DB_DIR)
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedder
    )

    # Instantiates Conversational Agent
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(openai_api_key=args.openai_api_key),
        vectordb.as_retriever(search_kwargs={'k': args.nearest_k}),
        return_source_documents=True
    )

    # Runs Conversational Agent with chat history in-memory
    chat_history = []

    while True:
        # this prints to the terminal, and waits to accept an input from the user
        input_query = input('Insert query: ')

        # give us a way to exit the script
        if input_query == "exit" or input_query == "quit" or input_query == "q":
            print('Exiting')
            sys.exit()

        print()
        print('QUERY: ', input_query)

        try:
            # feed the query along with relevant contexts to the LLM and print out response.
            # chat history is also passed along as context to the LLM.
            start = perf_counter()
            response = query(input_query, chat_history)
            end = perf_counter()
            print('ANSWER: ' + response['answer'])
            print(f"Time: {end - start:.3f} secs")

            # chat_history gets updated based on question and response
            chat_history.append((input_query, response['answer']))

        except Exception as e:
            print(f"There's an error with the Chat Completion API. Please try again later.")
            print(f"ERROR: {e}")
            pass
            
        print()
    
