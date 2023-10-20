# KWorld's Information Retrieval Chatbot
A Proof-of-Concept (PoC) of a conversational AI system that takes queries from users about the KPMG International website and provides an accurate, relevant, and on-topic response to them.

In its current PoC version, the system is able to:
- Answer questions about information found in the KPMG website.
- Disregard questions not related to the KPMG website and let the user know their question falls outside the scope of the information contained therein.
- Preserve a chat history to facilitate sequences of questions and answers.
- Generally provide a recommendation to the user on how to find the information they need if it fails to provide an answer.

The system engineering relies on the following tools and frameworks, all released under licenses that allow commercial use:

- **BeautifulSoup:** A package used to scrape and parse HTML data into a tree-like structure.
- **HuggingFace:** A [Sentence Transformer model](https://huggingface.co/thenlper/gte-base) to generate embeddings.
- **OpenAI:** An [InstructGPT](https://openai.com/research/instruction-following) model powering the system's conversational AI.
- **Langchain:** A framework supporting different processes along the system's pipeline (e.g., chunking, embedding, LLM chaining, etc) 
- **Chroma:** A vector database used to persist the generated embeddings.


## Requirements
- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- [Poetry](https://python-poetry.org/)


## Installation
1. Unzip the provided `kworld_ai.zip` file into a local directory.
2. Using a `bash` terminal or the IDE of your choice, navigate to the root directory (where the `pyproject.toml` file is) and install the `kworld_ai` package by running the following comand:

```
poetry install
```
3. Activate the poetry shell:
```
poetry shell
```
4. Run the `main.py` file using poetry:
```
poetry run python src/kworld_ai/main.py
```
**NOTE** the `main.py` file can take the following arguments:

`--temperature`: Sets the OpenAI model's temperature. Use a lower value for more rigid responses.

`--openai_api_key`: Overrides the default `OPENAI_API_KEY`` provided in the `.env` file.

`--embedding_model`: Overrides default Embedding model. Must be a valid HuggingFace model with dim=768.

`--nearest_k`: Overrides default nearest K neighbours to return when performing semantic search in the vector space.

## Usage
Once the `main.py` script has been run, the user can start interacting with the system via the terminal. Below are some example queries and responses:
```
Insert query: In how many countries does KPMG has a presence?

QUERY:  In how many countries does KPMG has a presence?
ANSWER: KPMG has a presence in 147 countries.
Time: 9.344 secs
```

Here's an example of an exchange involving a follow-up questions to a previously provided answer:

```
Insert query: Does KPMG have offices in Canada?

QUERY:  Does KPMG have offices in Canada?
ANSWER: Yes, there are KPMG offices located in Canada.

Insert query: In which cities?

QUERY:  In which cities?
ANSWER: KPMG has offices located in various cities across Canada, including Toronto, Vancouver, Calgary, Edmonton, Montreal, Ottawa, and Halifax.
```

## Limitations

- The system's pipeline currently relies on synchronous processing, which can make certain processes such as data scraping/loading run considerably slow.
- The embedding workflow introduced in `2-Embedder.ipynb` is designed to run on a single GPU, which can take some time to run (~3 hours with an A100 GPU in Google Colab).
- In-memory caching: In the interest of speed, all data structures and databases were kept in-memory.
- The user can only interact with the system through the terminal screen, once the `main.py` file has been executed.
- No adequate error handling when the Chat Completion API returns a `4XX` or `5XX` error.
- No evaluations have been yet conducted to test the impact of the following components in the system's overall performance:
    - Chunk sizes and overlaps
    - Different embedding models (i.e., `sentence-t5-xxl`)
    - Different chat models (i.e., `meta-llama/Llama-2-7b-chat-hf`)