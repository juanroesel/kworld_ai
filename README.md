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
- [Python (3.10)](https://www.python.org/downloads/release/python-3100/)
- [Poetry (1.6.1)](https://python-poetry.org/)


## Installation
1. Download and unzip the provided `kworld_ai.zip` file into a local directory. The source code can be found in the `code` directory.
2. Download and unzip the persisted instance of the Chroma Vector DB from [this folder](https://drive.google.com/drive/folders/1_jlKQk8Zvp26d12XJz4ukVokWR5_e7Hl?usp=share_link) and place the unzipped file `chroma.sqlite3` and folder `b02981fe-7174-43a1-995d-c5fc31293159` in the following location within the project directory: `./data/vector_db/`. The following is important: 
- The file `data_level0.bin` might get renamed to `data_level0-001.bin` after unzipping. Rename the file to its original format if this happens.
- The folder `./data/vector_db/` should look like this after all files have been gathered:
```
vector_db
|___chroma.sqlite3
|___b02981fe-7174-43a1-995d-c5fc31293159
    |___data_level0.bin
    |___header.bin
    |___index_metadata.pickle
    |___length.bin
    |___link_lists.bin
```
3. Using a `bash` terminal or the IDE of your choice, navigate to the root project directory (where the `pyproject.toml` file is) and install the `kworld_ai` package by running the following comand:

```
poetry install
```
4. Activate the poetry shell:
```
poetry shell
```
5. Run the `main.py` file using poetry:
```
poetry run python src/kworld_ai/main.py
```
**NOTE** the `main.py` file can take the following arguments:

`--temperature`: Sets the OpenAI model's temperature. Use a lower value for more rigid responses.

`--openai_api_key`: Overrides the default `OPENAI_API_KEY` provided in the `.env` file.

`--embedding_model`: Overrides default Embedding model. Must be a valid HuggingFace model with dim=768.

`--nearest_k`: Overrides default nearest K neighbours to return when performing semantic search in the vector space.

Here's an example of how you could run `main.py` with arguments:

```
poetry run python src/kworld_ai/main.py --temperature=0.5 --nearest_k=4
```

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

- The web scraper only extracted information contained within the `<p>` tags.
- The system's pipeline currently relies on synchronous processing, which can make certain processes such as data scraping/loading run considerably slow.
- The embedding workflow introduced in `2-Embedder.ipynb` is designed to run on a single GPU, which can take some time to run (~3 hours with an A100 GPU in Google Colab).
- In-memory caching: In the interest of speed, all data structures and databases were kept in-memory.
- The user can only interact with the system through the terminal screen, once the `main.py` file has been executed.
- No adequate error handling when the Chat Completion API returns a `4XX` or `5XX` error.
- No evaluations have been yet conducted to test the impact of the following components in the system's overall performance:
    - Chunk sizes and overlaps
    - Different embedding models (i.e., `sentence-t5-xxl`)
    - Different chat models (i.e., `meta-llama/Llama-2-7b-chat-hf`)