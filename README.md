# Medical Document Analysis and RAG

This repository contains a Python script for analyzing medical documents and retrieving information using advanced NLP techniques. The script leverages several components from the LangChain library, including text splitting, embeddings, and retrieval-based question answering.

## Features

1. **Text Processing**: Utilizes `RecursiveCharacterTextSplitter` for efficient text splitting.
2. **Embedding Generation**: Generates embeddings using `SentenceTransformerEmbeddings`.
3. **Document Loading and Processing**: Loads documents from a directory and processes PDF files.
4. **Vector Store Creation**: Uses `Chroma` for creating a vector store from documents.
5. **Model Initialization**: Initializes a language model for text generation.
6. **Retrieval-based QA**: Implements a retrieval-based question-answering system.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3
- LangChain (`pip install langchain`)
- Sentence Transformers (`pip install sentence-transformers`)

## Usage

1. **Prepare Your Documents**: Place your PDF documents in the `pdf/` directory.
2. **Run the Script**: Execute the script to process the documents and set up the retrieval system.

## Script Workflow

1. The script processes text from PDF documents in a specified directory.
2. It then generates embeddings for the processed text.
3. Creates a vector store for the documents using Chroma.
4. Initializes a language model for retrieval and question answering.
5. Implements a retrieval-based QA system to extract relevant information from the processed documents.

## Limitations

- The system is currently configured for medical documents in English.
- Requires appropriate hardware and software setup to handle large document sets and complex NLP operations.


# NOTES
- Also for this third and last problem, Colab gave me some challenges: it wouldn't giving me access to A100s because they were unavailable, and gave me access to V100. So I had to opt for a even smaller model (GPT2). I wanted to use Meditron 7B but it was bigger then the memory of the V100 (16G)
- At this point, the second problem was that Colab dependencies were having problem with both Chroma, and chromadb libraries, resulting in 'Chroma is not installed' despite being
- I then opted to download the Quantized version of MediTron 7B (.gguf file available at https://huggingface.co/TheBloke/meditron-7B-GGUF/tree/main) so that I could run it on my local CPU. I chose meditron since the heavier version (70B) has recently showed similar performances to PaLM2.
### Things to do if I had more time
- RAG that could read the tables. Usually RAG suffers from that. So I would find a way to show relative tables
- marella (creator of langchain) suggests the used configs of CTransformers. I would to go deeper and play with those, to see which is better and why
- I would extract the context form the summary and put it automatically into the prompt, to make the retrieval among differnt documents faster
- I would upload more medical books so that it could be tested RAG among different, vast topics
