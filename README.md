# Context-Aware Chatbot

## Overview

The `context_chatbot` project is a sophisticated document interaction system that leverages the **Flan-T5 model** and **LangChain** to provide context-aware responses based on a text document.

This project includes two Python scripts:

* `app_ui.py`: Enhanced user interface with Streamlit and custom styling.
* `app.py`: Basic Streamlit interface with simpler functionality.

Both versions utilize a vector store (FAISS) to index and retrieve information from `data/my_notes.txt`, offering a practical solution for document querying and conversation management.

## Features

* **Enhanced UI Version (`app_ui.py`)**:
  Polished web interface with custom CSS styling, real-time progress indicators, chat history with timestamps, session management, and sidebar metrics (e.g., word count, message count).

* **Basic UI Version (`app.py`)**:
  Minimalistic Streamlit-based interface with essential chat functionality. Suitable for local testing or deployment in resource-constrained environments.

* **Document Processing**:
  Indexes `data/my_notes.txt` using FAISS, splitting it into manageable chunks for efficient retrieval.

* **Context Awareness**:
  Maintains conversation context with LangChain’s `ConversationBufferMemory`, enabling follow-up questions based on prior interactions.

* **Customizable**:
  Supports different embedding models and can be adapted for various document sources or model variants.

## What This Project Does

This project creates an intelligent chatbot capable of understanding and responding to questions about a specific document (`my_notes.txt`). It processes the document into a searchable vector store, enabling the chatbot to retrieve relevant information and provide context-aware answers.

The enhanced UI version offers an interactive experience with visual feedback, while the basic version provides a lightweight alternative for users who prefer simplicity or need to run it locally.

## What This Project Helps With

* **Document Exploration**: Quickly find information in large text documents.
* **Knowledge Management**: Useful for notes, reports, or manuals.
* **Learning and Research**: Interactive document querying for study or research.
* **Automation**: Reduces manual review by automating Q\&A tasks.

## What This Project Teaches You

* **Natural Language Processing (NLP)**: Use of transformer models like Flan-T5 for text generation.
* **Vector Embeddings**: Semantic search using FAISS and sentence-transformers.
* **LangChain Integration**: Memory, chains, and context handling.
* **Streamlit Development**: Building interactive apps with custom styling.
* **Data Processing**: Chunking and indexing large documents.
* **Software Optimization**: Creating full-featured and minimal app versions.

## Prerequisites

* Python 3.8 or higher
* `pip` (Python package manager)

## Dependencies

All required packages are listed in `requirements.txt`. Install them using:

```
pip install -r requirements.txt
```

Key dependencies include:

* `streamlit`: Web interfaces
* `langchain_community`: Vector store and memory tools
* `transformers`: Flan-T5 model
* `sentence-transformers`: Embeddings with `all-MiniLM-L6-v2`

**Notes on `requirements.txt`:**

* Specifies compatible versions to avoid conflicts
* Use a virtual environment (`venv`, `conda`)
* Internet is required for model downloads and updates

## Installation

**Clone the Repository:**

```
git clone <repository-url>
cd context_chatbot
```

**Set Up a Virtual Environment (Recommended):**

```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

**Install Dependencies:**

Run the command shown above.

**Prepare the Document:**

* Place your text in `data/my_notes.txt`
* Example content: meeting notes, articles, reports, etc.

## Running the Application

### Enhanced UI Version (`app_ui.py`)

```
streamlit run app_ui.py
```

* Open in your browser: [http://localhost:8501](http://localhost:8501)
* Features: Styled chat UI, sidebar metrics, history management, model refresh

### Basic UI Version (`app.py`)

```
streamlit run app.py
```

* Also runs on [http://localhost:8501](http://localhost:8501)
* Simple UI for fast use and local testing

## Usage

* **Enhanced UI**:
  Type questions in the chat. See responses with avatars and timestamps. Sidebar shows metrics and controls.

* **Basic UI**:
  Simple input box and response area with basic chat history.

**Tips**:
Be specific with questions (e.g., “What was discussed in the July meeting?”). Follow-up questions will use prior context from `my_notes.txt`.

## Customization

* **Document**: Replace `my_notes.txt` with your own file in the `data/` directory.
* **Model**: Change `model_id` in scripts to use another Flan-T5 variant.
* **Embedding**: Update `embedding_model` to a different model from `sentence-transformers`.
* **UI Styling**: Modify CSS in `app_ui.py` under "CUSTOM CSS STYLING". Simplify `app.py` further if needed.

## Troubleshooting

* **Missing Document**: Ensure `data/my_notes.txt` exists. Otherwise, the app will raise an error.
* **Model Loading Issues**: Check your internet connection and memory availability. Use the “Refresh Models” button or restart.
* **Performance**: For large documents, tweak `chunk_size` (default: 500) and `chunk_overlap` (default: 50). Use `app.py` for better performance on low-resource systems.

## Contributing

Fork the repository, make changes, and submit a pull request. Update `requirements.txt` for new dependencies and document your changes in the README.

## Contact

For support or issues, open an issue in the repository or contact the maintainers.

## Author

Maryum Fasih  
FAST NUCES
