# KrishGPT! 🦚 - A RAG-Based Bhagavad Gita Query Assistant

**KrishGPT!** is an interactive RAG web app that lets you ask questions about the Bhagavad Gita. It uses advanced NLP to retrieve relevant text and generate insightful, context-aware responses, with a built-in feature to translate answers into Hindi for broader accessibility.

---

## Key Features

- **Retrieval-Augmented Generation (RAG):** Combines information retrieval and language generation to provide accurate, context-rich answers from the Bhagavad Gita.
- **Document Processing:** The Bhagavad Gita PDF is processed, split into text chunks, and embedded for efficient search and retrieval.
- **Vector-Based Search:** Uses FAISS for fast and efficient retrieval of relevant text chunks.
- **Contextual Answers:** Responses include source references from the Bhagavad Gita.
- **Translation to Hindi:** Allows users to translate responses into Hindi for accessibility.
- **Predefined Questions:** Offers a set of spiritual and philosophical questions for easy exploration.
- **Interactive Chat Interface:** Built with Streamlit, providing a user-friendly chat experience.

---

## Technologies Used

- **Langchain:** For document processing, text splitting, embedding generation, and retrieval.
- **FAISS:** Enables efficient vector-based search for document retrieval.
- **Hugging Face Transformers:** Used for embeddings and response generation.
- **Streamlit:** Powers the interactive web interface.
- **Translators API:** Facilitates translation of responses into Hindi.
- **Python:** Core language for backend logic and NLP processing.

---

## How It Works

1. **Document Loading:** The Bhagavad Gita PDF is loaded and split into smaller text chunks.
2. **Embeddings Generation:** Chunks are transformed into embeddings using `sentence-transformers`.
3. **Vector Search:** FAISS retrieves the most relevant text chunks based on the query.
4. **Contextual Response Generation:** The system uses retrieved chunks to generate a meaningful answer.
5. **Translation (Optional):** Users can translate responses into Hindi for better comprehension.

---

### Clone the Repository

```bash
git clone https://github.com/vashu2425/ask-krishna.git
cd ask-krishna


