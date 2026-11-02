# AI-Based FAQ Chatbot Using NLP

## Project Overview
This project is an AI-Based FAQ Chatbot developed using Python, Natural Language Processing (NLP), Scikit-learn and Streamlit. It helps users get answers to frequently asked college-related questions.

## Features
- Interactive chatbot interface
- College FAQ question-answer system
- Text preprocessing
- TF-IDF Vectorization
- Cosine Similarity
- Chat history
- Fallback response for unknown questions

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- NLTK
- JSON

## Project Files
- `app.py` – Main chatbot application
- `faqs.json` – FAQ dataset
- `requirements.txt` – Required libraries

## How to Run

1. Install Python.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Run the application:

   `streamlit run app.py`

## Working
The chatbot compares the user's question with predefined FAQ questions using TF-IDF and Cosine Similarity. It returns the answer for the closest matching question or a fallback response if no suitable match is found.

## Author
Diksha Talekar

## Note
This is an educational demonstration project using sample college FAQ data.
