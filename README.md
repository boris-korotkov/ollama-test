# Retrieval-Augmented Generation (RAG) AI Chatbot

## Introduction
I've always wanted to create a chatbot that leverages Retrieval-Augmented Generation (RAG) to enhance responses using additional files or information from the Internet. However, life’s priorities often got in the way.

One day, a friend reached out with a question about Ollama, asking if it was realistic to build an AI chatbot using Ollama—not just a simple chatbot, but one capable of enhancing responses with articles from the Internet. That question reignited my motivation. I had been procrastinating on building one for myself, but if it could help someone else, why not?

## Project Background
I'm not a full-stack developer, so I collaborated with ChatGPT to build this Proof of Concept (PoC). Initially, my plan was to test it with DeepSeek or Qwen if the results were unsatisfactory. Surprisingly, ChatGPT performed well, and through iterative prompting, I was able to create a working prototype.

In the first iteration, I developed a simple chatbot using:
- **Backend:** Python, Ollama, FastAPI, ChromaDB, and Uvicorn web server
- **Frontend:** Node.js and React

I found that Ollama provided a quick and simple way to experiment with different models. Since this was a PoC, I selected the `deepseek-r1` model with 8 billion parameters. The model size was around 5GB, so my hardware needed to be sufficient to run it.

## Implementing RAG
The most interesting challenge was implementing RAG. Since additional information was to be retrieved from Internet articles, I developed a Python module for web scraping based on links stored in a text file.

Modern AI models are typically trained on vast amounts of internet content. To ensure my chatbot was providing up-to-date information, I identified the latest training date of the DeepSeek model using DeepSeek chat itself. It was determined to be **July 2024** (though this was based on the V3 model, the R1 and V3 release dates are very close, implying similar training data).

![DeepSeek Latest Training Date](/images/DeepSeek_latest_training_date.png)

To ensure relevance, I selected articles published after July 2024. For this, I chose my favorite company website—[Sun Life](https://www.sunlife.com)—and targeted its news section. What better source for fresh information than news articles?

![Sun Life News Example](/images/slf_news.png)

## Architecture Overview
The overall architecture is as follows:
1. A user enters a prompt in the chat window.
2. The frontend chatbot engine sends the prompt to the backend server using an API call.
3. The backend retrieves relevant information from the local vector database (ChromaDB) and sends it as context to the model.
4. The generated response from the model is translated back to the frontend and is displayed to the user.  

![Architecture Diagram](/images/chatbot.png)

## Key Components

### Backend Server
- Located in `chatbot-backend/backend.py`
- Run using the command: `python backend.py`
- Starts a FastAPI server on port **8000**, open for API calls

### Frontend Server
- Built with Node.js and React
- The core file is `chatbot-ui/src/Chatbot.js`
- Start the frontend server at port **3000** with: `npm start` (from the `chatbot-ui` directory)

### Web Scraping Module
- The script `article_ingestion.py` scrapes articles from links listed in `links.txt`
- Extracted content is converted into vector embeddings and stored in ChromaDB
- To avoid duplicate processing, scraped links are stored in `processed_links.txt`

## Validating RAG Integration
To confirm that my chatbot was successfully using RAG, I tested it with a fact I found in a Sun Life news article stating that *Sun Life had C$1.54 trillion in total assets under management as of December 31, 2024*.

![Sun Life Total Assets](/images/slf_total_assets.png)

I entered the prompt: **"What were the total assets managed by Sun Life as of December 31, 2024?"**

The chatbot provided the correct response! Voilà—RAG was working!

![Chatbot Result](/images/chatbot_result.png)

---

## Conclusion
Thank you for reading! I hope this project inspires you to start your own AI chatbot journey. If you have any questions, feel free to reach out!

