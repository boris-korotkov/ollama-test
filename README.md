# A Retrieval-Augmented Generation (RAG) AI chatbot.

Since I've created a Chatbot Proof of Concept (PoC), I always wanted to create a chatbot that would use Retrieval-Augmented Generation (RAG) technology to enhance it's response based on the additional files or information from the Internet. 

However, there were other priorities in my life as usual until my friend reached me one day with a question about Ollama. He asked if it was realistic to build an AI chatbot using Ollama,  not a simple one, but with ability to enhance a response using the articles from the Internet.

That was a gift of the fate for me and a strong motivation to start doing something. I was lazy to do that for myself, but if it could help someone else, why not?

I'm not a full stack developer. Therefore, I cooperate with Chat GPT on building a Proof of Concept for this chatbot. The plan was to try it and go with Deep Seek or Qwen if it was not successful. Surprisingly, the Chat GPT was quite good and I was able to create a working prototype going back and forward with multiple prompts.

As the first iteration, I've created a simple chatbot using Python, Ollama,FastAPI, and uvicorn web-server as my backend interface and Nodes.JS and React as a front-end interface.
I found that Ollama provided very quick and simple way to use different models. Because the purpose as to create a proof of concept I selected deepseek-r1 model with 8 billion parameters. The model size was about 5Gb and my hardware had to be sufficient to run it. 

The next and the most interesting portion was to implement RAG. Given that additional information is to come from the Internet articles, I created a Python module that performs web scrapping based on the links provided in the text file. 

The modern models are usually trained based on Internet content. Therefore, I identified the latest training date for Deep Seek model using Deep Seek chat itself. It was defined as  July 2024. Despite the chat used V3 model, the release date of R1 and V3 are very close and they most likely had very close training date.
![R1 latest training date](/images/DeepSeek_latest_training_date.png)

I had to make sure that articles I selected for augmentation had been ublished after July 2024. So, I chose my favorite company web-site (www.sunlife.com) and went for the news section. What could be better for my purpose than news?!

![SLF News example](/images/slf_news.png)

The overall architecture is below.
A user enter a prompt in the chat window. The front end chatbot engine sends the prompt to the backend server which retrieve the relevant information from the local vector database (ChromaDB) and send it as a context to the model. 

![Architecture diagram](/images/chatbot.png)

### The key files are the following:

+ Backend server: chatbot-backend\backend.py
It needs to be run from chatbot-backend using "Python backend.py" command. It becomes available at port 8000 and open for API calls.

+ Front-end server: It comes with Node.JS and React framework, but the core file is chatbot-ui\src\Chatbot.js. To start front-end server, we need to execute 'npm start' command from the chatbot-ui folder.

+ Web Scrappng. Finally, to scrape the articles from the Internet we need to run article_ingestion.py file which takes links from links.txt file, create a vector from the article content, and update the Chroma database. To avoid duplication and increase the efficiency, it adds the processed links to the processed_links.txt file and does not scrape the same links multiple times.

To validate that my chatbot became a RAG one, I've seen in one article that Sun Life had C$1.54 trillion total assets under management as of December 31, 2024. 

![SLF Total assets](/images/slf_total_assets.png)

I entered a prompt "What was total assets managed by Sun Life as of December31, 2024?" and I got the correct response! Voilà!  It worked!

![Chatbot result](/images/chatbot_result.png)

Thank you so much for reading! I hope that can motivate you to start your journey!


