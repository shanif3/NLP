# Distributional Similarity

### Assignment 1 in NLP course (89-680).   
The goal of this assignment is to explore distributional similarities for words, using the output of
word2vec word embeddings and the output of a Large Language Model (LLM, specifically
ChatGPT). The text below guides you which explorations to perform, and you’re asked to submit
a report which presents the results for all the specified tasks.   

### How to use
The project includes 5 files: `main.py`, `README.md`, `word2vec.pkl`, `twit_model.pkl` and `wiki_news_model.pkl`.   
First, install gensim package by running `pip install gensim` in terminal.   
By running `main.py` script, you will have to choose which command you want to execute: 
- enerating lists of the most similar words.
- Polysemous Words.
- Synonyms and Antonyms.
- The Effect of Different Corpora.
- Plotting words in 2D (the plot is saved as a png file to your current working directory).
- MAP evaluation
- 
**Note:** The project using the models: `word2vec-google-news-300`, `glove-wiki-gigaword-200`, `glove-twitter-200`, so it may take a while to load them.   

### Authors
Shani Finkelstein & Devora Siminovsky