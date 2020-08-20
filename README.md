# Text-Mining---Topic-Modeling
## Background
Amazon receives huge amount of reviews on various Brands and products it sells, and it needs sophisticated machine learning techniques to understand customer feedback which is usually in the form of unstructured text data. For understanding customer concerns and to be able to do it accurately with least manual intervention, its necessary to employ machines to understand data and make sense of it. We want to employ techniques like Topic Modeling, Sentiment analysis and various machine learning algorithms for user rating predictions to make sense of these large unstructured datasets.
### Goal 

In this study, I will be exploring over 68k customer reviews of over 720 mobile phones posted on Amazon.
I aim to take a two-pronged approach. One, to use the techniques of topic modeling to point out the top positive and negative aspects of purchase that the users associate with a brand / product based on their reviews. In this respect, we will focus on the application of LDA (Latent Dirichlet Allocation) and NMF (Non-Matrix Factorization) towards achieving this goal and then comparing the outputs obtained by these two techniques
The second prong of the project will be pointed at creating a predictive model to predict user ratings by exploring logistic regression, Support Vector Machine, Random Forest and Naïve Bayes and arrive at the best model to do the same.

An online version of our project is hosted on Google Colab.
Please follow the link https://colab.research.google.com/drive/1WolAb0Al-9LwdQp10THen-38VLlzpjfb#scrollTo=YvKBs7h5rjwL&uniqifier=1 and request access.


### Sentiment Analysis

Sentiment Analysis is contextual mining of text which identifies and extracts subjective information in source material and helps a business understand the social sentiment of a brand. Here we are dealing with user reviews of multiple brands and hence this technique is very essential to understand the customer sentiment.
Input to a Sentiment analysis technique is a corpus, which is basically a collection of words where order matters. 
Methods used for Sentiment analysis:
* TextBlob: TextBlob is a part of NLTK library in python. The output of sentiment analysis is a sentiment score ranging from -1 to 1 (which is polarity) indicating how positive or negative they are and a subjectivity score of 0 to 1 where 0 indicates a fact and 1 indicates an opinion. TextBlob finds all the words and phrases that it can assign a polarity and sensitivity to and averages them all together. Finally, each phone is assigned one polarity and one subjectivity scores.
* Vader: Vader (Valence Aware Dictionary and Sentiment Reasoner) is a part of vaderSentiment library in Python for applying sentiment analysis techniques. The output of Vader method is a compound score metric which is calculated by summing the valence scores of each word in the lexicon, adjusted according to the rules, and normalized to be between -1 (most extreme negative) and 1 (most extreme positive).

### Topic Modeling

Topic modeling is a technique used to extract meaningful information from vast amounts of data. In this study I am trying to label different topics among all the cell phone reviews in order to provide business recommendations to sellers. 
Input to the topic modelling technique is a document-term Matrix: It is a matrix where the rows are different documents and columns are different terms and values in the matrix are the word counts. Each topic will consist of a bag of words not necessarily ordered.
I am going to implement topic modeling by using a python library called Gensim. This package utilizes a topic modeling technique called Latent Dirichlet Allocation (LDA). This technique aims to find the hidden probability distributions where every document is a probability distribution of topics and every topic is a distribution of words. The document-term matrix, is given number of topics and number of iterations as input to the Gensim LDA process. Gensim will go through the process of finding the best word distribution for each topic and best topic distribution for each document.
I have also performed Nonnegative-Matrix Factorization, input to this algorithm is Document-Term matrix and number of topics. The output we receive from the algorithm are two non-negative matrices of the original words by K-topics and those k topics by the m original documents. 
