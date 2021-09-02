# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk


paragraph = """Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an 
 and business magnate. He is the founder, CEO, and Chief Engineer at SpaceX;
 early stage investor,[note 2] CEO, and Product Architect of Tesla, Inc.; founder 
 of The Boring Company; and co-founder of Neuralink and OpenAI. A centibillionaire, 
 Musk is one of the richest people in the world.


Musk was born to a Canadian mother and South African father and raised in Pretoria, 
South Africa. He briefly attended the University of Pretoria before moving to Canada 
aged 17 to attend Queen's University. He transferred to the University of Pennsylvania 
two years later, where he received bachelor's degrees in economics and physics. He moved
 to California in 1995 to attend Stanford University but decided instead to pursue a business
 career, co-founding the web software company Zip2 with brother Kimbal. The startup was acquired
 by Compaq for $307 million in 1999. Musk co-founded online bank X.com that same year, which 
 merged with Confinity in 2000 to form PayPal. The company was bought by eBay in 2002 
 for $1.5 billion.

In 2002, Musk founded SpaceX, an aerospace manufacturer and space transport 
services company, of which he is CEO and CTO. In 2004, he joined electric 
vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.) as chairman 
 product architect, becoming its CEO in 2008. In 2006, he helped create
 SolarCity, a solar energy services company that was later acquired by Tesla 
 became Tesla Energy. In 2015, he co-founded OpenAI, a nonprofit research 
 that promotes friendly artificial intelligence. In 2016, he co-founded Neuralink, 
 neurotechnology company focused on developing brain–computer interfaces, and founded 
 The Boring Company, a tunnel construction company. Musk has proposed the Hyperloop,
 a high-speed vactrain transportation system.

Musk has been the subject of criticism due to unorthodox or unscientific 
 and highly publicized controversies. In 2018, he was sued for defamation by a 
 British caver who advised in the Tham Luang cave rescue; a California jury ruled 
 
 in favor of Musk. In the same year, he was sued by the US Securities and 
 Commission (SEC) for falsely tweeting that he had secured funding for a private 
 takeover of Tesla. He settled with the SEC, temporarily stepping down from his 
 chairmanship and accepting limitations on his Twitter usage. Musk has 
 misinformation about the COVID-19 pandemic and has received criticism from experts
 for his other views on such matters as artificial intelligence, cryptocurrency
 and public transport."""
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

sentence = nltk.sent_tokenize(paragraph)
ps =PorterStemmer()
wnl= WordNetLemmatizer()

corpus = []

for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])
    review = review.lower()
    review = review.split()
    review = [wnl.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
    