#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:29:21 2024

@author: woosace
"""

from bs4 import BeautifulSoup
import requests
import csv

page_to_scrape = requests.get("https://news.mit.edu/topic/artificial-intelligence2?type=2")
soup = BeautifulSoup(page_to_scrape.text, "html.parser")

titles = soup.findAll('div', attrs={'class': 'term-page--itm-item--descr'})
authors = soup.findAll('h3', attrs={'class': 'term-page--itm-item--outlet'})

file = open("scraped_articles.csv", "w")
writer = csv.writer(file)
    
writer.writerow(["TITLES", "AUTHORS"])

for title, author in zip(titles, authors):
    print(title.text + " - " + author.text)
    writer.writerow([title.text, author.text])
file.close()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import nltk
from nltk.classify import SklearnClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Analyze sentiment
    sentiment_scores = sia.polarity_scores(text)
    
    # Extract positive, negative, and neutral scores
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    neutral_score = sentiment_scores['neu']
    
    return positive_score, negative_score, neutral_score

def process_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        
        # Add sentiment columns to the header
        header.extend(['Positive', 'Negative', 'Neutral'])
        
        rows = []
        for row in reader:
            text = row[0]
            positive, negative, neutral = analyze_sentiment(text)
            row.extend([positive, negative, neutral])
            rows.append(row)
            
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
        
if __name__ == "__main__":
    input_file = "scraped_articles.csv"
    output_file = "scraped_articles.csv"
    
    process_csv(input_file, output_file)

df = pd.read_csv('scraped_articles.csv')
print(df.shape)
print(list(df.columns.values))
df.head()

# NLTK Method
def NLTK_sentiment(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    sid = SentimentIntensityAnalyzer()
    score = []
    sentiment = []
    
    for d in text:
        s = sid.polarity_scores(d)['compound']
        score.append(s)
        if s > 0.05:
            sentiment.append('pos')
        else:
            if s < -0.05:
                sentiment.append('neg')
            else:
                sentiment.append('neu')
    
    return score, sentiment

# Define process_csv function
def process_csv(input_file, output_file):
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header
            
            # Add sentiment columns to the header
            header.extend(['Score', 'Sentiment'])
            
            rows = []
            for row in reader:
                text = row[0]  # Assuming the text is in the first column
                scores, sentiments = NLTK_sentiment([text])  # Call NLTK_sentiment function
                row.extend([scores[0], sentiments[0]])  # Append score and sentiment to row
                rows.append(row)
        
        # Write results back to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
    
    except StopIteration:
        print("CSV file is empty.")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    input_file = "scraped_articles.csv"
    output_file = "scraped_articles.csv"
    
    process_csv(input_file, output_file)