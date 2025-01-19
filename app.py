import nltk
import pandas as pd
import streamlit as st


# For Vader
from nltk.sentiment import SentimentIntensityAnalyzer

st.title("Sentiment Analysis App")
st.write("This is a simple web app to perform sentiment analysis on text data.")
nltk.download('vader_lexicon')

# For RoBERTa
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Toggle between CSV upload and text input
input_type = st.radio("Select Input Type", ("Upload CSV", "Enter Text"))

# CSV File Upload
if input_type == "Upload CSV":
    file_uploaded = st.file_uploader("Upload a CSV file", type=["csv", "txt", "xlsx", "xls"])
    df = None
    if file_uploaded is not None:
        try:
            df = pd.read_csv(file_uploaded)
            st.write("Here is a preview of the data:")
            st.write(df.head())
        except Exception as e:
            st.write(e)
            st.write("Please upload a valid CSV file.")
    else:
        st.write("Please upload a CSV file.")
    
    if df is not None:
        st.write("Kindly select the column that contains the text data.")
        column_name = st.text_input("Enter the column name")

        file_columns = df.columns
        if column_name in file_columns:
            text_data = df[column_name].dropna().astype(str)
            st.write("Data loaded successfully.")
        else:
            st.write("Please enter a valid column name.")
    if df is not None and column_name in file_columns:
        if st.button("Perform Sentiment Analysis Using Vader"):
            res = {}
            sia = SentimentIntensityAnalyzer()
            for i, text in text_data.items():
                sentiment_score = sia.polarity_scores(text)
                res[i] = {'text': text, **sentiment_score}
            st.write("Sentiment analysis completed.")
            st.write(pd.DataFrame(res).T)

        if st.button("Perform Sentiment Analysis Using RoBERTa"):
            res = {}
            for i, text in text_data.items():
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                outputs = model(**inputs)
                probs = softmax(outputs.logits.detach().numpy(), axis=1)
                res[i] = {'text': text, 'negative': float(probs[0][0]), 'neutral': float(probs[0][1]), 'positive': float(probs[0][2])}
            st.write("Sentiment analysis completed.")
            st.write(pd.DataFrame(res).T)

# Text Input for Sentiment Analysis
if input_type == "Enter Text":
    user_input = st.text_area("Enter a text string for sentiment analysis")

    if user_input:
        st.write(f"Performing sentiment analysis on the entered text: {user_input}")

        # Perform sentiment analysis using Vader
        if st.button("Perform Sentiment Analysis Using Vader"):
           
            sia = SentimentIntensityAnalyzer()
            sentiment_score = sia.polarity_scores(user_input)
            st.write("Sentiment analysis completed using Vader.")
            st.write(sentiment_score)

        # Perform sentiment analysis using RoBERTa
        if st.button("Perform Sentiment Analysis Using RoBERTa"):
            
            inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)
            outputs = model(**inputs)
            probs = outputs[0][0].detach().numpy()
            probs = softmax(probs)
            st.write("Sentiment analysis completed using RoBERTa.")
            st.write({"negative": float(probs[0]), "neutral": float(probs[1]), "positive": float(probs[2])})
