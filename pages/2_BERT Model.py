import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import praw
import pandas as pd
import datetime
import configparser
import matplotlib.pyplot as plt
import os

st.title("Bert Model")
st.subheader("Confusion Matrix")

st.image('./assets/bert.png')

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

@st.cache_resource
def load_model(output_dir):
    my_model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    my_model.to(device)
    
    print("BERT Model loaded!")
    return my_model, tokenizer

model, tokenizer = load_model("./assets/best_model_save")

def process_and_predict_sentence(sentence, model, tokenizer, device=device):
    encoded_dict = tokenizer.encode_plus(
        sentence,                     
        add_special_tokens=True,     
        max_length=64,                
        pad_to_max_length=True,      
        truncation=True,              
        return_attention_mask=True,   
        return_tensors='pt'           
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)
    return probabilities

user_input = st.text_area("Enter your tweet here:")
if st.button("Submit"):
    sentiment = { 0: "Negative", 1: "Positive"}
    if user_input:
        prob = process_and_predict_sentence(user_input, model, tokenizer)
        negative, positive = prob[0][0], prob[0][1]
        result = 0 if negative > positive else 1
        st.write(f"Your tweet: '{user_input}'")
        st.write(f"Predicted Sentiment: {sentiment[result]}")
        st.write(f"Probabilities - Negative: {negative:.4f}, Positive: {positive:.4f}")
        pass
    else:
        st.write("Please enter some text before submitting.")

config = configparser.ConfigParser()
config.read('config.ini')

reddit = praw.Reddit(
    client_id=config['reddit']['client_id'],
    client_secret=config['reddit']['client_secret'],
    user_agent=config['reddit']['user_agent']
)

file_name = 'sentiment.csv'

# Check if the file exists
if os.path.exists(file_name):
    sentiment_data = pd.read_csv(file_name)
else:
    sentiment_data = pd.DataFrame(columns=['Date', 'Negative', 'Positive'])

def collect_reddit_posts(subreddit_name, post_limit=100, selected_date=None):
    all_posts = []
    subreddit = reddit.subreddit(subreddit_name)
    
    for submission in subreddit.new(limit=post_limit):
        post_date = datetime.datetime.fromtimestamp(submission.created_utc).date()
        if selected_date is None or post_date == selected_date:
            all_posts.append({
                'id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext,
                'created_utc': post_date,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'subreddit': subreddit_name
            })

    return pd.DataFrame(all_posts)

st.title("Reddit Post Collection")
selected_date = st.date_input("Pick a date", datetime.date.today())

if st.button("Collect Posts"):
    df_posts = collect_reddit_posts(subreddit_name="politics", post_limit=1000, selected_date=selected_date)
    
    if df_posts.empty:
        st.write(f"No posts found for the date {selected_date}.")
    else:
        st.write(f"Collected {len(df_posts)} posts from the 'politics' subreddit on {selected_date}.")
        st.dataframe(df_posts)
        
    if not df_posts.empty:
        sentiment_counts = { "Negative": 0, "Positive": 0 }
        for i, r in df_posts.iterrows():
            prob = process_and_predict_sentence(r['title'], model, tokenizer)
            negative, positive = prob[0][0], prob[0][1]
            result = 0 if negative > positive else 1
            if result == 0: 
                sentiment_counts['Negative'] += 1
            elif result == 1: 
                sentiment_counts['Positive'] += 1
        
        # Convert the dictionary to a Pandas Series for plotting
        sentiment_series = pd.Series(sentiment_counts)
        
        plt.figure(figsize=(10, 6))
        sentiment_series.plot(kind='bar', color=['#225b91', '#b3cede'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Counts')
        plt.xticks(rotation=0)
        
        # Display the plot in Streamlit
        st.pyplot(plt)
        
        # Add the sentiment counts to the sentiment_data DataFrame
        sentiment_data = sentiment_data.append({
            'Date': selected_date,
            'Negative': sentiment_counts['Negative'],
            'Positive': sentiment_counts['Positive']
        }, ignore_index=True)
        
        sentiment_data.to_csv('sentiment.csv', index=False)
        
        # Convert the 'Date' column to string format and sort by date
        sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
        sentiment_data = sentiment_data.sort_values(by='Date')
        sentiment_data['Date'] = sentiment_data['Date'].dt.strftime('%Y-%m-%d')
        
        # Plotting the sentiment trends over time
        plt.figure(figsize=(10, 6))
        plt.plot(sentiment_data['Date'], sentiment_data['Negative'], marker='o', color='#225b91', label='Negative')
        plt.plot(sentiment_data['Date'], sentiment_data['Positive'], marker='o', color='#b3cede', label='Positive')
        plt.title('Sentiment Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Count')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Display the trend plot in Streamlit
        st.pyplot(plt)
