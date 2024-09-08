import streamlit as st
import torch.nn as nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from ast import literal_eval
import spacy
import configparser
import datetime
import matplotlib.pyplot as plt
import praw

st.title("LSTM Model")
st.subheader("Confusion Matrix")

st.image('./assets/lstm.png')

class MyFirstLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(MyFirstLSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=3,bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size*2)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.predictor = nn.Linear(in_features=hidden_size, out_features=2)
        
    def forward(self,seq):
        embeddings = self.embedding(seq)
        embeddings = self.layer_norm(embeddings)
        lstm_out, (hidden, _) = self.encoder(embeddings)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        hidden = self.batch_norm(hidden)
        hidden = torch.relu(self.fc1(hidden))
        preds = self.predictor(hidden)
        return preds

@st.cache_resource
def load_model(best_model_path):
    # Load data and model
    df = pd.read_csv("./assets/train-processed.csv", converters={"tokens":literal_eval})
    train_data, _ = train_test_split(df, test_size=0.1, random_state=ord("H"))

    vocab = build_vocab_from_iterator(train_data["tokens"], min_freq=2,specials=["<unk>","<pad>"],special_first=True, max_tokens=40000)
    vocab.set_default_index(vocab["<unk>"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_model = MyFirstLSTM(50, 180, len(vocab)).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    nlp = spacy.load("en_core_web_sm")
    print("LSTM Model loaded!")
    return best_model, vocab, nlp, device

best_model, vocab, nlp, device = load_model("./assets/best_model_epoch5.pth")

def preprocess_tweet(tweet, vocab, tokenizer, max_len=30):
    tokens = tokenizer(tweet)
    numericalized = [vocab[token] for token in tokens]
    
    if len(numericalized) < max_len:
        numericalized += [vocab["<pad>"]] * (max_len - len(numericalized))
    elif len(numericalized) > max_len:
        numericalized = numericalized[:max_len]

    return torch.tensor(numericalized, dtype=torch.long).unsqueeze(0)

def predict_tweet_sentiment(model, tweet, vocab, tokenizer, max_len=30):
    model.eval()
    processed_tweet = preprocess_tweet(tweet, vocab, tokenizer, max_len).to(device)
    
    with torch.no_grad():
        output = model(processed_tweet)
        probabilities = nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities.squeeze().tolist()

def tokenize(text):
    return [token.text for token in nlp(text.lower())]

def process_input(model, new_tweet, vocab, tokenize):
    return predict_tweet_sentiment(model, new_tweet, vocab, tokenizer=tokenize)

user_input = st.text_area("Enter your tweet here:")

if st.button("Submit"):
    sentiment = { 0: "Negative", 1: "Positive"}
    if user_input:
        predicted_class, probabilities = process_input(best_model, user_input, vocab, tokenize)
        st.write(f"Your tweet: '{user_input}'")
        st.write(f"Predicted Sentiment: {sentiment[predicted_class]}")
        st.write(f"Probabilities - Negative: {probabilities[0]:.2f}, Positive: {probabilities[1]:.2f}")
    else:
        st.write("Please enter some text before submitting.")

config = configparser.ConfigParser()
config.read('config.ini')

reddit = praw.Reddit(
    client_id=config['reddit']['client_id'],
    client_secret=config['reddit']['client_secret'],
    user_agent=config['reddit']['user_agent']
)

def collect_reddit_posts(subreddit_name, post_limit=100, selected_date=None):
    all_posts = []
    subreddit = reddit.subreddit(subreddit_name)
    
    for submission in subreddit.top(limit=post_limit, time_filter='day'):
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
            predicted_class, _ = process_input(best_model, user_input, vocab, tokenize)
            if predicted_class == 0: 
                sentiment_counts['Negative'] += 1
            elif predicted_class == 1: 
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