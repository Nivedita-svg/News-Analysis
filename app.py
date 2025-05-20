from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import requests
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Initialize NLP components
nltk.download('punkt')
nltk.download('stopwords')

# Load or create spam detection model
if os.path.exists('spam_model.pkl') and os.path.exists('vectorizer.pkl'):
    spam_model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
else:
    # Load the Fake and Real News Dataset
    try:
        df_fake = pd.read_csv('data/Fake.csv')
        df_true = pd.read_csv('data/True.csv')
        
        # Label the data (0 for real, 1 for fake)
        df_fake['label'] = 1
        df_true['label'] = 0
        
        # Combine datasets
        df = pd.concat([df_fake, df_true])
        
        # Preprocess text
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            return text
            
        df['text'] = df['title'] + ' ' + df['text']
        df['text'] = df['text'].apply(preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42
        )
        
        # Vectorize and train
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        
        spam_model = LogisticRegression(max_iter=1000)
        spam_model.fit(X_train_vec, y_train)
        
        # Save the model
        pickle.dump(spam_model, open('spam_model.pkl', 'wb'))
        pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
        
        print("Model trained and saved successfully")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        # Fallback to dummy data if dataset not found
        spam_model = LogisticRegression()
        vectorizer = TfidfVectorizer()
        X_train = ["free money", "legit news", "win prize", "important news"]
        y_train = [1, 0, 1, 0]
        X_train = vectorizer.fit_transform(X_train)
        spam_model.fit(X_train, y_train)

def generate_60_word_summary(text):
    """Generate exactly 60-word summary using LSA"""
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    summarizer = LsaSummarizer()
    
    # Get more sentences than needed
    sentences = [str(s) for s in summarizer(parser.document, 10)]
    full_text = ' '.join(sentences)
    
    # Split into words and take first 60
    words = full_text.split()[:60]
    
    # Ensure proper ending
    if len(words) == 60:
        # Remove partial last word if needed
        if '.' not in words[-1] and '!' not in words[-1] and '?' not in words[-1]:
            last_word = words[-1].split('.')[0].split('!')[0].split('?')[0]
            words = words[:-1] + [last_word + '.']
    
    return ' '.join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url')
    text = data.get('text', '')
    
    # Scrape if URL provided
    if url:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer']):
                element.decompose()
                
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        except Exception as e:
            return jsonify({'error': f'Failed to scrape URL: {str(e)}'}), 400
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Preprocess the input text
        processed_text = text.lower()
        processed_text = re.sub(r'[^\w\s]', '', processed_text)
        processed_text = re.sub(r'\d+', '', processed_text)
        
        # Spam detection
        features = vectorizer.transform([processed_text])
        is_spam = bool(spam_model.predict(features)[0])
        spam_probability = float(spam_model.predict_proba(features)[0][1])
        
        # 60-word summary
        summary = generate_60_word_summary(text)
        
        # Topic modeling
        tokens = [nltk.word_tokenize(text.lower())]
        tokens = [[word for word in doc if word not in stopwords.words('english') and word.isalpha()] for doc in tokens]
        if len(tokens[0]) > 0:
            dictionary = corpora.Dictionary(tokens)
            corpus = [dictionary.doc2bow(text) for text in tokens]
            lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
            topics = lda.print_topics(num_words=5)
            formatted_topics = []
            for topic in topics:
                words = [word.split('*')[1].strip('"') for word in topic[1].split(' + ')]
                formatted_topics.append(f"Topic {topic[0]}: {', '.join(words)}")
        else:
            formatted_topics = ["Not enough meaningful words to identify topics"]
        
        return jsonify({
            'text': text[:1000] + '...' if len(text) > 1000 else text,
            'is_spam': is_spam,
            'spam_probability': round(spam_probability * 100, 2),
            'summary': summary,
            'word_count': len(summary.split()),
            'topics': formatted_topics
        })
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)