import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import streamlit as st
nltk.download('wordnet')

model = pickle.load(open('review_system.pkl','rb'))
tf_idf = pickle.load(open('TF_IDF.pkl', 'rb'))

def text_preprocessing(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = WordNetLemmatizer()
    all_words = stopwords.words('english')
    all_words.remove('not')
    text = [ps.lemmatize(word) for word in text if word not in set(all_words)]
    return ' '.join(text)

st.title('Restaurant Review System')
input_text = st.text_area('please enter the review here')

if st.button('Predict'):
    transform = text_preprocessing(input_text)
    vectors = tf_idf.transform([transform]).toarray()
    result = model.predict(vectors)[0]

    if result == 1:
        st.header('Positive Review 😍')
    else:
        st.header('Negative Review 😔')

value = 'default'
if st.button('Refresh'):
    value = ' '
