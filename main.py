import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from keras.utils import pad_sequences
import numpy as np
model=load_model('model.h5')
with open('token.pkl','rb' ) as file:
    token=pickle.load(file)

st.text('Tweets Sentiment Analysis')
tweet=st.text_area('Enter the tweet:')
if st.button('Predict Sentiment') and tweet.strip():
    sequence=token.texts_to_sequences([tweet])
    sequence=pad_sequences(sequence,padding='pre',maxlen=99)
    predicton=model.predict(sequence)
    predicted_class=np.argmax(predicton,axis=1)[0]
    sentiment_map={0:'Negative',1:'Neutral',2:'Positive'}
    st.write('Sentiment',sentiment_map[predicted_class])