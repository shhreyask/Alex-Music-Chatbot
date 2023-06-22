import streamlit as st
import requests
from streamlit_chat import message

#Cell1 training model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD
#!pip install canaro
import canaro
import tensorflow as tf
#from tensorflow.keras.optimizers import SGD
import random

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle




#Cell2 Model functions
#For user needs, creating functions using model

from keras.models import load_model
model = load_model('files_required/chatbot_model.h5')

intents = json.loads(open('files_required/intents.json').read())
words = pickle.load(open('files_required/words.pkl','rb'))
classes = pickle.load(open('files_required/classes.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))
def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


#cell3 chatbot section
msg = list()
text = str()

#defining response
#@anvil.server.callable
def responsed(msg1):
    msg.append(msg1)
    ints = predict_class(msg1)
    res = getResponse(ints, intents)
    return res

#getting google nlp api
from google.cloud import language_v1
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('files_required/spry-compound-385912-eafbf1b97368.json')
client = language_v1.LanguageServiceClient(credentials=credentials)

def analyze_sentiment(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_sentiment(request={'document': document})
    sentiment = response.document_sentiment
    return sentiment.score, sentiment.magnitude

def chatbot_response(user_input):
    # Add your chatbot logic here
    # Example: Echo the user's input
    #return "You said: " + user_input
    convo = []
    # responded function takes text of user and returns chatbot output
    m = user_input
    convo.append(m)
    res = responsed(m)
    return res

def song_emotion(emoticon):

    # Get song recommendations from Last.fm API
    dic1 = dict()
    api_key='a498b96f66d13dc88b20a1c68da02f8a'
    url = f'http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={emoticon}&api_key={api_key}&format=json&limit=10'
    response = requests.get(url)
    payload = response.json()

    for i in range(10):
        r=payload['tracks']['track'][i]
        dic1[r['name']] = r['url']
    return dic1


st.set_page_config(
    page_title="Jadzia",
    page_icon=":robot:"
)

st.header("Alex")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'scores' not in st.session_state:
    st.session_state['scores'] = []

if 'j' not in st.session_state:
    st.session_state['j'] = 0

if 'emotion' not in st.session_state:
    st.session_state['emotion'] = ""

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chatbot_response(user_input)
    st.session_state['j'] = st.session_state['j'] + 1
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    if st.session_state['j'] > 4:
        for talk in st.session_state['past']:
            text = talk
            score, magnitude = analyze_sentiment(text)
            st.session_state.scores.append(score)

        avg_score = sum(st.session_state.scores)/5
        if(avg_score > 0.5):
            st.session_state.generated.append(f"Your sentiment score is: {avg_score}\nEmotion: Joy")
            st.session_state['emotion'] ="joy"
        elif (avg_score > 0 and avg_score < 0.5):
            st.session_state.generated.append(f"Your sentiment score is: {avg_score}\nEmotion: Happy")
            st.session_state['emotion'] ="happy"
        elif (avg_score < 0 and avg_score > -0.5):
            st.session_state.generated.append(f"Your sentiment score is: {avg_score}\nEmotion: Sad")
            st.session_state['emotion'] ="sad"
        else:
            st.session_state.generated.append(f"Your sentiment score is: {avg_score}\nEmotion: Depressed")
            st.session_state['emotion']="depressed"
        emoticon = st.session_state['emotion']
        ans = song_emotion(emoticon)
        lst = list(ans.keys())
        songrec = "Song Recommendations :\n"
        for i in range(10):
          songrec = songrec + "Song_name : "+lst[i] + "\n"
          songrec = songrec + "Song_URL :\n " + ans[lst[i]] + "\n\n"
        st.session_state.generated.append(songrec)

if st.session_state['generated']:
    if st.session_state['j'] < 5:
        for i in range(len(st.session_state['past'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    else:
        message(st.session_state["generated"][5], key="emotionblock")
        message(st.session_state["generated"][6], key="songblock")
        for i in range(len(st.session_state['past'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
