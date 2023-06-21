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


words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('/content/drive/MyDrive/Colab Notebooks/intents.json').read()
intents = json.loads(intents_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)


#Cell2 Model functions
#For user needs, creating functions using model

from keras.models import load_model
model = load_model('chatbot_model.h5')

intents = json.loads(open('Alex-Music-Chatbot/files_required/intents.json').read())
words = pickle.load(open('Alex-Music-Chatbot/files_required/words.pkl','rb'))
classes = pickle.load(open('Alex-Music-Chatbot/files_required/classes.pkl','rb'))
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
credentials = service_account.Credentials.from_service_account_file('Alex-Music-Chatbot/files_required/spry-compound-385912-eafbf1b97368.json')
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
