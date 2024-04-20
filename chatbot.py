import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=tf.keras.models.load_model('chatbot_model.h5')

def clean_up_sentence (sentence):
    sentence_words = nltk.word_tokenize (sentence)
    sentence_words = [lemmatizer.lemmatize (word) for word in sentence_words]
    return sentence_words
def bag_of_words (sentence):
    sentence_words = clean_up_sentence (sentence)
    bag=[0]*len (words)
    for w in sentence_words:
        for i, word in enumerate (words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_TRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r>ERROR_TRESHOLD]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list
def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result

print("GO Bot is running")

import tkinter
from tkinter import *

def send():
    message = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if message != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + message + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        
        ints=predict_class(message)
        res=get_response(ints,intents)

        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
