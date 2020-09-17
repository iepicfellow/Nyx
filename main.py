import json
import numpy as np
import random
import pickle
import tflearn
import tensorflow as tf
import time

import nltk
from nltk.stem.lancaster import LancasterStemmer

import speech_recognition as sr
from gtts import gTTS 
import os 
import pyttsx3

import cv2 as cv
import imutils
import numpy as np
import keras

emo = ""

r = sr.Recognizer()
engine = pyttsx3.init()

stemmer = LancasterStemmer()

with open("intents.json") as myfile:
	data = json.load(myfile)
try:
	with open("input_data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern) #break the sentences into words
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])
		if intent["tag"] not in labels:
			labels.append(intent["tag"])
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))
	labels = sorted(labels)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x,doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		
		output_row = out_empty[:]    
		output_row[labels.index(docs_y[x])] = 1 

		training.append(bag)
		output.append(output_row)

	training = np.array(training)
	output = np.array(output)

	with open("input_data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output),f) 

tf.reset_default_graph()

network = tflearn.input_data(shape=[None, len(training[0])])

network = tflearn.fully_connected(network,8)
network = tflearn.fully_connected(network,8)

network = tflearn.fully_connected(network,len(output[0]),activation="softmax")
network = tflearn.regression(network)

model = tflearn.DNN(network)

try:
	model.load("chatbot.tflearn",)
except:
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("chatbot.tflearn")
	
def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)



model_path = "./model/CNN-Emotion-Model"
face_path = cv.haarcascades + 'haarcascade_frontalface_alt.xml'
emotion_label = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv.VideoCapture(0)
face_model = cv.CascadeClassifier()
if not face_model.load(face_path):
    print('load face model failed F')
    exit(0)

emotion_model = keras.models.load_model(model_path)

def drawFace(frame, frame_grey):
	faces = face_model.detectMultiScale(frame, minSize=(50,50))
	emotion = np.array([[0,0,0,0,0,0,0]])
	for a,b,c,d in faces:

		face = frame_grey[b:b+d,a:a+c]
		face = cv.resize(face, (48, 48))

		face = np.array(face)
		face = face.reshape(1, 48, 48, 1)

		emotion = emotion_model.predict(face/255)
		text = emotion_label[np.argmax(emotion)]

		cv.rectangle(frame, (a,b), (a+c, b+d), (0, 0, 255), 1)
		cv.putText(frame, text, (a,b-10),cv.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
	return emotion



def start_chat():
	print("\n\nBot is ready to talk to you. ")
	mytext = "You look sad, how can i help you?"
	engine.say(mytext)
	engine.runAndWait()
	while True:
		with sr.Microphone() as source:
			print("Go ahead!")
			audio_text = r.listen(source)
			try:
				inp = r.recognize_google(audio_text)
				print("Text: "+r.recognize_google(audio_text))
			except:
				print("Sorry, I did not get that")

		inp = str(inp)

		results = model.predict([bag_of_words(inp,words)])[0] 

		results_index = np.argmax(results)
		tag = labels[results_index]


		if results[results_index] < 0.8 or len(inp)<2:
			if results[results_index] < 0.8 or len(inp)<2:
				print("Hmmm, I couldn't quite understand you\n")
				mytext = "Hmmm, I couldn't quite understand you"
				engine.say(mytext)
				engine.runAndWait()
 				
		else:
			for tg in data['intents']:
				if tg['tag'] == tag:
					responses = tg['responses']

			ran = random.choice(responses)
			engine.say(ran)
			engine.runAndWait()
			print("Bot: "+ran+"\n")
		
			
def real():
	t = 0
	while True:

		if not cap.isOpened():
			print('open camera failed')
			break
		frame = cap.read()[1]
		frame = imutils.resize(frame, width=1000)
		prob = np.ones((300, 500, 3), dtype='uint8')

		frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		emotion = drawFace(frame, frame_grey)
		text = emotion_label[np.argmax(emotion)]
		if str(text) == "Sad":
			t += 1
		if t >= 10:
			start_chat()

		sum = emotion.sum()
		if not sum == 0:
			emotion = emotion / sum

		for i in range(0, 7):
			cv.putText(prob, emotion_label[i], (10, (i + 1) * 35 + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
						1)
			lr = 90 + int(emotion[0][i] * 300)
			cv.rectangle(prob, (90, i * 35 + 25), (90 + int(emotion[0][i] * 300), (i + 1) * 35 + 30), (0, 255, 0), -1)
			cv.putText(prob, str(int(emotion[0][i] * 100)) + '%', (lr, (i + 1) * 35 + 10), cv.FONT_HERSHEY_COMPLEX, 0.5,
						(255, 255, 255), 1)

		cv.imshow('probability', prob, )

		cv.imshow('video', frame)
		if cv.waitKey(1) == ord('q'):
			break
	cap.release()

real() 


"""
def start_chat():
	print("\n\nBot is ready to talk to you. (type 'quit' to stop) ")
	while True:
		inp = input("You: ")
		if inp.lower() in ["quit","exit"]:
			break

		results = model.predict([bag_of_words(inp,words)])[0] 

		results_index = np.argmax(results)
		tag = labels[results_index]


		if results[results_index] < 0.8 or len(inp)<2:
			print("Bot: Sorry, I didn't get you. Please try again.\n")
 				
		else:
			for tg in data['intents']:
				if tg['tag'] == tag:
					responses = tg['responses']

			print("Bot: "+random.choice(responses)+"\n")
		
			
start_chat() 

"""




