# Kobee Raveendran
# chatbot using deep NLP
# uses Cornell movie corpus dataset

import numpy as np
import tensorflow as tf
import re
import time

# get the two datasets (conversations and lines)
lines = open('movie_lines.txt', 'r', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# map the lines to their unique IDs
id_to_line = {}

for line in lines:
    splitline = line.split(' +++$+++ ')
    if len(splitline) == 5:
        id_to_line[splitline[0]] = splitline[4]

# get list of all conversations
conversation_list = []
for conversation in conversations[:-1]:
    splitconvos = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", '').replace(' ', '')
    splitbyid = splitconvos.split(',')
    conversation_list.append(splitbyid)
    
# get questions and answers
questions = []
answers = []

# NOTE: check for correctness if current method fails
'''
for conversation in conversation_list:
    for i in range(len(conversation)):
        if i % 2 == 0:
            questions.append(id_to_line[conversation[i]])
        else:
            answers.append(id_to_line[conversation[i]])
'''
for conversation in conversation_list:
    for i in range(len(conversation) - 1):
        questions.append(id_to_line[conversation[i]])
        answers.append(id_to_line[conversation[i + 1]])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?!,]", "", text)
    return text

clean_questions = []
clean_answers = []

for question in questions:
    clean_questions.append(clean_text(question))

for answer in answers:
    clean_answers.append(clean_text(answer))

wordcount = {}

for string in clean_questions + clean_answers:
    _wordlist = string.split(' ')
    for word in _wordlist:
        if word in wordcount:
            wordcount[word] += 1
        else:
            wordcount[word] = 1

threshold = 10

questionwordsbyid = {}
word_num = 0

# includes only the frequently-occuring words (those that have counts above threshold)
for word, count in wordcount.items():
    if count >= threshold:
        questionwordsbyid[word] = word_num
        word_num += 1
        
answerwordsbyid = {}
word_num = 0

for word, count in wordcount.items():
    if count >= threshold:
        answerwordsbyid[word] = word_num
        word_num += 1

# add start of sentence and end of sentence tokens
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

lenquestions = len(questionwordsbyid)
lenanswers = len(answerwordsbyid)

for token in tokens:
    questionwordsbyid[token] = lenquestions + 1
    answerwordsbyid[token] = lenanswers + 1

# invert the mapping between the answers to int (id) dictionary
answeridsbyword = {key: value for value, key in answerwordsbyid.items()}

# append the end of string token to each answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# convert the text from questions/answers into integers, and 
# replaces all the removed words with the OUT token
questions_to_int = []

for sentence in clean_questions:
    ints = []
    for words in sentence.split():
        if word not in questionwordsbyid:
            ints.append(questionwordsbyid['<OUT>'])
        else:
            ints.append(questionwordsbyid[word])

    questions_to_int.append(ints)

answers_to_int = []

for sentence in clean_answers:
    ints = []
    for words in sentence.split():
        if word not in answerwordsbyid:
            ints.append(answerwordsbyid['<OUT>'])
        else:
            ints.append(answerwordsbyid[word])

    answers_to_int.append(ints)