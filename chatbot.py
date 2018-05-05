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
    for i in range(len(conversation)):
        questions.append(id_to_line[conversation[i]])
        answers.append(id_to_line[conversation[i + 1]])