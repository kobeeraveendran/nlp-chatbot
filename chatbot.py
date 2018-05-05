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