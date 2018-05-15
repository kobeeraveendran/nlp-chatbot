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

# sort question and answer pairs by length of the QUESTIONS
# think of it like teaching humans; children start out with shorter
# sentenced books, then gradually read longer sentences and books
sorted_clean_questions = []
sorted_clean_answers = []

# sentence lengths: min 1, max 25
for length in range(1, 26):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


###############################
#        SEQ2SEQ MODEL        #
###############################

# create placeholders for input tensors
def model_inputs():
    inputs = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'input')
    targets = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'target')
    learning_rate = tf.placeholder(dtype = tf.float32, name = 'learning_rate')
    # controls the dropout rate of neurons
    keep_prob = tf.placeholder(dtype = tf.float32, name = 'keep_prob')

    return inputs, targets, learning_rate, keep_prob

# adds SOS token to targets, and excludes the last token
def preprocess_targets(targets, word_to_int, batch_size):
    left_side = tf.fill(dims = [batch_size, 1], value = word_to_int['<SOS>'])
    # similar to python's slice(), but for tensors
    right_side = tf.strided_slice(input_ = targets, begin = [0, 0], end = [batch_size, -1], strides = [1, 1])
    preprocessed_target = tf.concat(values = [left_side, right_side], axis = 1)

    return preprocessed_target

# create the RNN's encoding layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # deactivates a percentage of neurons: this is usually 20%
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell, 
                                                       sequence_length = sequence_length, 
                                                       inputs = rnn_inputs, 
                                                       dtype = tf.float32)
    
    return encoder_state

# decode training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros(shape = [batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name = 'attn_dec_train')
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function, 
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length, 
                                                                                                              decoding_scope)

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)

    return output_function(decoder_output_dropout)

# decode test and validation sets
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, max_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros(shape = [batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    testing_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, 
                                                                                 encoder_state[0], 
                                                                                 attention_keys, 
                                                                                 attention_values, 
                                                                                 attention_score_function, 
                                                                                 attention_construct_function, 
                                                                                 decoder_embeddings_matrix, 
                                                                                 sos_id, 
                                                                                 eos_id, 
                                                                                 max_length, 
                                                                                 num_words, 
                                                                                 name = 'attn_dec_inf')
    
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                    testing_decoder_function,
                                                                    decoding_scope)

    return test_predictions

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                      num_words, 
                                                                      None, 
                                                                      scope = decoding_scope, 
                                                                      weights_initializer = weights, 
                                                                      biases_initializer = biases)

        training_predictions = decode_training_set(encoder_state, 
                                                   decoder_cell, 
                                                   decoder_embedded_input, 
                                                   sequence_length, 
                                                   decoding_scope, 
                                                   output_function, 
                                                   keep_prob, 
                                                   batch_size)

        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state, 
                                           decoder_cell, 
                                           decoder_embeddings_matrix, 
                                           sos_id = word2int['<SOS>'], 
                                           eos_id = word2int['<EOS>'], 
                                           sequence_length = sequence_length - 1, 
                                           max_length = sequence_length, 
                                           num_words = num_words, 
                                           decoding_scope = decoding_scope, 
                                           output_function = output_function, 
                                           keep_prob = keep_prob, 
                                           batch_size = batch_size)
    
    return training_predictions, test_predictions

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, 
                                                              answers_num_words + 1, 
                                                              encoder_embedding_size, 
                                                              initializer = tf.random_uniform_initializer(0, 1))
    
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.n.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, 
                                                         decoder_embeddings_matrix, 
                                                         encoder_state, 
                                                         questions_num_words, 
                                                         sequence_length, 
                                                         rnn_size, 
                                                         num_layers, 
                                                         questionswords2int, 
                                                         keep_prob, 
                                                         batch_size)
                                                         
    return training_predictions, test_predictions
