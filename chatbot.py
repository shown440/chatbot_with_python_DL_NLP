# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:04:39 2019

@author: SHIFULLAH AHMED KHAN
"""

#Building a chatbot with Deep NLP

#Importing libraries
import numpy as np
import tensorflow as tf
import re
import time

########################################################################################################
########################################################################################################
########################### PART 1 - Data Pre-Processing ###############################################
########################################################################################################
########################################################################################################

#Importing Dataset
lines = open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
conversations = open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

#Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    #print(_line)
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
#print(id2line)

#Creating a list of all of the movie_conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    #a = _conversation
    conversations_ids.append(_conversation.split(","))
# print(type(a))
# print(a)
#print(type(conversations_ids))
#print(conversations_ids)


#Getting seperately the questions and answers
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        # with open("questions.txt", "a") as q1:
        #     q1.write(id2line[conversation[i]]+"\n")

        answers.append(id2line[conversation[i+1]])
        # with open("answers.txt", "a") as a1:
        #     a1.write(id2line[conversation[i+1]]+"\n")
#print(questions)
#print(answers)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)

    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"[-(\"#/@;:<>{}+=~|.?,,)]", "", text)

    return text

#Cleaning the Questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    # with open("cleaned_questions.txt", "w+") as c_q1:
    #     c_q1.write(question+"\n")

#Cleaning the Answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    # with open("cleaned_answers.txt", "w+") as c_a1:
    #     c_a1.write(answer+"\n")

#Creating a dictionary that maps each word to its number of occurence
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
#print(word2count) #Counts the word in cleaned_questions and cleaned_answers
#print(word2count.items())

#Creating two dictionaries that map the questions words and the answers words to a unique Integer
#Thats mean set a primary key for every word (Answer and Question Both)
threshold = 20 #Good threshold should be between 10 and 20 || 20 will remove 5% of least appearing word

questionsWord2Int = {}
word_number = 0
#print(word2count.items())
#print(word2count)
for word, count in word2count.items():
    if count >= threshold:
        questionsWord2Int[word] = word_number
        word_number += 1
        #print(word_number)
#print(questionsWord2Int)
answersWord2Int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answersWord2Int[word] = word_number
        word_number += 1
#print(answersWord2Int)

#Adding the last token two these two dictionaries
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"] #EOS=End of Str, SOS=Start of string
for token in tokens:
    questionsWord2Int[token] = len(questionsWord2Int)+1
for token in tokens:
    answersWord2Int[token] = len(answersWord2Int)+1

#Creating Inverse dictionary of the answersWord2Int dictionary
answersints2word = {w_i : w for w, w_i in answersWord2Int.items()}
   
#Adding the End of String token to end of every answer 
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"

#Translating all the clean_questions and clean_answer into integers
#and replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionsWord2Int: #in questionsword2int, more than threshold count word stayed
            ints.append(questionsWord2Int["<OUT>"])
        else:
            ints.append(questionsWord2Int[word])
    questions_into_int.append(ints)

answers_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in answersWord2Int: #in questionsword2int, more than threshold count word stayed
            ints.append(answersWord2Int["<OUT>"])
        else:
            ints.append(answersWord2Int[word])
    answers_into_int.append(ints)


#Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25+1): #here we set training question length = 25
    for i in enumerate(questions_into_int): #i is couple for enumerate
        if len(i[1]) == length:             #1st element of i=index ans 2nd of i=question
            sorted_clean_questions.append(questions_into_int[i[0]])            
            sorted_clean_answers.append(answers_into_int[i[0]])
 


########################################################################################################
########################################################################################################
########################### PART-2 BUILDING SEQ2SEQ MODEL ##############################################
########################################################################################################
########################################################################################################

            
#Creating placeholder for the inputs and the targets
            # placeholder is used for future training
def model_inputs():
    #this fun will take input directly and convert the into placeholder and go to output
    inputs = tf.placeholder(tf.int32, [None, None], name='input') # placeholder has 3 parameters. 1=type of input
                                                                  # 2=dimention metrix of input data, 3=name='input'
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    
    
    lr = tf.placeholder(tf.float32, name='learning_rate') #used to see the learning rate
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #used to control drop out rate
    
    return inputs, targets, lr, keep_prob

#Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int["<SOS>"]) #Left side of concateneted file
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1]) #Right side of concateneted file
    
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    # concat has 3 args. 1=values=left+right side, 2=axis(1=horizontal concat, 0=vertical concat), 3=name
    #print(preprocessed_targets)
    
    return preprocessed_targets
#print(preprocess_targets())


############################################################################################################################################    
#Creating the Encoder RNN Layer
############################################################################################################################################
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    #1. rnn_inputs=model_inputs.inputs 2. rnn_size , 3. num_layers=number of layer, 4. keep_prob, 
    #5. sequence_length=list of the length of each question in the batch
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    # _, encoder_state = encoder_output, encoder_state
    return encoder_state

#############################################################################################################################################
#Decoding the training set
#############################################################################################################################################
# here we decoded the observation the training set. here some observation will go into the Neural Network and update the weight -->
    # and improve the ability of ChatBot to talk like Human. 
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    # encoder_state = will return from encoder_rnn_layer
    # decoder_cell = is the cell of recurrent nural network of Decoder
    # decoder_embedded_input = is the input of decoder. || Goto Tensorflow website->Embeded area
    # sequence_length = list of the length of each question in the batch
    # decoding_scope =  is object of variable scope|| Goto Tensorflow website->tf.variable_scope area
    # output_function = is a fun will used to return the decoder output
    # keep_prob = is a Regulator which used to control drop out rate
    # batch_size = will defined size of batch, 
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])  #Have to initialize it with 3-D matrices containing only ZERO(0) 
    # We set batch_size=0, column=1 and its value=0, decoder_cell.output_size=0 initially
    #now we have to prepare keys, values, core function, construct function for the attention
    # NB: tensorflow fun core has built in function and construct function as sec2sec subModule in Contrib module for prepare attention
    #*** Now we prepare training data for attention process
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option="bahdanau",
                                                                                                                                    num_units=decoder_cell.output_size)
    # attention_keys = is a key to compare with the target states
    # attention_values = will use to construct the context vectors, and context is return by the Encoder and will used by Decoder --- in first element of Decoding
    # attention_score_function = will use to compute similarity between Keys and Target states
    # attention_construct_function = used to build the attention states
    
    # attention_states = is a variable of top
    # num_units = a argument come from decoder_cell
    
    # Now we will make Training Decoder Function to decoding training set
    # It is also in sec2sec subModule in Contrib module of Tensorflow Library
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                               attention_keys,
                                                                               attention_values,
                                                                               attention_score_function,
                                                                               attention_construct_function,
                                                                               name="attn_dec_train") #It will decode the training set
    # encoder_state[0] = will return from 1st index of encoder_rnn_layer  
    # name = is a NameScope for Decoding Fun which is Tensorflow name Scope
    
    #decoder_output, _, _, =
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope=decoding_scope)
    # NB: decoder_embedded_input, sequence_length = is only use for training time not for prediction. so we we use it into decode_training_set func, but
        # we will not use it in decode_test_set fun.
    # decoder_output = will hold o/p of Dynamic RNN Decoder
    # Dynamic RNN Decoder = this Fun is used to get o/p to final state and final context state of the decoder
    # Dynamic RNN Decoder Function = returns 3 elements but we need only first
    # dynamic_rnn_decoder = is a Function
    
    # decoder_output_dropout = will get o/p from dropout fun of NN Module. which is used for Over feeding and improve accuracy.
    # dropout fun = take 2 arguments, 1)decoder_output 2)keep_prob
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    
    return output_function(decoder_output_dropout)
    

######################################################################################################################################
# Decoding the test / Validation set
######################################################################################################################################
# Here we  do like decode_training_set function but this observation is like new kind of observation of test set and validation set and
    # this are the new observation which not used in training previously. Here we do some cross validation technique for 10% of training set.
    # It will help to increase predictive power on new observation. This is very useful technique to improve accuracy for new observation.   
def decode_test_set(encoder_state, decoder_cell, decoder_embedding_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    # encoder_state = will return from encoder_rnn_layer
    # decoder_cell = is the cell of recurrent nural network of Decoder
    # decoder_embedding_matrix = is the input of decoder. || Goto Tensorflow website->Embeded area
    # sos_id = Start of String token ID
    # eos_id = End of String token ID
    # maximum_length = maximum length of the batch
    # num_words =  is the total number of words in Batch input
    
    # sequence_length = list of the length of each question in the batch
    # decoding_scope =  is object of variable scope|| Goto Tensorflow website->tf.variable_scope area
    # output_function = is a fun will used to return the decoder output
    # keep_prob = is a Regulator which used to control drop out rate
    # batch_size = will defined size of batch, 
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])  #Have to initialize it with 3-D matrices containing only ZERO(0) 
    # We set batch_size=0, column=1 and its value=0, decoder_cell.output_size=0 initially
    #now we have to prepare keys, values, core function, construct function for the attention
    # NB: tensorflow fun core has built in function and construct function as sec2sec subModule in Contrib module for prepare attention
    #*** Now we prepare training data for attention process
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option="bahdanau",
                                                                                                                                    num_units=decoder_cell.output_size)
    # attention_keys = is a key to compare with the target states
    # attention_values = will use to construct the context vectors, and context is return by the Encoder and will used by Decoder --- in first element of Decoding
    # attention_score_function = will use to compute similarity between Keys and Target states
    # attention_construct_function = used to build the attention states. this the attention part of NN of Decoder. It plays the powerful role in 
        # predicting the final outcome. So we have to keep the attention to propagate the signal and get relevent predictions
    
    # attention_states = is a variable of top
    # num_units = a argument come from decoder_cell
    
    # Now we will make Test Decoder Function to decoding test set from attention_decoder_fn_infrance, which test the new observation
        # gives answer of the question deduce logically and perfectly. That mean the Bot is learned by new observation.
    # It is also in sec2sec subModule in Contrib module of Tensorflow Library
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embedding_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name="attn_dec_inf") #It will decode the training set
    # attention_decoder_fn_train = here attention is fun which have power of prediction
    # encoder_state[0] = will return from 1st index of encoder_rnn_layer  
    # name = is a NameScope for Decoding Fun which is Tensorflow name Scope
    
    #decoder_output, _, _, =
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope=decoding_scope)
    # scope=decoding_scope == this is need for test prediction and validation.
    # test_predictions = will hold o/p of Dynamic prediction of new observation
    # Dynamic RNN Decoder = this Fun is used to get o/p to final state and final context state of the decoder
    # Dynamic RNN Decoder Function = returns 3 elements but we need only first
    # dynamic_rnn_decoder = is a Function
    
    # decoder_output_dropout = will get o/p from dropout fun of NN Module
    # dropout fun = take 2 arguments, 1)decoder_output 2)keep_prob
    
    return test_predictions

#################################################################################################################################################
#Creating the Decoder RNN   #Step-23
#################################################################################################################################################
def decoder_rnn(decoder_embedded_input, decoder_embedding_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    # decoder_embedded_input =  is the input of decoder. || Goto Tensorflow website->Embeded area
    # decoder_embedding_matrix = is the input of decoder. || Goto Tensorflow website->Embeded area
    # encoder_state =  will return from encoder_rnn_layer and || encoder_state[0] = will return from 1st index of encoder_rnn_layer
    # num_words = total number of words in our corpus of answer
    # sequence_length = list of the length of each question in the batch
    
    # rnn_size = 
    # num_layers = is the number of layer in RNN Decoder
    # word2int = is a Dictionary type variable/argument
    # keep_prob = is a Regulator which used to control drop out rate. bcz here also some training implimentation of decoder RNN.
    # batch_size = will defined size of batch. This is the heart of any nural network
    
    # NB: Now we will create decoding_scope inside of variable_scope. Then we can do everything inside here. 1st = we will create LSTM Layer.
        # 2nd = 
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size) 
        # tf.contrib.rnn.BasicLSTMCell() = it take only 1 input that is rnn_size
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        # tf.contrib.rnn.DropoutWrapper() = it take 2 argument, 1st is lstm and 2nd is  keep_prob to control dropout
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        # tf.contrib.rnn.MultiRNNCell() = it take lstm_dropout and multiply with number of layers
        
        # here we make fully connected layers with weights and biases
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer();        

        # we make the o/p fun. this fun will stablish fully connection with every layers of RNN decoder.  
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None, # here Normalizer = None
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases) 
        # training_predictions = return value of decode_training_set.
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        # now we are going to cross validation part. 1st we re use the decoding_scope
        decoding_scope.reuse_variables()
        # now we are ready to test prediction. This the return value of decode_test_set
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embedding_matrix,
                                           word2int["<SOS>"], #sos_id = word2int["<SOS>"],
                                           word2int["<EOS>"], # eos_id = word2int["<EOS>"],
                                           sequence_length-1, # maximum_length = sequence_length-1,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        
    return training_predictions, test_predictions


#################################################################################################################################################
#Builing SEQ 2 SEQ Model   #Step-24
#################################################################################################################################################
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionsWord2Int):
    # inputs = is the questions of cornel movie corpus dataset
    # targets = is the answers of the question
    # keep_prob = is a Regulator which used to control drop out rate. bcz here also some training implimentation of decoder RNN.
    # batch_size = size of the batches input

    # answers_num_words = is total_num_words of questions. we used this argument in decoder_rnn function as num_words
    # questions_num_words = is total_num_words of answers
    # encoder_embedding_size = is the number of dimension of the embedding matrix of the Encoder
    # decoder_embedding_size = is the number of dimension of the embedding matrix of the Decoder
    # rnn_size = size of the Recurrent Nural Network
    # num_layers = is number of layer in our decoder cell containing the Stacked LSTM with Dropout applied.
    # questionsWord2Int = is dictionary where we stored the unique integer number for every word of questions. 

    # NB: This function will return the training prediction and test prediction
    # assemblage = putting together. It will assemble encoder_state which is returned by encoder_rnn function 
        # and [training prediction and test prediction] which is returned by decoder_rnn function
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1, # bcz the upper bound of the sequence is excluded
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1)) #will return the embeded input of encoder

    # NB: we have to mind that Output of the Encoder will be the Input of Decoder
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    # here encoder_state = is the output of encoder_rnn
    
    preprocessed_targets = preprocess_targets(targets, questionsWord2Int, batch_size)
    # preprocessed_targets = tis is Pre-Process Targets. Which is used to Back Propagate  between the last prediction and targets

    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1, decoder_embedding_size], 0, 1)) #dimension of the matrix
    # decoder_embeddings_matrix = is an Object of variable class.
    # questions_num_words = number of line will use for input in chatbot
    # tf.random_uniform([questions_num_words+1, decoder_embedding_size], 0, 1) = here 0, 1 is the lower bound and upper bound of random Number
      
    # here to get decoder_embedded_input = we will use decoder_embeddings_matrix
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    #Now we r ready to training prediction and test predictions. Now we have everything to feed decoder_rnn_function
    
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionsWord2Int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
    
    
    
############################################################################################################################################
############################################################################################################################################
########################### PART-3 TRAINING THE SEQ2SEQ MODEL ##############################################################################
############################################################################################################################################
############################################################################################################################################


#################################################################################################################################################
#Setting the Hyper Parameters | Hypo Parameters     #Step-25
#################################################################################################################################################    

epochs = 100        # epochs = time | epochs = is the whole iteration of the Training
# epochs is the whole process of getting the batches of input into NN and propagating them inside the Encoder, For the propagating Encoder States
    # with targets inside the Decoder RNN to Get the Final O/P.

batch_size = 64 # we will start batch_size = 64. If training text is too long then we can Try 128
rnn_size = 512
num_layers = 3  # num_layers = number of layers is the layers number of RNN
encoding_embedding_size = 512  # encoding_embedding_size = is the number of columns in our Embedding Matrix.
# encoding_embedding_size = 512 that means we have 512 columns in our Embedding Matrix.
decoding_embedding_size = 512 # decoding_embedding_size = is the number of columns in our Embedding Matrix.
# decoding_embedding_size = 512 that means we have 512 columns in our Embedding Matrix.
learning_rate = 0.01    # It must not be too high or Low. if High the CB will learn too fast, if low CB will learn too slowly. It should be very tricky    
learning_rate_decay = 0.9  # Reduced Percentage(%) rate in per Iteration of Training. For learning_rate_decay, CB can learn in more depth the logic
                            # of human conversations. that means can Identify the co-relation of dataset.
min_learning_rate = 0.0001  # min_learning_rate = minimum learning rate
keep_probability = 0.5    # keep_prob = (1-Dropout rate). It is the probability of neuron to be present during the training.  
#NB: keep_prob = keep_probability = Only Training time we will apply the Dropout Rate(keep_prob). Never will apply in Test Time bcz all Nurons are present on that time.
    # keep_prob = 20% optimal for input and 50% optimal for Hidden Units. So we will use it for Hidden Units
    # keep_prob is the TF API Name so this name isn't work here so we have to use another name. So we use name = keep_probability here.
  
    
#################################################################################################################################################
# Defining a Tensorflow Session   #Step-26
#################################################################################################################################################    

# Here all the Tensorflow training will be run
# Here we will create a session Object of the Interactive session class of TF. But befor create Object we have to reset the TF graph to ensure that 
    # the graph is ready for the training.
tf.reset_default_graph()
session = tf.InteractiveSession()
    
#################################################################################################################################################
# Loading the Model Inputs   #Step-27
#################################################################################################################################################

# Here we will load inputs into our SEQ2SEQ Model
# For load inputs we already made a function in [Part-2 - BUILDING SEQ2SEQ MODEL]  section. That Function Name is [ def model_inputs() ] which will
    # return for us [ return inputs, targets, lr, keep_prob ] 
inputs, targets, lr, keep_prob = model_inputs()
# inputs = is the questions that will feed into our SEQ2SEQ Network/Model    
# targets = are the answer of the questions  
# lr = is the learning rate    

#################################################################################################################################################
# Settings the Sequence Length   #Step-28
#################################################################################################################################################

# sequence_length is the maximum length which will be 25. we already did this previously in [Part-1 - DATA PRE PROCESSING] section as the variable
    # name is [sorted_clean_questions = [] and sorted_clean_answers = [] ]. and the variables already we used in [Part-2 - BUILDING SEQ2SEQ MODEL] 
    # section as encoder_rnn() function and decoder_rnn() function.
sequence_length = tf.placeholder_with_default(input = 25, # input=25 = sequence_length maximum value
                                              shape = None,
                                              name = "sequence_length") # this placeholder_with_default = is use when O/P is not feed into RNN.
# here sequence_length = not a variable which load functions. Here sequence_length = is a real variable which will use later to integrate.
# sequence_length = 25, mean that "Maximum length of all the input questions will be 25. "


#################################################################################################################################################
# Gettings the Shape of inputs Tensor(Tensor is a 3-D array)   #Step-29
#################################################################################################################################################

input_shape = tf.shape(inputs) # shape takes Tensor as a argument and Returns the shape of this Tensor
# this is the argument of future function. we did for this sequence length
# inputs = is the questions that will feed into our SEQ2SEQ Network/Model    

# NB: Now we are ready to Training and Test of our SEQ2SEQ Models

#################################################################################################################################################
# Gettings the Training and Test Predictions   #Step-30
#################################################################################################################################################



















































    
###########################################################################################################################

