from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy
import time
import os

def cleaning(words, using_allowed_words=True):

    words = words.lower()
    words = StemmerFactory().create_stemmer().stem(words)
    words = words.split(' ')

    allowed_words = []

    if(using_allowed_words):
        with open("allowedwords.txt") as file:
            content = file.readlines()
            allowed_words = [x.strip() for x in content]
    else:
        allowed_words = words

    clean_words = []

    for word in words:
        if word in allowed_words:
            clean_words.append(word)

    return clean_words

def bag_of_words(bag, words):
    bow = []
    for b in bag:
        bow.append(words.count(b))
    return bow

def sigmoid(value):
    return 1/(1+numpy.exp(-value))

def sigmoid_output_to_derivative(value):
    return value*(1-value)

def generate_synapse(input_and_target, hidden_neurons = 3, alpha=1, epochs=100, is_training=True):

    input_list = []
    output_list = []

    for x in input_and_target:
        input_list.append(x['input'])
        output_list.append(x['target'])
    
    input_list = numpy.array(input_list)
    output_list = numpy.array(output_list)

    if(is_training):
        print("Total Neuron Input  : %s" % len(input_list))
        print("Total Neuron Hidden : %s" % hidden_neurons)

    numpy.random.seed(1)

    last_mean_error = 1

    synapse_0 = 2*numpy.random.random((len(input_and_target[0]['input']), hidden_neurons)) - 1
    synapse_1 = 2*numpy.random.random((hidden_neurons, len(input_and_target[0]['target']))) - 1

    synapse_0_weight_update_prev = numpy.zeros_like(synapse_0)
    synapse_1_weight_update_prev = numpy.zeros_like(synapse_1)

    synapse_0_weight_update_next = numpy.zeros_like(synapse_0)
    synapse_1_weight_update_next = numpy.zeros_like(synapse_1)

    for i in iter(range(epochs+1)):
        
        layer_0 = input_list
        layer_1 = sigmoid(numpy.dot(layer_0,synapse_0))
        layer_2 = sigmoid(numpy.dot(layer_1,synapse_1))

        error_layer_2 = output_list - layer_2
        delta_layer_2 = error_layer_2 * sigmoid_output_to_derivative(layer_2)

        error_layer_1 = delta_layer_2.dot(synapse_1.T)
        delta_layer_1 = error_layer_1 * sigmoid_output_to_derivative(layer_1)

        if (i% 1000) == 0:
            err_iteration = numpy.mean(numpy.abs(error_layer_2))
            if err_iteration < last_mean_error:
                print ("delta after "+str(i)+" iterations:" + str("{:10.10f}".format(err_iteration))+ " or "+str("{:10.10f}".format(1-err_iteration)))
                last_mean_error = err_iteration
            else:
                print ("break:", err_iteration, ">", last_mean_error )
                break

        synapse_1_weight_update = (layer_1.T.dot(delta_layer_2))
        synapse_0_weight_update = (layer_0.T.dot(delta_layer_1))
        
        if(i>0):
            synapse_0_weight_update_next += numpy.abs(((synapse_0_weight_update > 0)+0) - ((synapse_0_weight_update_prev > 0) + 0))
            synapse_1_weight_update_next += numpy.abs(((synapse_1_weight_update > 0)+0) - ((synapse_1_weight_update_prev > 0) + 0)) 

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        synapse_0_weight_update_prev = synapse_0_weight_update
        synapse_1_weight_update_prev = synapse_1_weight_update

    synapse = {
        'synapse_0' : synapse_0.tolist(),
        'synapse_1' : synapse_1.tolist()
    }

    return synapse

def generate_training(training_datas, hidden_neurons = 3, alpha=1, epochs=100, use_allowed_words=True, is_debugging=True):

    classes = []
    words = []

    if(is_debugging):
        print("Total training data: %s"%len(training_datas))
        print("")
    
    for d in training_datas:
        d['values'] = cleaning(d['value'], use_allowed_words)
        if(is_debugging):
            print(d['values'])
            print("")

    for d in training_datas:
        if d['class'] not in classes:
            classes.append(d['class'])
        for w in d['values']:
            if w not in words:
                words.append(w)
    
    if(is_debugging):
        print("classes: ", len(classes))
        print(classes)
        print("")
        print("Words: ", len(words))
        print(words)
        print("")

    input_and_target = []

    for d in training_datas:
        documentBow = bag_of_words(words,d['value'])
        documentClass = []
        for c in classes:
            documentClass.append(1) if c==d['class'] else documentClass.append(0)
        
        print("")
        print("BoW")
        print(documentBow)
        print("Class")
        print(documentClass)
        print("")

        input_and_target.append({
            "target" : documentClass,
            "input" : documentBow
        })
    
    return {
        'words' : words,
        'classes': classes,
        'use_allowed_words': use_allowed_words,
        'input_and_target' : input_and_target,
        'synapse' : generate_synapse(input_and_target,hidden_neurons,alpha,epochs,is_debugging)
    }


def classify(training, sentence, is_debugging=True):
    
    wordList = cleaning(sentence, training['use_allowed_words'])
    classList = training['classes']
    trainingWordList = training['words']

    bow = []
    for x in trainingWordList:
        bow.append(wordList.count(x))

    bow = numpy.array(bow)

    layer_0 = bow 
    layer_1 = sigmoid(numpy.dot(layer_0,training['synapse']['synapse_0']))
    layer_2 = sigmoid(numpy.dot(layer_1,training['synapse']['synapse_1']))

    print("")
    print("")
    print("Classify")
    print(wordList)
    print(layer_2)

    response = []

    i = 0
    for x in classList:
        response.append({
            'class': x,
            'value': "{:10.10f}".format(layer_2[i])
        })
        i = i+1

    return response

training_datas = [
    {
        'class':'positive',
        'value':'selamat pagi, nama saya andi muqsith ashari'
    },
    {
        'class':'positive',
        'value':'hai! apa kabar semoga harimu cerah dan indah!'
    },
    {
        'class':'positive',
        'value':'senang sekali bisa berjumpa denganmu! aku sangat merindukanmu'
    },
    {
        'class':'negative',
        'value':'aku tidak suka denganmu'
    },
    {
        'class':'negative',
        'value':'saya mau membuang air besar'
    },
    {
        'class':'negative',
        'value':'aku sangat sangat kesepian'
    }
]

training = generate_training(training_datas, 3, 1, 50000, False, True)

classify_results = []

for x in training_datas:
    classify_results.append(classify(training,x['value']))

print("")
print("")
print("Testing Final result:")
for x in classify_results:
    print(x)

print("")
print("")
print("Classify Final result:")
classify(training,"aku sangat senang bisa berjumpa dan bertemu denganmu")
classify(training,"bagaimana kabarmu siang ini aku sangat merindukanmu")
classify(training,"dia tidak suka denganku")
classify(training,"dia membenciku dan tidak menyukaiku")