import tensorflow as tf
import numpy as np

#postavljanje hiperparametara
max_len = 40
#broj koraka koje moze da gleda
step = 2
#broj hidden layera
num_units = 128
learning_rate = 0.001
#broj karaktera koji se analiziraju u jednom trenutku
batch_size = 200
#broj epoha za treniranje
epoch = 60
#hiperparametar za verovatnocu
temperature = 0.5
output_data = []
file = open('output.txt', 'w')

def read_data(file_name):
    #otvara i cita tekst fajl
    text = open(file_name, 'r').read()
    return text.lower()

def featurize(text):
    #featurize the text to train and target dataset
    unique_chars = list(set(text))
    len_unique_chars = len(unique_chars)

    input_chars = []
    output_char = []

    for i in range(0, len(text) - max_len, step):
        input_chars.append(text[i:i+max_len])
        output_char.append(text[i+max_len])

    train_data = np.zeros((len(input_chars), max_len, len_unique_chars))
    target_data = np.zeros((len(input_chars), len_unique_chars))

    for i , each in enumerate(input_chars):
        for j, char in enumerate(each):
            train_data[i, j, unique_chars.index(char)] = 1
        target_data[i, unique_chars.index(output_char[i])] = 1
    return train_data, target_data, unique_chars, len_unique_chars

def rnn(x, weight, bias, len_unique_chars):
    #definise rnn celiju i pogadjanje
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, len_unique_chars])
    x = tf.split(x, max_len, 0)
    #deklarisanje LSTM celije
    #forget_bias smanjuje zaboravljanje u ranim stazama ucenja
    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    #deklarisanje RNN
    #static_rnn je skup tensora velicine [batch_size, num_units]
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction

def sample(predicted):
    #pomocna funkcija koja daje primer iz niza verovatnoce
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

def run(train_data, target_data, unique_chars, len_unique_chars):
    #definisanje placeholdera ciju vrednost kasnije menjamo
    x = tf.placeholder("float", [None, max_len, len_unique_chars])
    y = tf.placeholder("float", [None, len_unique_chars])
    #weigh i bias odgovarajuceg oblika
    weight = tf.Variable(tf.random_normal([num_units, len_unique_chars]))
    bias = tf.Variable(tf.random_normal([len_unique_chars]))

    prediction = rnn(x, weight, bias, len_unique_chars)
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    #optimizuje learning rate u zavisnosti od cene
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    num_batches = int(len(train_data)/batch_size)

    for i in range(epoch):
        print("----------- Epoha {0}/{1} -----------".format(i+1, epoch))
        count = 0
        for _ in range(num_batches):
            train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
            count += batch_size
            sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})

        #uzima tekst is training set-a kao seed
        seed = train_batch[:1:]

        #ispisuje seme od 40 karaktera
        seed_chars = ''
        for each in seed[0]:
                seed_chars += unique_chars[np.where(each == max(each))[0][0]]
        print("Seme:", seed_chars)

        #pogadja narednih 1000 karaktera
        for i in range(1000):
            if i > 0:
                remove_fist_char = seed[:,1:,:]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        print('Rezultat:', seed_chars)
        output_data.append(seed_chars)
        file.write("\n%s\n" % seed_chars)
    sess.close()


if __name__ == "__main__":
    text = read_data('results.txt')
    train_data, target_data, unique_chars, len_unique_chars = featurize(text)
    run(train_data, target_data, unique_chars, len_unique_chars)