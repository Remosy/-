import tensorflow as tf
learningRate = 0.0005
trainingEpoch = 5
batchsize = 256
display_step = 1
examples_to_show = 10
inputNum = 100800 #210*160*3
X = tf.placeholder("int", [None, inputNum])
# hidden layer settings
n_hidden_1 = 318
n_hidden_2 = 18 #18 actions
weights = {
        'encoder_h1': tf.Variable(tf.random_normal([inputNum, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
}
biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
}

class LearnVideos:
    def __init__(self, videoFrames):
        self.trainingDate = videoFrames
        self.train()

    def train(self):
        theEncoder = self.encoder(X)
        with tf.Session() as sess:
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
           
        return ""

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2