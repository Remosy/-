import tensorflow as tf
import tensornets as nets
import numpy as np

class LearnVideos:
    def __init__(self):
        self.trainData = np.load("videoFrames.npy")
        self.train()


if __name__ == "__main__":

    x = LearnVideos()
    def train(self):
        learningRate = 0.0005
        trainingEpoch = 5
        batchsize = 256
        display_step = 1
        examples_to_show = 10
        # ImageNet input image shape is (244, 244, 3)
        inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])

        # Output is dependent on your situation (10 for CIFAR-10)
        outputs = tf.placeholder(tf.float32, [None, 10])

        # VGG19 returns the last layer (softmax)
        # model to give the name
        logits = nets.VGG19(inputs, is_training=True, classes=10)
        model = tf.identity(logits, name='logits')

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)