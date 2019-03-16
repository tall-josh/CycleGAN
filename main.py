import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random
import sys
from tqdm import tqdm

from config import *
from layers import *
from model import *
import json

class CycleGAN():

    def input_setup(self):

        '''
        This function basically sets up variables for taking image input.

        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        '''

        filenames_A = tf.train.match_filenames_once(pattern_A)
        self.queue_length_A = tf.size(filenames_A)

        filenames_B = tf.train.match_filenames_once(pattern_B)
        self.queue_length_B = tf.size(filenames_B)

        filename_queue_A = tf.train.string_input_producer(filenames_A)
        filename_queue_B = tf.train.string_input_producer(filenames_B)

        image_reader = tf.WholeFileReader()
        (self._A, image_file_A) = image_reader.read(filename_queue_A)
        (self._B, image_file_B) = image_reader.read(filename_queue_B)

        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[256,256]),127.5),1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[256,256]),127.5),1)



    def input_read(self, sess):


        '''
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_input/self.B_input -> Stores all the training images in python list
        '''

        # Loading images into the tensors
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_files_A = sess.run(self.queue_length_A)
        num_files_B = sess.run(self.queue_length_B)

        self.fake_images_A = np.zeros((pool_size,1,img_height, img_width, img_layer))
        self.fake_images_B = np.zeros((pool_size,1,img_height, img_width, img_layer))


        self.A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
        self.B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
        paths = {"file_A": [], "file_B" :[]}
        for i in tqdm(range(max_images)):
            image_tensor, path_A = sess.run([self.image_A, self._A])
            if(image_tensor.size == img_size*batch_size*img_layer):
                self.A_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

            paths["file_A"].append(str(path_A))

        for i in tqdm(range(max_images)):
            image_tensor, path_B = sess.run([self.image_B, self._B])
            if(image_tensor.size == img_size*batch_size*img_layer):
                self.B_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

            paths["file_B"].append(str(path_B))

        with open(os.path.join(output_dir, "path_orders.json"), 'w') as f:
            json.dump(paths, f, indent=2)

        coord.request_stop()
        coord.join(threads)

    def model_setup(self):

        ''' This function sets up the model to train

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding self.fake_A/self.fake_B to corresponding generator. This is use to calcualte cyclic loss
        '''

        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")

        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_fake_inputs = 0

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:
            self.fake_B = build_generator_resnet_9blocks(self.input_A, name="g_A")
            self.fake_A = build_generator_resnet_9blocks(self.input_B, name="g_B")
            self.rec_A = build_gen_discriminator(self.input_A, "d_A")
            self.rec_B = build_gen_discriminator(self.input_B, "d_B")

            scope.reuse_variables()

            self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_A")

            scope.reuse_variables()

            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, "d_B")

    def loss_calc(self):

        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))

        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))

        g_loss_A = cyc_loss*10 + disc_loss_B
        g_loss_B = cyc_loss*10 + disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.rec_B,1)))/2.0


        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: print(var.name)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def save_training_images(self, sess, epoch):
        img_dir = os.path.join(output_dir,"imgs")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        for i in range(0,10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})

            _im = Image.fromarray(((fake_A_temp[0]+1)*127.5).astype(np.uint8))
            _im.save(os.path.join(img_dir, "fakeB_"+ str(epoch) + "_" +\
                str(i)+".jpg"))

            _im = Image.fromarray(((fake_B_temp[0]+1)*127.5).astype(np.uint8))
            _im.save(os.path.join(img_dir, "fakeA_"+ str(epoch) + "_" +\
                str(i)+".jpg"))

            _im = Image.fromarray(((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
            _im.save(os.path.join(img_dir, "cycA_"+ str(epoch) + "_" +\
                str(i)+".jpg"))

            _im = Image.fromarray(((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
            _im.save(os.path.join(img_dir, "cycB_"+ str(epoch) + "_" +\
                str(i)+".jpg"))

            _im = Image.fromarray(((self.A_input[i][0]+1)*127.5).astype(np.uint8))
            _im.save(os.path.join(img_dir, "inputA_"+ str(epoch) + "_" +\
                str(i)+".jpg"))

            _im = Image.fromarray(((self.B_input[i][0]+1)*127.5).astype(np.uint8))
            _im.save(os.path.join(img_dir, "inputB_"+ str(epoch) + "_" +\
                str(i)+".jpg"))

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on feeling the pool till it is full and then randomly selects an
        already stored image and replace it with new one.'''

        if(num_fakes < pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake


    def train(self):


        ''' Training Function '''


        # Load Dataset from the dataset folder
        print("input setup: start")
        self.input_setup()
        print("input setup: done\n")

        #Build the network
        print("model setup: start")
        self.model_setup()
        print("model setup: done\n")

        #Loss function calculations
        print("loss calc: start")
        self.loss_calc()
        print("loss calc: done\n")

        # Initializing the global variables
        print("init global vars: start")
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        print("init global vars: done\n")

        print("Session: start")
        with tf.Session() as sess:
            sess.run(tf.initialize_local_variables())
            sess.run(init)

            #Read input to nd array
            self.input_read(sess)

            #Restore the model to run the model from last checkpoint
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(os.path.join(output_dir,"2"))

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step),100):
                print ("In the epoch ", epoch)
                saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100

                if(save_training_images):
                    self.save_training_images(sess, epoch)

                # sys.exit()

                for ptr in range(0,max_images):
                    print("In the iteration ",ptr)
                    print("Starting",time.time()*1000.0)

                    # Optimizing the G_A network

                    _, fake_B_temp, summary_str = sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network
                    _, summary_str = sess.run([self.d_B_trainer, self.d_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_B:fake_B_temp1})
                    writer.add_summary(summary_str, epoch*max_images + ptr)


                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})

                    writer.add_summary(summary_str, epoch*max_images + ptr)


                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run([self.d_A_trainer, self.d_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_A:fake_A_temp1})

                    writer.add_summary(summary_str, epoch*max_images + ptr)

                    self.num_fake_inputs+=1



                sess.run(tf.assign(self.global_step, epoch + 1))

            writer.add_graph(sess.graph)

    def test(self):


        ''' Testing Function'''

        print("Testing the results")

        self.input_setup()

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            sess.run(tf.initialize_local_variables())

            self.input_read(sess)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

            test_img_path = os.path.join(*[output_dir,"imgs","test"])
            if not os.path.exists(test_img_path):
                os.makedirs(test_img_path)

            for i in tqdm(range(max_images)):
                fake_A_temp, fake_B_temp =\
                sess.run([self.fake_A, self.fake_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})

                _im = Image.fromarray(((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                _im.save(os.path.join(test_img_path, "fakeB_"+str(i)+".jpg"))

                _im = Image.fromarray(((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                _im.save(os.path.join(test_img_path, "fakeA_"+str(i)+".jpg"))

                _im = Image.fromarray(((self.A_input[i][0]+1)*127.5).astype(np.uint8))
                _im.save(os.path.join(test_img_path, "inputA_"+str(i)+".jpg"))

                _im = Image.fromarray(((self.B_input[i][0]+1)*127.5).astype(np.uint8))
                _im.save(os.path.join(test_img_path, "inputB_"+str(i)+".jpg"))

def main():


    assert mode in ["train", "test"]
    print("Main setup model")
    model = CycleGAN()
    if mode == "train":
        print("Training\n")
        model.train()
    elif mode == "test":
        print("Testing\n")
        model.test()

if __name__ == '__main__':

    main()
