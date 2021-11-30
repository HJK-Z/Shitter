import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model, save_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv3D, Conv2D, Conv1D, Dropout, MaxPooling2D, LeakyReLU, BatchNormalization, Reshape, InputLayer, Input, ZeroPadding3D, ZeroPadding2D, UpSampling2D, UpSampling3D, MaxPool3D, MaxPooling3D
from tensorflow.keras.optimizers import Adam
import os, time
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import imageio
import tensorflow.keras.backend as K

CWD = os.getcwd()  # Set current working directory
# Video params
img_rows = 24
img_cols = 24
channels = 30  # Depth but can also be used for color channels
frames_per_batch = 30   # Frames per training example (s/b sequential)
NOISE_VALUES = 100     # Noise input array size
img_shape = (img_rows, img_cols, channels)  # If not color channels = # frames per examaple

# Input paths
INPUT_VIDEOS_PATH = os.path.join(CWD, r'Data\Input_Videos')
SAVED_OUTPUT_PATH = r'Data\Saved_Output'

# Training data paths
TRAINING_DATA_PATH = os.path.join(CWD, r'Data\Training_Data_Video\x_training_data.npy')

# Misc params
MODEL_SAVE_FOLDER = r'Models\Saved_GAN'      # Path where models will be saved
CREATE_TRAINING_DATA = True    # True = new numpy file. False will use existing file @ TRAINING_DATA_PATH

USE_EXISTING_DISC = False
USE_EXISTING_GEN = False

class GAN():
    def __init__(self):
        # Training existing Models (False will create new models)
        self.EXISTING_DISCRIMINATOR = os.path.join(MODEL_SAVE_FOLDER, 'discriminator.hdf5')
        self.EXISTING_GENERATOR = os.path.join(MODEL_SAVE_FOLDER, 'generator.hdf5')

        self.IMAGE_COUNTER = 0

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        if USE_EXISTING_DISC and os.path.exists(self.EXISTING_DISCRIMINATOR):
            self.discriminator = load_model(self.EXISTING_DISCRIMINATOR)
        else:
            self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        if USE_EXISTING_GEN and os.path.exists(self.EXISTING_GENERATOR):
            self.generator = load_model(self.EXISTING_GENERATOR)
        else:
            self.generator = self.build_generator()

        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generates video frames
        z = Input(shape=(NOISE_VALUES,))
        img = self.generator(z)

        # For the combined model, only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load_image_file(self, path_to_image):
        img = Image.open(path_to_image).convert('RGB')
        w, h = img.size
        return img, w, h

    def create_numpy_dataset_from_video(self):
        x_data = np.array([])
        x_data_batch = np.array([])

        # Read images frames
        listdirectory = os.listdir(INPUT_VIDEOS_PATH)
        training_list = []
        total_batches = 1
        for video in listdirectory:
            frames = []
            videopath = os.path.join(INPUT_VIDEOS_PATH, video)
            loadedvideo = cv2.VideoCapture(videopath)
            success, img = loadedvideo.read()
            
            count = 0
            batch_count = 0

            # This loop converts real input video frames in to sets of N frames and finally to numpy array
            while success:
                count += 1
                batch_count += 1
                #cv2.imwrite("frame%d.jpg" % count, img)  # save frame as JPEG file

                img = cv2.resize(img, (img_rows, img_cols))
                # w/o color channel
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # If first frame in batch
                if batch_count == 1:
                    x_data_batch = img
                    x_data_batch = np.reshape(x_data_batch, (x_data_batch.shape[0], x_data_batch.shape[1], 1))
                else:
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                    x_data_batch = np.dstack([x_data_batch, img])

                x_data_batch_v = np.reshape(x_data_batch, (1, x_data_batch.shape[0], x_data_batch.shape[1], x_data_batch.shape[2]))

                if batch_count == frames_per_batch:
                    if total_batches == 1:
                        x_data = x_data_batch_v
                        batch_count = 0
                        total_batches += 1
                    else:
                        x_data = np.vstack((x_data, x_data_batch_v))
                        total_batches += 1  
                        batch_count = 0     

                success, img = loadedvideo.read()

        # Save to numpy files
        print('Saving x data...')
        np.save(TRAINING_DATA_PATH, x_data)


    ######## GENERATOR ########

    def build_generator(self):

        noise_shape = (NOISE_VALUES,)

        model = Sequential()

        model.add(InputLayer(input_shape=noise_shape, name='Input_1'))
        model.add(Dense(5 * 4 * 5 * 1, activation="relu", name='Dense_1'))
        model.add(Reshape((5, 4, 5, 1), name='Reshape_1'))
        model.add(UpSampling3D())  # Transforms small input to a large image output
        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', name='Conv3D_1'))
        model.add(BatchNormalization(momentum=0.8))  # Improves performance/stability and helps generalize
        model.add(UpSampling3D())
        model.add(Conv3D(64, kernel_size=(2, 2, 2), activation='relu', name='Conv3D_2'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform', name='Dense_2'))
        model.add(Dense(np.prod(img_shape), activation='tanh', name='Dense_Output'))  # Size of samples
        model.add(Reshape(img_shape))  # Reshape to our video sample dimensions
        model.summary()

        noise = Input(shape=noise_shape)  # Create Input for Model
        img = model(noise)  # Create Output for Model

        return Model(noise, img)

    ###############################

    ######## DISCRIMINATOR ########

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(128, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    ###############################

    def train_my_gan(self, epochs, batch_size=128, save_interval=50, save_model_epoch=5000):
        # Create training data
        if CREATE_TRAINING_DATA:
            self.create_numpy_dataset_from_video()

        # Load the dataset
        X_train = np.load(TRAINING_DATA_PATH)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            #print(idx)
            imgs = X_train[idx]
            #print('imgs.shape', imgs.shape)

            noise = np.random.normal(0, 1, (half_batch, NOISE_VALUES))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, NOISE_VALUES))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_my_imgs(epoch)

            if save_model_epoch == epoch and save_model_epoch > 0:
                self.save_my_gan()

    def save_my_imgs(self, epoch):
        self.IMAGE_COUNTER += 1
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, NOISE_VALUES))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Display/save sample image from generator
        file_name = 'generator_sample_' + str(self.IMAGE_COUNTER) + '.png'
        file_path = os.path.join(SAVED_OUTPUT_PATH, file_name)
        save_img = gen_imgs[0, :, :, 0].copy()
        save_img = save_img * 255
        save_img = save_img.astype(np.uint8)
        cv2.imwrite(file_path, save_img)

        temp_img = gen_imgs[0, :, :, 0].copy()
        temp_img = cv2.resize(temp_img, (temp_img.shape[1] * 4, temp_img.shape[0] * 4))
        cv2.imshow('Sample Img', temp_img)
        cv2.waitKey(1)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                #axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        sample_frame_file_name = 'training_sample_frames_epoch_' + str(epoch) + '.png'
        sample_frame_file_path = os.path.join(SAVED_OUTPUT_PATH, sample_frame_file_name)
        fig.savefig(sample_frame_file_path)
        plt.close()

    def save_my_gan(self):
        print('Saving GAN models to:', MODEL_SAVE_FOLDER)
        save_model(self.discriminator, os.path.join(MODEL_SAVE_FOLDER, 'discriminator.hdf5'), overwrite=True)
        print('Saved discriminator.h5...')
        save_model(self.generator, os.path.join(MODEL_SAVE_FOLDER, 'generator.hdf5'), overwrite=True)
        print('Saved generator.h5...')
        save_model(self.combined, os.path.join(MODEL_SAVE_FOLDER, 'combined.hdf5'), overwrite=True)
        print('Saved combined.h5...')


# Main program
if __name__ == '__main__':
    # GAN Class
    gan = GAN()

    # Control Params
    train_gan = True       # Train a new GAN
    gen_gan = True          # Run the GAN

    # Run GAN Model Params
    generator = None
    FRAMES = 300      # Frames per video
    
    # extra options
    display = False
    output_size = (680, 480)

    # Train a new GAN
    if train_gan:
        print(86 * '=')
        gan.train_my_gan(epochs=2000, batch_size=32, save_interval=100, save_model_epoch=500)
        print(86 * '=')
        gan.save_my_gan()
        print(86 * '=')

    # Run the trained GAN
    if gen_gan:
        print(86 * '=')
        print('Loading saved generator model...')
        generator = load_model(os.path.join(MODEL_SAVE_FOLDER, 'generator.hdf5'))

        print('Generating noise for GAN...')
        # Create random noise as input
        noise = np.random.normal(0, 1, (FRAMES, NOISE_VALUES))

        print('Running inference to generate frames...')
        # Generate video output
        gen_imgs = generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        result = imageio.get_writer('video.mp4', format='FFMPEG', mode='I', fps=10)

        # Display/Save Output
        for i in range(FRAMES):
            temp_img = gen_imgs[i, :, :, 0].copy()
            temp_img = cv2.resize(temp_img, output_size)
            temp_img = cv2.normalize(temp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            result.append_data(temp_img)
            if display:
                cv2.imshow('Generated Video', temp_img)
                cv2.waitKey(2)

        result.close()
            
print('Done')

