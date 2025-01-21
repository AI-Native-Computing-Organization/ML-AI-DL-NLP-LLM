    # Building the CNN-VAE model
     
    # Importing the libraries
    import numpy as np
    from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
    from keras.models import Model
    from keras import backend as K
    from keras.callbacks import EarlyStopping
     
    # Setting the input dimensions (size of frames)
    INPUT_DIM = (64,64,3)
     
    # Setting the number of convolutional filters, kernel sizes, strides and activations per layer
    CONV_FILTERS = [32,64,64,128]
    CONV_KERNEL_SIZES = [4,4,4,4]
    CONV_STRIDES = [2,2,2,2]
    CONV_ACTIVATIONS = ['relu','relu','relu','relu']
     
    # Setting the dense layer size
    DENSE_SIZE = 1024
     
    # Setting the layer parameters for the decoder part of the VAE
    CONV_T_FILTERS = [64,64,32,3]
    CONV_T_KERNEL_SIZES = [5,5,6,6]
    CONV_T_STRIDES = [2,2,2,2]
    CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']
     
    # Setting the dimensions of the latent vectors
    Z_DIM = 32
     
    # Setting the number of epochs and batch size
    EPOCHS = 1
    BATCH_SIZE = 32
     
    # Making a function that creates centralized latent vectors for the VAE
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.,stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
     
    # Building the CNN-VAE model within a class
     
    class ConvVAE():
     
        # Initializing all the parameters and variables of the ConvVAE class
        def __init__(self):
            self.models = self._build()
            self.model = self.models[0]
            self.encoder = self.models[1]
            self.decoder = self.models[2]
            self.input_dim = INPUT_DIM
            self.z_dim = Z_DIM
     
        # Building the model
        def _build(self):
            # Creating the model and the encoder inputs
            vae_x = Input(shape=INPUT_DIM)
            # Creating the first convolutional layer of the Encoder
            vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0])(vae_x)
            # Creating the second convolutional layer of the Encoder
            vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0])(vae_c1)
            # Creating the third convolutional layer of the Encoder
            vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0])(vae_c2)
            # Creating the fourth convolutional layer of the Encoder
            vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0])(vae_c3)
            # Flattening the last convolutional layer so we can input it in the dense layers
            vae_z_in = Flatten()(vae_c4)
            # Using two separate files to calculate z_mean and z_log
            vae_z_mean = Dense(Z_DIM)(vae_z_in)
            vae_z_log_var = Dense(Z_DIM)(vae_z_in)
            # Using the Lambda Keras class around the sampling function we created above
            vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
            # Getting the inputs of the decoder part
            vae_z_input = Input(shape=(Z_DIM,))
            # Instantiating these layers separately so as to reuse them later
            vae_dense = Dense(1024)
            vae_dense_model = vae_dense(vae_z)
            # Reshaping the dense layer to 4 dimentions, so we can put it through the transposed convolution
            vae_z_out = Reshape((1,1,DENSE_SIZE))
            # Getting the output from this last layer
            vae_z_out_model = vae_z_out(vae_dense_model)
            # Defining the first transposed convolutional layer
            vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])
            # Creating the first decoder layer
            vae_d1_model = vae_d1(vae_z_out_model)
            # Defining the second transposed convolutional layer
            vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])
            # Creating the second decoder layer
            vae_d2_model = vae_d2(vae_d1_model)
            # Defining the third convolutional layer
            vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])
            # Creating the third decoder layer
            vae_d3_model = vae_d3(vae_d2_model)
            # Defining the fourth convolutional layer
            vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])
            # Creating the fourth decoder layer
            vae_d4_model = vae_d4(vae_d3_model)
            # Getting the latent vector output of the decoder
            vae_dense_decoder = vae_dense(vae_z_input)
            vae_z_out_decoder = vae_z_out(vae_dense_decoder)
            vae_d1_decoder = vae_d1(vae_z_out_decoder)
            vae_d2_decoder = vae_d2(vae_d1_decoder)
            vae_d3_decoder = vae_d3(vae_d2_decoder)
            vae_d4_decoder = vae_d4(vae_d3_decoder)
            # Defining the end-to-end VAE Model, composed of both the encoder and the decoder
            vae = Model(vae_x, vae_d4_model)
            vae_encoder = Model(vae_x, vae_z)
            vae_decoder = Model(vae_z_input, vae_d4_decoder)
            # Implementing the training operations
            # Defining the MSE loss
            def vae_r_loss(y_true, y_pred):
                y_true_flat = K.flatten(y_true)
                y_pred_flat = K.flatten(y_pred)
                return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)
            # Defining the KL divergence loss
            def vae_kl_loss(y_true, y_pred):
                return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)
            # Defining the total VAE loss, summing the MSE and KL losses
            def vae_loss(y_true, y_pred):
                return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)
            # Compiling the whole model with the RMSProp optimizer, the vae loss and custom metrics
            vae.compile(optimizer='rmsprop', loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])
            return (vae, vae_encoder, vae_decoder)
     
        # Loading the model
        def set_weights(self, filepath):
            self.model.load_weights(filepath)
     
        # Creating early stopping callbacks to prevent overfitting
        def train(self, data, validation_split = 0.2):
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
            callbacks_list = [earlystop]
            self.model.fit(data, data,
                           shuffle=True,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           validation_split=validation_split,
                           callbacks=callbacks_list)
            self.model.save_weights('vae/weights.h5')
     
        # Saving the model
        def save_weights(self, filepath):
            self.model.save_weights(filepath)
        
        # Generating data for the MDN-RNN
        def generate_rnn_data(self, obs_data, action_data):
            rnn_input = []
            rnn_output = []
            for i, j in zip(obs_data, action_data):    
                rnn_z_input = self.encoder.predict(np.array(i))
                conc = [np.concatenate([x,y]) for x, y in zip(rnn_z_input, j.reshape(-1, 1))]
                rnn_input.append(conc[:-1])
                rnn_output.append(np.array(rnn_z_input[1:]))
            rnn_input = np.array(rnn_input)
            rnn_output = np.array(rnn_output)
            print("Rnn inputs size: {}".format(rnn_input.shape), " Rnn outputs size: {}".format(rnn_output.shape))
            return (rnn_input, rnn_output)
