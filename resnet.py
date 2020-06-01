import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.initializers import glorot_uniform


class Resnet():

    def __init__(self,input_shape=(64,64,3),classes=6):
        self.input_shape = input_shape
        self.num_classes = classes
        self.stages = []
        self.filter_size = [[64, 64, 256],[128,128,512],[256, 256, 1024],[512, 512, 2048]]
    
    def get_resnet(self):
        self.setupResnet()
        return self.stages

    def convolutional_block(self,X,f,filters,stage,block,s = 2):
            # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X


        ##### MAIN PATH #####
        # First component of main path 
        X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        
        ### START CODE HERE ###

        
        # Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X_shortcut,X])
        X = Activation('relu')(X)
        
        ### END CODE HERE ###
        
        return X


    def identity_block(self, X, f, filters, stage, block):

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X_shortcut,X])
        X = Activation('relu')(X)
        
        return X


    def setupResnet(self):
        X_input = Input(self.input_shape)
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        self.stages.append(X)

        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)
        self.stages.append(X)

        # Stage 2
        X = self.convolutional_block(X,3,self.filter_size[0], stage = 2, block='a', s = 2)
        X = self.identity_block(X, 3, self.filter_size[0], stage=2, block='b')
        X = self.identity_block(X, 3, self.filter_size[0], stage=2, block='c')
        self.stages.append(X)
        # print(X)
        # Stage 3 (≈4 lines)
        X = self.convolutional_block(X,3,self.filter_size[1], stage = 3, block='a', s = 2)
        X = self.identity_block(X, 3, self.filter_size[1], stage=3, block='b')
        X = self.identity_block(X, 3, self.filter_size[1], stage=3, block='c')
        X = self.identity_block(X, 3, self.filter_size[1], stage=3, block='d')
        self.stages.append(X)
        # Stage 4 (≈6 lines)

        X = self.convolutional_block(X,3,self.filter_size[2], stage = 4, block='a', s = 2)
        X = self.identity_block(X, 3, self.filter_size[2], stage=4, block='b')
        X = self.identity_block(X, 3, self.filter_size[2], stage=4, block='c')
        X = self.identity_block(X, 3, self.filter_size[2], stage=4, block='d')
        X = self.identity_block(X, 3, self.filter_size[2], stage=4, block='e')
        X = self.identity_block(X, 3, self.filter_size[2], stage=4, block='f')
        self.stages.append(X)

        # Stage 5 (≈3 lines)
        X = self.convolutional_block(X,3, self.filter_size[3], stage = 5, block='a', s = 2)
        X = self.identity_block(X, 3, self.filter_size[3], stage=5, block='b')
        X = self.identity_block(X, 3, self.filter_size[3], stage=5, block='c')
        self.stages.append(X)