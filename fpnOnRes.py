from resnet import *

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.layers.merge import concatenate, add

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)




class fpn:

    def __init__(self):
        self.resBlock = Resnet()
        self.resBlock.setupResnet()
        self.pyramidLayers = []
        self.__get_pyramidLayers(256)
        # resBlock.get_resnet()

    def __get_pyramidLayers(self,downPSize):
        # downPSize = 256
        stages = self.resBlock.get_resnet()
        # for stage in stages:
        #     pLayer = Conv2D(downPSize,stride = (1,1))(stage)

        P5 = Conv2D(downPSize,(1,1),padding='SAME',name = 'p5')(stages[5])
        P4 = Conv2D(downPSize,(1,1),padding='SAME',name = 'p4')(stages[4])
        P4 = Add()([UpSampling2D(name = 'UPp4')(P5),P4])
        P3 = Conv2D(downPSize,(1,1),padding='SAME',name = 'p3')(stages[3])
        P3 = Add()([UpSampling2D(name = 'UPp3')(P4),P3])
        P2 = Conv2D(downPSize,(1,1),padding='SAME',name = 'p2')(stages[2])
        P2 = Add()([UpSampling2D(name = 'UPp2')(P3),P2])
        # print("Beforsss")
        # print(P2,P3,P4,P5)


        P2 = Conv2D(downPSize,(3,3),padding='SAME')(P2)
        P3 = Conv2D(downPSize,(3,3),padding='SAME')(P3)
        P4 = Conv2D(downPSize,(3,3),padding='SAME')(P4)
        P5 = Conv2D(downPSize,(3,3),padding='SAME')(P5)
        print("PyLayers")
        print(P2,P3,P4,P5)


        head1 = Conv2D(downPSize//2,(3,3),padding='SAME')(P2)
        head1 = Conv2D(downPSize//2,(3,3),padding='SAME')(head1)

        head2 = Conv2D(downPSize//2,(3,3),padding='SAME')(P3)
        head2 = Conv2D(downPSize//2,(3,3),padding='SAME')(head2)

        head3 = Conv2D(downPSize//2,(3,3),padding='SAME')(P4)
        head3 = Conv2D(downPSize//2,(3,3),padding='SAME')(head3)

        head4 = Conv2D(downPSize//2,(3,3),padding='SAME')(P5)
        head4 = Conv2D(downPSize//2,(3,3),padding='SAME')(head4)

        print("heads")
        print(head1,head2,head3,head4)

        p2 = UpSampling2D(size=(8,8))(head4)
        p3 = UpSampling2D(size=(4,4))(head3)
        p4 = UpSampling2D(size=(2,2))(head2)
        p5 = head1

        x = Concatenate(axis=-1)([p2,p3,p4,p5])
        x = Flatten()(x)
        # x = Dense()

fpnn = fpn()
# fpnn.finalModel(256)