from resnet import *

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, Concatenate



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


        P2 = Conv2D(downPSize,(3,3),padding='SAME',name = 'fpn_p2')(P2)
        P3 = Conv2D(downPSize,(3,3),padding='SAME',name = 'fpn_p3')(P3)
        P4 = Conv2D(downPSize,(3,3),padding='SAME',name = 'fpn_p4')(P4)
        P5 = Conv2D(downPSize,(3,3),padding='SAME',name = 'fpn_p5')(P5)
        # print("PyLayers")
        # print(P2,P3,P4,P5)


        head1 = Conv2D(downPSize//2,(3,3),padding='SAME',name = 'fpn_head1')(P2)
        head1 = Conv2D(downPSize//2,(3,3),padding='SAME',name = 'fpn_head1_conv')(head1)

        head2 = Conv2D(downPSize//2,(3,3),padding='SAME',name = 'fpn_head2')(P3)
        head2 = Conv2D(downPSize//2,(3,3),padding='SAME',name = 'fpn_head2_conv')(head2)

        head3 = Conv2D(downPSize//2,(3,3),padding='SAME',name = 'fpn_head3')(P4)
        head3 = Conv2D(downPSize//2,(3,3),padding='SAME',name = 'fpn_head3_conv')(head3)

        head4 = Conv2D(downPSize//2,(3,3),padding='SAME',name = 'fpn_head4')(P5)
        head4 = Conv2D(downPSize//2,(3,3),padding='SAME',name = 'fpn_head4_conv')(head4)

        print("heads")
        print(head1,head2,head3,head4)

        fpn_f_p2 = UpSampling2D(size=(8,8),name = 'fpn_finalP2')(head4)
        fpn_f_p3 = UpSampling2D(size=(4,4),name = 'fpn_finalP3')(head3)
        fpn_f_p4 = UpSampling2D(size=(2,2),name = 'fpn_finalP4')(head2)
        fpn_f_p5 = head1

        x = Concatenate(axis=-1)([fpn_f_p2,fpn_f_p3,fpn_f_p4,fpn_f_p5])
        x = Flatten()(x)
        # x = Dense()

fpnn = fpn()
# fpnn.finalModel(256)