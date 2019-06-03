''' 
Network architecture for EDVR in Keras
Based on the:

Winning Solution in NTIRE19 Challenges on Video Restoration and Enhancement (CVPR19 Workshops) - 
Video Restoration with Enhanced Deformable Convolutional Networks 
https://xinntao.github.io/projects/EDVR
https://github.com/xinntao/EDVR

Jenia Golbstein, June-2019
'''


import numpy as np
from keras.layers import *
from keras import backend as K
from subpixel import *
from keras.models import Model


class EDVR:
        
        
    def __init__(self, inp_shape=(256, 256, 3), nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None, predeblur=False, HR_in=False):
        self.H, self.W, self.C = inp_shape
        self.is_predeblur = True if predeblur else False
        self.center = nframes // 2 if center is None else center
        self.nf = nf
        self.nframes = nframes
        self.groups = groups
        self.front_RBs = front_RBs
        self.back_RBs = back_RBs
        self.HR_in = HR_in
        
    def __ResidualBlock_noBN(self, x):
        identity = x
        out = Conv2D(self.nf, kernel_size=3, padding='same', activation='relu')(x)
        out = Conv2D(self.nf, kernel_size=3, padding='same')(x)
        return add([identity, out])
    
    def __conv1_block(self, x):
        x = Conv2D(self.nf, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def __conv2_block(self, x):
        x = Lambda(lambda x: K.spatial_2d_padding(x))(x)
        x = Conv2D(self.nf, 3, strides=2, padding='valid')(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def __get_center_layer(self, x):
        center_layer = Lambda(lambda x: x[:, self.center, :, :, :])(x)
        return center_layer
    
    def __Predeblur_ResNet_Pyramid(self, x):
        L1_fea = self.__conv1_block(x)
        if self.HR_in:
            for i in range(2):
                L1_fea = self.__conv2_block(L1_fea)
        L2_fea = self.__conv2_block(L2_fea)
        L3_fea = self.__conv2_block(L2_fea)
        L3_fea = self.__ResidualBlock_noBN(L3_fea)
        L3_fea = UpSampling2D(interpolation='bilinear')(L3_fea)
        L2_fea = add([self.__ResidualBlock_noBN(L2_fea), L3_fea])
        L2_fea = self.__ResidualBlock_noBN(L2_fea)
        L2_fea = UpSampling2D(interpolation='bilinear')(L2_fea)
        L1_fea = add([self.__ResidualBlock_noBN(L1_fea), L2_fea])
        out = self.__ResidualBlock_noBN(L1_fea)
        for i in range(2):
            out = ResidualBlock_noBN(out)
        return out

    def __PCD_Align(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,H,W,C] features
        '''
        # L3
        L3_offset = Concatenate()([nbr_fea_l[2], ref_fea_l[2]])
        for _ in range(2):
            L3_offset = Conv2D(self.nf, 3, padding='same')(L3_offset)
            L3_offset = LeakyReLU(alpha=.1)(L3_offset)

        # Deformable Conv Layer Should be here and take (nbr_fea_l[2], L3_offset) as input
        L3_fea = Conv2D(self.nf, 3, padding='same')(L3_offset)
        #     L3_fea = DeformableConvLayer(nf, 3, strides=1, 
        #                                  padding='same', dilation_rate=1, 
        #                                  num_deformable_group=groups)(nbr_fea_l[2], L3_offset)

        # L2
        L2_offset = Concatenate()([nbr_fea_l[1], ref_fea_l[1]])
        
        L2_offset = self.__conv1_block(L2_offset)
        L3_offset = UpSampling2D(interpolation='bilinear')(L3_offset)
        L3_offset = Lambda(lambda x: x*2)(L3_offset)
        concat_offset = Concatenate()([L2_offset, L3_offset])
        
        L2_offset = self.__conv1_block(concat_offset)
        L2_offset = self.__conv1_block(L2_offset)

        # Deformable Conv Layer Should be here and take (nbr_fea_l[1], L2_offset) as input
        L2_fea = Conv2D(self.nf, 3, padding='same')(L2_offset)
        #     L2_fea = DeformableConvLayer(nf, 3, strides=1, 
        #                                  padding='same', dilation_rate=1, 
        #                                  num_deformable_group=groups)(nbr_fea_l[1], L2_offset)
        L3_fea = UpSampling2D(interpolation='bilinear')(L3_fea)
        concat_fea = Concatenate()([L2_fea, L3_fea])
        L2_fea = self.__conv1_block(concat_fea)

        # L1
        L1_offset = Concatenate()([nbr_fea_l[0], ref_fea_l[0]])
        L1_offset = self.__conv1_block(L1_offset)
        L2_offset = UpSampling2D(interpolation='bilinear')(L2_offset)
        L2_offset = Lambda(lambda x: x*2)(L2_offset)
        concat_offset = Concatenate()([L1_offset, L2_offset])
        L1_offset = self.__conv1_block(concat_offset)
        L1_offset = self.__conv1_block(L1_offset)

        # Deformable Conv Layer Should be here and take (nbr_fea_l[0], L1_offset) as input
        L1_fea = Conv2D(self.nf, 3, padding='same')(L1_offset)
        #     L1_fea = DeformableConvLayer(nf, 3, strides=1, 
        #                                  padding='same', dilation_rate=1, 
        #                                  num_deformable_group=groups)(nbr_fea_l[0], L1_offset)
        L2_fea = UpSampling2D(interpolation='bilinear')(L2_fea)
        concat_fea = Concatenate()([L1_fea, L2_fea])
        L1_fea = self.__conv1_block(concat_fea)

        # Cascading
        offset = Concatenate()([L1_fea, ref_fea_l[0]])
        for _ in range(2):
            offset = self.__conv1_block(offset)
        # Deformable Conv Layer Should be here and take (L1_fea, offset) as input
        L1_fea = Conv2D(self.nf, 3, padding='same')(offset)
        #     L1_fea = DeformableConvLayer(nf, 3, strides=1, 
        #                                  padding='same', dilation_rate=1, 
        #                                  num_deformable_group=groups)(L1_fea, offset)
        L1_fea = LeakyReLU(alpha=.1)(L1_fea)
        return L1_fea

    def __TSA_Fusion(self, aligned_fea):
        ''' Temporal Spatial Attention fusion module
        Temporal: correlation;
        Spatial: 3 pyramid levels.
        '''
        B, N, H, W, C = K.int_shape(aligned_fea)
        #### temporal attention

        emb_ref = Conv2D(self.nf, 3, padding='same')(self.__get_center_layer(aligned_fea))
        reshaped_fea = Lambda(lambda x: K.reshape(x, (-1, H, W, C)))(aligned_fea)
        reshaped_emb = Conv2D(self.nf, 3, padding='same')(reshaped_fea)
        emb = Lambda(lambda x: K.reshape(x, (-1, N, H, W, C)))(reshaped_emb)
        cor_l = []
        for i in range(N):
            emb_nbr = Lambda(lambda x: x[:, i, :, :, :])(emb)
            m_emb = Multiply()([emb_nbr, emb_ref])
            cor_tmp = Lambda(lambda x: K.sum(x, axis=-1))(m_emb)
            cor_tmp = Lambda(lambda x: K.expand_dims(x, -1))(cor_tmp)
            cor_l.append(cor_tmp)
        cor_prob = Lambda(lambda x: K.sigmoid(x))(Concatenate()(cor_l))
        cor_prob = Lambda(lambda x: K.expand_dims(x, axis=-1))(cor_prob)
        cor_prob = Concatenate()(C*[cor_prob])
        cor_prob = Lambda(lambda x: K.reshape(x, (-1, H, W, C*N)))(cor_prob)
        aligned_fea = Lambda(lambda x: K.reshape(x, (-1, H, W, C*N)))(aligned_fea)
        aligned_fea = Multiply()([aligned_fea, cor_prob])

        #### fusion
        fea = Conv2D(self.nf, 1)(aligned_fea)
        fea = LeakyReLU(alpha=.1)(fea)
        #### spatial attention
        att = Conv2D(self.nf, 1)(aligned_fea)
        att = LeakyReLU(alpha=.1)(att)
        att_max = Lambda(lambda x: K.spatial_2d_padding(x))(att)
        att_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(att_max)
        att_avg = Lambda(lambda x: K.spatial_2d_padding(x))(att)
        att_avg = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(att_avg)
        cat_max_avg = Concatenate()([att_max, att_avg])
        cat_max_avg = Conv2D(self.nf, 1)(cat_max_avg)
        att = LeakyReLU(alpha=.1)(cat_max_avg)
        # pyramid levels
        att_L = Conv2D(self.nf, 1)(att)
        att_L = LeakyReLU(alpha=.1)(att_L)
        att_max = Lambda(lambda x: K.spatial_2d_padding(x))(att_L)
        att_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(att_max)
        att_avg = Lambda(lambda x: K.spatial_2d_padding(x))(att_L)
        att_avg = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(att_avg)
        cat_max_avg = Concatenate()([att_max, att_avg])
        att_L = self.__conv1_block(cat_max_avg)
        att_L = UpSampling2D(interpolation='bilinear')(att_L)
        att = self.__conv1_block(att)
        att = add([att, att_L])
        att = Conv2D(self.nf, 1)(att)
        att = LeakyReLU(alpha=.1)(att)
        att = UpSampling2D(interpolation='bilinear')(att)
        att = Conv2D(self.nf, 3, padding='same')(att)
        att_add = Conv2D(self.nf, 1)(att)
        att_add = LeakyReLU(alpha=.1)(att_add)
        att_add = Conv2D(self.nf, 1)(att_add)
        att = Lambda(lambda x: K.sigmoid(x))(att)
        fea = Lambda(lambda x: x[0]*x[1]*2 + x[2])([fea, att, att_add])
        return fea
    
    def get_EDVR_model(self):
        input_x = Input((self.nframes, self.H, self.W, self.C))
        x_center = self.__get_center_layer(input_x)
        x_reshaped = Lambda(lambda x: K.reshape(x, (-1, self.H, self.W, self.C)))(input_x)
        # L1
        if self.is_predeblur:
            L1_fea = self.__Predeblur_ResNet_Pyramid(x_reshaped)
            L1_fea = Conv2D(self.nf, 1)(L1_fea)
            if self.HR_in:
                self.H, self.W = self.H // 4, self.W // 4
        else:
            L1_fea = self.__conv1_block(x_reshaped)
            if self.HR_in:
                for i in range(2):
                    L1_fea = self.__conv2_block(L1_fea)
                self.H, self.W = self.H // 4, self.W // 4
        for _ in range(self.front_RBs):
            L1_fea = self.__ResidualBlock_noBN(L1_fea)
        # L2
        L2_fea = self.__conv2_block(L1_fea)
        L2_fea = self.__conv1_block(L2_fea)
        # L3
        L3_fea = self.__conv2_block(L2_fea)
        L3_fea = self.__conv1_block(L3_fea)
        L1_fea = Lambda(lambda x: K.reshape(x, (-1, self.nframes, self.H, self.W, self.C)))(L1_fea)
        L2_fea = Lambda(lambda x: K.reshape(x, (-1, self.nframes, self.H//2, self.W//2, self.C)))(L2_fea)
        L3_fea = Lambda(lambda x: K.reshape(x, (-1, self.nframes, self.H//4, self.W//4, self.C)))(L3_fea)

        #### pcd align
        # ref feature list
        ref_fea_l = [self.__get_center_layer(L1_fea), self.__get_center_layer(L2_fea), 
                     self.__get_center_layer(L3_fea)]

        aligned_fea = []
        for i in range(self.nframes):
            nbr_fea_l = [Lambda(lambda x: x[:, i, :, :, :])(L1_fea), 
                         Lambda(lambda x: x[:, i, :, :, :])(L2_fea),
                         Lambda(lambda x: x[:, i, :, :, :])(L3_fea)]
            aligned_fea.append(self.__PCD_Align(nbr_fea_l, ref_fea_l))

        aligned_fea = Lambda(lambda x: K.stack(x, axis=1))(aligned_fea)

        fea = self.__TSA_Fusion(aligned_fea)

        for _ in range(self.back_RBs):
            fea = self.__ResidualBlock_noBN(fea)

        out = Subpixel(self.nf, 3, 2, padding='same')(fea)
        out = LeakyReLU(alpha=.1)(out)
        out = Subpixel(64, 3, 2, padding='same')(out)
        out = LeakyReLU(alpha=.1)(out)
        out = Conv2D(64, 3, padding='same')(out) # HR conv
        out = LeakyReLU(alpha=.1)(out)
        out = Conv2D(3, 3, padding='same')(out) # Conv last
        if self.HR_in:
            base = x_center
        else:
            base = UpSampling2D(size=(4, 4), interpolation='bilinear')(x_center)
        out = add([out, base])
        return Model(input_x, out, name='EDVR')


def main():
    inp_shape = (256, 256, 3)
    nframes = 5
    VideoSuperResolution = EDVR(inp_shape=inp_shape, nframes=nframes)
    model = VideoSuperResolution.get_EDVR_model()
    print(model.summary())
    return model
    
if __name__ == "__main__":
    main()
