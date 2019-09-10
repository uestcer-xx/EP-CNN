from keras.engine.topology import Layer
import keras.backend as K


class ElasticPooling(Layer):
    def __init__(self,p1=128,wordvec=16, **kwargs):   
        
        self.dim_ordering = K.image_dim_ordering()
        self.p1=p1
        self.wordvec=wordvec

        super(ElasticPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return(input_shape[0],self.p1,self.wordvec,self.nb_channels)

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        num_rows = K.cast(input_shape[1]/self.p1,'int32')  #rows

        out=[]
        for i in range(self.p1):
            for j in range(self.wordvec):
                p_area=x[:,i*num_rows:(i+1)*num_rows,j*2:(j+1)*2,:]
                pooled_val = K.max(p_area, axis=(1, 2))
                out.append(pooled_val)
        out=K.concatenate(out)
        outs=K.reshape(out,(input_shape[0],self.p1,self.wordvec,self.nb_channels))

        return outs
