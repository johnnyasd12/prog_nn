import tensorflow as tf
import numpy as np
from pprint import pprint
from param_collection import ParamCollection
from tensorflow.examples.tutorials.mnist import input_data

class InitialColumnProgNN(object):

    def __init__(
        self
#         , topology, activations
#         , layers_func
        , input_dims, output_dims, session, dtype_X, dtype_y
    ):
#         input_dims = topology[0] # TODO: after modified into function
        # Layers in network.
#         L = len(topology) - 1
        
        #         self.L = L # n_layers except input layer
#         self.topology = topology
#         self.layers_func = layers_func # TODO: want to modify to FUNCTION
        self.L = 0 # n_layers except input layer
        self.topology = [input_dims] # output dims of each layer
        self.activations = []
        # tensorflow
        self.session = session
        self.dtype_X = dtype_X
        self.dtype_y = dtype_y
        self.Xs = tf.placeholder(dtype_X,shape=[None, input_dims]) # output of input layer
        self.ys = tf.placeholder(dtype_y,shape=[None, output_dims])
        # below are Tensor
        self.W = [] # weights in each layer
        self.b =[] # biases in each layer
        self.h = [self.Xs] # activation output in each layer
        self.params = [] # store all Ws and bs, will be modified when W and b is trained i think
        self.logits = None # neurons before input to final activation
        self.prediction = None
        self.loss = None # output loss
        # above are Tensor
        self.opt = None
        self.train_op = None # opt.minimize(self.loss)
        # loss history
        self.loss_hist_train = []
        self.loss_hist_val = []
        # param collection
        self.pc = None
        
    def weight_fc(self, shape, stddev=0.1, initial=None):
        if initial is None:
            initial = tf.truncated_normal(shape,stddev=stddev,dtype=self.dtype_X)
            initial = tf.Variable(initial)
            # initial = tf.Variable(tf.random_normal(shape), dtype=self.dtype_X)
        return initial

    def bias_fc(self, shape, init_bias=0.1, initial=None):
        if initial is None:
            initial = tf.constant(init_bias,shape=shape,dtype=self.dtype_X)
            initial = tf.Variable(initial)
            # initial = tf.Variable(tf.zeros(shape) + 0.1, dtype=self.dtype_X)
        return initial
    
    def add_fc(
        self
#         , inputs
#         , in_size
        , out_size, activation_func=None, output_layer=False, initial=None):
        
#         Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#         biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        inputs = self.h[-1] # last layer output as input
        shape_inputs = inputs.get_shape().as_list()
        print('FC_layer, shape_inputs =',shape_inputs)
        in_size = shape_inputs[1]#self.session.run(tf.shape(inputs))[1] # TODO: get input shape = [1,out_size]
        shape_W = [in_size,out_size]
        shape_b = [1,out_size]
        print('FC_layer, shape_W =',shape_W,', shape_b =',shape_b)
        
        Weights = self.weight_fc(shape_W)
        biases = self.bias_fc(shape_b)
        WX_b = tf.matmul(inputs, Weights) + biases
        if activation_func is None:
            out = WX_b
        else:
            out = activation_func(WX_b)

        self.topology.append(out_size)
        self.activations.append(activation_func)
        self.L = self.L + 1
        self.W.append(Weights)
        self.b.append(biases)
        self.h.append(out)
        if output_layer:
            self.logits = WX_b
            self.prediction = self.h[-1]
        
        # for ParamCollection
        self.params.append(self.W[-1])
        self.params.append(self.b[-1])
#         self.pc = ParamCollection(self.session, params) # TODO: watch this

    def compile_nn(self, loss, opt, metrics=None):
        # metrics:list
        
        # loss_func:
        # tf.nn.softmax_cross_entropy_with_logits, 
        # tf.losses.mean_squared_error
        if self.logits is None:
            print("Error: no output layer set.")
#         self.loss = loss_func(self.ys, self.logits) # TODO: if ReLU + MSE then not proper
        self.loss = loss
        self.opt = opt
        self.train_op = self.opt.minimize(self.loss)
        self.pc = ParamCollection(self.session, self.params)
        
    def train(self, X, y, n_epochs, batch_size=None, data_valid=None, display_steps=50): 
        # data_valid:list

        self.session.run(tf.global_variables_initializer())
        
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples
        steps_per_epoch = int(n_samples/batch_size)
        counter = 0
        for epoch in range(1,n_epochs+1):
            for step in range(0,steps_per_epoch): # n_sample=1000, batch_size=10, steps_per_epoch=100
                if step != steps_per_epoch-1: # last step
                    X_batch = X[step*batch_size:(step+1)*batch_size]
                    y_batch = y[step*batch_size:(step+1)*batch_size]
                else:
                    X_batch = X[step*batch_size:]
                    y_batch = y[step*batch_size:]
                
                self.session.run(
                    self.train_op
                    , feed_dict={self.Xs:X_batch, self.ys:y_batch}
                )
                if counter%display_steps==0 or (epoch==n_epochs and step==steps_per_epoch-1):
                    loss_train = self.session.run(self.loss,feed_dict={self.Xs:X_batch, self.ys:y_batch})
                    self.loss_hist_train.append(loss_train)
                    print('Epoch',epoch,', step',step,', loss =',loss_train, end=' ')
                    if data_valid is not None:
                        X_val = data_valid[0]
                        y_val = data_valid[1]
                        loss_val = self.session.run(self.loss,feed_dict={self.Xs:X_val,self.ys:y_val})
                        self.loss_hist_val.append(loss_val)
                        print(', val_loss =',loss_val)
                    else:
                        print()
                counter += 1
                
    
    def predict(X):
        return self.session.run(self.prediction,feed_dict={self.Xs:X})


def check_obj(obj_str):
    obj = eval(obj_str)
    obj_type = type(obj)
    print(obj_str
        , obj_type
        , end = ' '
        )
    if obj_type == np.ndarray:
        print(obj.shape)
    else:
        try:
            iterator = iter(obj)
        except TypeError:
            # not iterable
            print(obj)
        else:
            # iterable
            print(len(obj))

if __name__ == "__main__":
    mem_fraction = 0.25
    gpu_options = tf.GPUOptions(
        allow_growth=True
        ,per_process_gpu_memory_fraction=mem_fraction
        )
    config = tf.ConfigProto(gpu_options=gpu_options)
    session = tf.Session(config = config)
#     session.run(tf.global_variables_initializer())

#     X_data = np.random.random((6000))[:, np.newaxis]*100
#     noise = np.random.normal(0, 0.05, X_data.shape).astype(np.float32)*0
#     y_data = X_data*2 + 1 + noise
    try_reg = False
    if try_reg:
        X_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
        noise = np.random.normal(0, 0.05, X_data.shape).astype(np.float32)
        y_data = np.square(X_data) - 0.5 + noise
        print('X_data',X_data.shape,'\n',X_data[:5])
        print('y_data',y_data.shape,'\n',y_data[:5])
        
        input_dims = X_data.shape[1]
        col_0 = InitialColumnProgNN(
            input_dims=input_dims
            , output_dims=1
            , session=session
            , dtype_X=tf.float32, dtype_y=tf.float32
        )
        col_0.add_fc(10,activation_func=tf.nn.relu)
    #     col_0.add_fc(1024,activation_func=tf.nn.relu)
        col_0.add_fc(1,activation_func=None,output_layer=True)
        col_0.compile_nn(
    #         loss=tf.reduce_mean(tf.reduce_sum(tf.square(col_0.ys - col_0.prediction),reduction_indices=[1]))
            loss=tf.losses.mean_squared_error(col_0.ys,col_0.prediction)
    #         ,opt=tf.train.AdamOptimizer(learning_rate=1e-3)
            ,opt=tf.train.GradientDescentOptimizer(learning_rate=1e-1)
    #         ,mectrics=[]
        )
        col_0.train(
            X=X_data
            , y=y_data
            , batch_size=None
            , n_epochs=1000
            , display_steps=50
        )

    try_cls = True
    if try_cls:
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        X_train = mnist.train.images#.astype(np.float64)
        y_train = mnist.train.labels#.astype(np.float64)
        X_val = mnist.validation.images
        y_val = mnist.validation.labels
        X_test = mnist.test.images
        y_test = mnist.test.labels
        check_obj('X_train')
        check_obj('y_train')
        input_dims = X_train.shape[1]
        output_dims = y_train.shape[1]
        check_obj('input_dims')
        check_obj('output_dims')
        col_cls_0 = InitialColumnProgNN(
            input_dims=input_dims
            , output_dims=output_dims
            , session=session
            , dtype_X=tf.float64
            , dtype_y=tf.float64)
        col_cls_0.add_fc(512,activation_func=tf.nn.relu)
        col_cls_0.add_fc(256,activation_func=tf.nn.relu)
        col_cls_0.add_fc(output_dims,activation_func=tf.nn.softmax,output_layer=True)
        col_cls_0.compile_nn(
            loss=tf.losses.softmax_cross_entropy(col_cls_0.ys,col_cls_0.logits)
            , opt=tf.train.AdamOptimizer(learning_rate=1e-3))
        col_cls_0.train(
            X=X_train
            ,y=y_train
            ,data_valid=[X_val,y_val]
            ,batch_size=128
            ,n_epochs=20
            ,display_steps=200)











