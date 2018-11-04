import tensorflow as tf
import numpy as np
from pprint import pprint
# from param_collection import ParamCollection



class InitialColumnProgNN(object):

    def __init__(
        self
#         , topology, activations
#         , layers_func
        , n_input, n_output, session, dtype_X, dtype_y
    ):
#         n_input = topology[0] # TODO: after modified into function
        # Layers in network.
#         L = len(topology) - 1
        self.session = session
        self.dtype_X = dtype_X
        self.dtype_y = dtype_y
#         self.L = L # n_layers except input layer
#         self.topology = topology
#         self.layers_func = layers_func # TODO: want to modify to FUNCTION
        self.Xs = tf.placeholder(dtype_X,shape=[None, n_input]) # output of input layer
        self.ys = tf.placeholder(dtype_y,shape=[None, n_output])

        # below are Tensor
        self.W = [] # weights in each layer
        self.b =[] # biases in each layer
        self.h = [self.Xs] # activation output in each layer
        self.params = [] # store all Ws and bs, will be modified when W and b is trained i think
        self.logits = None
        self.prediction = None
        self.loss = None # output loss
        # above are Tensor
        
        self.train_op = None # opt.minimize(self.loss)
        
        self.loss_hist_train = []
        self.loss_hist_val = []
        
    def weight_fc(self, shape, stddev=0.1, initial=None):
        if initial is None:
            initial = tf.truncated_normal(shape,stddev=stddev,dtype=self.dtype_X)
        return initial

    def bias_fc(self, shape, init_bias=0.1, initial=None):
        if initial is None:
            initial = tf.constant(init_bias,shape=shape,dtype=self.dtype_X)
            initial = tf.Variable(initial)
        return initial
    def add_fc(
        self
#         , inputs
#         , in_size
        , out_size, activation_func=None, output_layer=False):
#         Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#         biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        inputs = self.h[-1] # last layer output as input
        in_size = inputs.get_shape().as_list()[1]#self.session.run(tf.shape(inputs))[1] # TODO: get input shape = [1,out_size]
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
        self.train_op = opt.minimize(self.loss)
        
    def train(self, X, y, n_epochs, batch_size=None, data_valid=None, display_steps=50): 
        # data_valid:tuple
        self.session.run(tf.global_variables_initializer())
        
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples
        steps_per_epoch = int(n_samples/batch_size)
        counter = 0
        for epoch in range(1,n_epochs+1):
#             print('Epoch',epoch,'start.')
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
                    print('Epoch',epoch,', step',step,', train loss =',loss_train)
                counter += 1
                
    
    def predict(X):
        return self.session.run(self.prediction,feed_dict={self.Xs:X})




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
    X_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, X_data.shape).astype(np.float32)
    y_data = np.square(X_data) - 0.5 + noise
    print('X_data',X_data.shape,'\n',X_data[:5])
    print('y_data',y_data.shape,'\n',y_data[:5])
    
    n_input = X_data.shape[1]
    col_0 = InitialColumnProgNN(
        n_input=n_input
        , n_output=1
        , session=session
        , dtype_X=tf.float32, dtype_y=tf.float32
    )
    col_0.add_fc(10,activation_func=tf.nn.relu)
#     col_0.add_fc(1024,activation_func=tf.nn.relu)
    col_0.add_fc(1,activation_func=None,output_layer=True)
    col_0.compile_nn(
#         loss_func=tf.losses.mean_squared_error
        loss=tf.reduce_mean(tf.reduce_sum(tf.square(col_0.ys - col_0.prediction),reduction_indices=[1]))
#         loss=tf.losses.mean_squared_error(col_0.ys,col_0.prediction)
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










