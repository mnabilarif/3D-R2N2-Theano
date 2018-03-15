import numpy as np

# Theano
import theano
import theano.tensor as tensor

from models.net import Net, tensor5
from lib.layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    FCConv3DLayer, TanhLayer, SigmoidLayer, ComplementLayer, AddLayer, \
    EltwiseMultiplyLayer, get_trainable_params


class LSTMNet(Net):

    def network_definition(self):

        # (multi_views, self.batch_size, 3, self.img_h, self.img_w),
        self.x = tensor5()
        self.is_x_tensor4 = False

        img_w = self.img_w
        img_h = self.img_h
        n_gru_vox = 4
        # n_vox = self.n_vox

        
        n_convfilter = [96, 128, 256, 256, 256, 256]
        n_fc_filters = [1024]
        n_deconvfilter = [128, 128, 128, 64, 32, 2]
        input_shape = (self.batch_size, 3, img_w, img_h)

        
        # To define weights, define the network structure first
        x = InputLayer(input_shape)
        conv1 = ConvLayer(x, (n_convfilter[0], 7, 7))
        pool1 = PoolLayer(conv1)
        
        conv2 = ConvLayer(pool1, (n_convfilter[1], 3, 3))
        pool2 = PoolLayer(conv2)
        
        conv3 = ConvLayer(pool2, (n_convfilter[2], 3, 3))
        pool3 = PoolLayer(conv3)
        
        conv4 = ConvLayer(pool3, (n_convfilter[3], 3, 3))
        pool4 = PoolLayer(conv4)
        
        conv5 = ConvLayer(pool4, (n_convfilter[4], 3, 3))
        pool5 = PoolLayer(conv5)
        
        conv6 = ConvLayer(pool5, (n_convfilter[5], 3, 3))
        pool6 = PoolLayer(conv6)
        
        flat6 = FlattenLayer(pool6)
        
        fc7 = TensorProductLayer(flat6, n_fc_filters[0])

        #LSTM
        # Set the size to be 256x4x4x4
        h_shape = (self.batch_size, n_gru_vox, n_deconvfilter[0], n_gru_vox, n_gru_vox)

        # Dummy 3D grid hidden representations
        prev_h = InputLayer(h_shape)
        
        t_x_s_forget = FCConv3DLayer(prev_h, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))
        t_x_s_input = FCConv3DLayer(prev_h, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))
        t_x_s_cell = FCConv3DLayer(prev_h, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3))
        
        #initialize hidden state and cell state with 0
        if self.hidden_last is None:
            self.hidden_last = theano.shared(np.zeros(h_shape, dtype=theano.config.floatX))
        if self.cell_last is None:
            self.cell_last = theano.shared(np.zeros(h_shape, dtype=theano.config.floatX))
        
        def recurrence(x_curr, prev_h_tensor, prev_s_tensor):
            #prev_h_tensor: previous hidden state output tensor
            #prev_s_tensor:previous cell state output tensor
            
            # Scan function cannot use compiled function.
            input_ = InputLayer(input_shape, x_curr)
            conv1_ = ConvLayer(input_, (n_convfilter[0], 7, 7), params=conv1.params)
            pool1_ = PoolLayer(conv1_)
            rect1_ = LeakyReLU(pool1_)
            conv2_ = ConvLayer(rect1_, (n_convfilter[1], 3, 3), params=conv2.params)
            pool2_ = PoolLayer(conv2_)
            rect2_ = LeakyReLU(pool2_)
            conv3_ = ConvLayer(rect2_, (n_convfilter[2], 3, 3), params=conv3.params)
            pool3_ = PoolLayer(conv3_)
            rect3_ = LeakyReLU(pool3_)
            conv4_ = ConvLayer(rect3_, (n_convfilter[3], 3, 3), params=conv4.params)
            pool4_ = PoolLayer(conv4_)
            rect4_ = LeakyReLU(pool4_)
            conv5_ = ConvLayer(rect4_, (n_convfilter[4], 3, 3), params=conv5.params)
            pool5_ = PoolLayer(conv5_)
            rect5_ = LeakyReLU(pool5_)
            conv6_ = ConvLayer(rect5_, (n_convfilter[5], 3, 3), params=conv6.params)
            pool6_ = PoolLayer(conv6_)
            rect6_ = LeakyReLU(pool6_)
            flat6_ = FlattenLayer(rect6_)
            fc7_ = TensorProductLayer(flat6_, n_fc_filters[0], params=fc7.params)
            rect7_ = LeakyReLU(fc7_)
            
            #LSTM
            # Dummy 3D grid hidden representations for previous hidden state and cell state
            prev_h_ = InputLayer(h_shape, prev_h_tensor)
            prev_s_ = InputLayer(h_shape, prev_s_tensor)
            
            t_x_s_forget_ = FCConv3DLayer(
                    prev_h_, 
                    rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                    params=t_x_s_forget.params)
            
            t_x_s_input_ = FCConv3DLayer(
                    prev_h_, 
                    rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                    params=t_x_s_input.params)
            
            t_x_s_cell_ = FCConv3DLayer(
                    prev_h_, 
                    rect7_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                    params=t_x_s_cell.params)
            
            forget_gate_ = SigmoidLayer(t_x_s_forget_)
            input_gate_ = SigmoidLayer(t_x_s_input_)
            tanh_t_x_s_cell_ = TanhLayer(t_x_s_cell_)
            #current cell state
            cell_state_ = AddLayer(
                    EltwiseMultiplyLayer(forget_gate_, prev_s_),
                    EltwiseMultiplyLayer(input_gate_, tanh_t_x_s_cell_))
            #current hidden state, i.e. the output of lstm 
            hidden_state_ = TanhLayer(cell_state_)
            
            return hidden_state_.output, cell_state_.output

        s_update, _ = theano.scan(recurrence,
            sequences=[self.x],  # along with images, feed in the index of the current frame
            outputs_info=[self.hidden_last.get_value(), self.cell_last.get_value()])
        #s_update means updates of hidden states and cell states
        cell_all = s_update[-1]
        h_all = s_update[0]
        h_last = h_all[-1]
        
        lstm_s = InputLayer(h_shape, h_last)

        unpool7 = Unpool3DLayer(lstm_s)
        conv7 = Conv3DLayer(unpool7, (n_deconvfilter[1], 3, 3, 3))
        rect7 = LeakyReLU(conv7)
        
        unpool8 = Unpool3DLayer(rect7)
        conv8 = Conv3DLayer(unpool8, (n_deconvfilter[2], 3, 3, 3))
        rect8 = LeakyReLU(conv8)
        
        unpool9 = Unpool3DLayer(rect8)
        conv9 = Conv3DLayer(unpool9, (n_deconvfilter[3], 3, 3, 3))
        rect9 = LeakyReLU(conv9)
        
        # unpool10 = Unpool3DLayer(rect9)
        conv10 = Conv3DLayer(rect9, (n_deconvfilter[4], 3, 3, 3))
        rect10 = LeakyReLU(conv10)
        
        conv11 = Conv3DLayer(rect10, (n_deconvfilter[5], 3, 3, 3))
        
        softmax_loss = SoftmaxWithLoss3D(conv11.output)
        self.loss = softmax_loss.loss(self.y)
        self.error = softmax_loss.error(self.y)
        self.params = get_trainable_params()
        self.output = softmax_loss.prediction()
        #activation of all cell states
        self.activations = [cell_all]
        self.new_hidden_last = h_last
        self.new_cell_last = cell_all[-1]
if __name__ == '__main__':
    from lib.solver import Solver
    from lib.config import cfg
    import time
    start_time = time.time()
    theano.config.floatX = 'float32'
    net = LSTMNet()
    
    solver = Solver(net)
    
    x = np.zeros((5, 36, 3, 127, 127)).astype(theano.config.floatX)
    y = np.ones((36, 32, 2, 32, 32)).astype(theano.config.floatX)
    
    loss = solver.train_loss(x, y)
    print("it takes %s seconds" % (time.time()-start_time))