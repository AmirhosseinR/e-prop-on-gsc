import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow import keras

# Cell = keras.layers.SimpleRNNCell
Cell = tf.keras.layers.Layer
# rd = np.random.RandomState(3000)


# PSP on output layer
@tf.function
def exp_convolve(tensor, decay, init=None, axis=1):  # tensor shape (trial, time, neuron)
    
    assert tensor.dtype in [tf.float16, tf.float32]

    l_shp = len(tensor.get_shape())
    perm = np.arange(l_shp)
    perm[0] = axis
    perm[axis] = 0

    tensor_time_major = tf.transpose(a=tensor, perm=perm)
    if init is not None:
        assert str(init.get_shape()) == str(tensor_time_major[0].get_shape())  # must be batch x neurons
        initializer = init
    else:
        initializer = tf.zeros_like(tensor_time_major[0])

    filtered_tensor = tf.scan(lambda a, x: a * decay + x, tensor_time_major, initializer=initializer)
    filtered_tensor = tf.transpose(a=filtered_tensor, perm=perm)
    return filtered_tensor

surrograte_type = 'MG' # Multi-Gaussian surrogate gradient
gamma = 0.5
lens = 0.5
scale = 6.0
hight = .15

@tf.function
def gaussian(x, mu=0., sigma=.5):
    return tf.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / tf.sqrt(2 * np.pi) / sigma

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        if surrograte_type == 'MG':
            temp = gaussian(v_scaled, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(v_scaled, mu=lens, sigma=scale * lens) * hight \
                - gaussian(v_scaled, mu=-lens, sigma=scale * lens) * hight
            dE_dv_scaled = dE_dz * temp * gamma
        else:
            dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
            dz_dv_scaled *= dampening_factor
            dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad

@tf.function
def shift_by_one_time_step(tensor, initializer=None):
    '''
    Shift the input on the time dimension by one.
    :param tensor: a tensor of shape (trial, time, neuron)
    :param initializer: pre-prend this as the new first element on the time dimension
    :return: a shifted tensor of shape (trial, time, neuron)
    '''
    assert tensor.dtype in [tf.bfloat16, tf.float16, tf.float32]
    r_shp = range(len(tensor.get_shape()))
    transpose_perm = [1, 0] + list(r_shp)[2:]
    # tensor_time_major = tf.transpose(tensor, perm=transpose_perm)
    tensor_time_major = tensor
    

    if initializer is None:
        initializer = tf.zeros_like(tensor_time_major[0])

    shifted_tensor = tf.concat([initializer[None, :, :], tensor_time_major[:-1]], axis=0)

    # shifted_tensor = tf.transpose(shifted_tensor, perm=transpose_perm)
    return shifted_tensor


class CustomALIF2(Cell):
    def __init__(self, n_rec, tau=20., thr=.615, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=.16,
                 stop_gradients=False, w_in_init=None, w_rec_init=None, n_refractory=1,
                 wr_init_tofile=False):
        """
        CustomALIF2 provides the recurrent tensorflow cell model for implementing LSNNs in combination with
        eligibility propagation (e-prop).
        :param n_in: number of input neurons
        :param n_rec: number of output neurons
        :param tau: membrane time constant
        :param thr: spike threshold
        :param dt: length of discrete time steps
        :param dtype: data type of tensors
        :param dampening_factor: used in pseudo-derivative
        :param tau_adaptation: time constant of adaptive threshold decay
        :param beta: impact of adapting thresholds
        :param stop_gradients: stop gradients between next cell state and visible states
        :param w_in_init: initial weights for input connections
        :param w_rec_init: initial weights for recurrent connections
        :param n_refractory: number of refractory time steps
        """
        super().__init__()

        self.state_size = (n_rec, n_rec, n_rec)

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

        self.thr = thr
        self.n_refractory = n_refractory
        tauM_inital_std = 5.
        tauAdp_inital_std = 5.
        # self.tau_adaptation = tau_adaptation
        # init_tau_adaptation = (rd.normal(loc=tau_adaptation, scale=tauAdp_inital_std, size=(n_rec))).astype(np.float32)
        init_tau_adaptation = tf.ones(n_rec, dtype=dtype) * (tau_adaptation)
        # tau = tf.cast(tau, dtype=dtype)
        self.tau_adaptation = tf.Variable(initial_value=init_tau_adaptation, shape=(n_rec), dtype=tf.float32, trainable=False, name='tau_adp')
        
        self.beta = beta
        # self.rho = np.exp(-dt / tau_adaptation) #decay_b -> rho
        

        # if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        # if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)

        # tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self.dampening_factor = dampening_factor
        self.stop_gradients = stop_gradients
        self.dt = dt
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        # self.tau = tau
        # init_tau = (rd.normal(loc=tau, scale=tauM_inital_std, size=(n_rec))).astype(np.float32)
        init_tau = tf.ones(n_rec, dtype=dtype) * (tau)
        self.tau = tf.Variable(initial_value=init_tau, shape=(n_rec), dtype=tf.float32, trainable=False, name='tauM')
        
        # self.alpha = tf.exp(-dt / tau)
        
        self.b_j0_value = 1.6 #thr

        self.wr_init_tofile = wr_init_tofile

        self.w_in_init  = w_in_init
        self.w_rec_init = w_rec_init

        self.w_in  = None
        self.w_rec = None
        self.disconnect_mask = None

    def build(self, input_shape):
        n_in  = input_shape[-1]
        n_rec = self.n_rec
        dtype = self.data_type

        rand_init = tf.keras.initializers.RandomNormal

        if self.wr_init_tofile:
            # Input weights
            init_w_in_var = self.w_in_init if self.w_in_init is not None else (rd.randn(n_in, n_rec) / np.sqrt(n_in)).astype(np.float32)
            self.w_in     = self.add_weight(shape=(n_in,n_rec), initializer=rand_init(stddev=1. / np.sqrt(n_in)), name='input_weights', dtype=dtype)
            #----------------------------------------
            # save to file
            with open('../files/w_in.dat', 'w') as outfile:
                np.savetxt(outfile, self.w_in, fmt='%-8.6f')
            #----------------------------------------
            # define the recurrent weight variable
            self.disconnect_mask = tf.cast(np.diag(np.ones(n_rec, dtype=bool)), tf.bool)
            init_w_rec_var = self.w_rec_init if self.w_rec_init is not None else (rd.randn(n_rec, n_rec) / np.sqrt(n_rec)).astype(np.float32)
            self.w_rec     = self.add_weight(shape=(n_rec, n_rec), initializer= rand_init(stddev=1. / np.sqrt(n_rec)), name='recurrent_weights', dtype=dtype)
            #----------------------------------------
            # save to file
            with open('../files/w_rec.dat', 'w') as outfile:
                np.savetxt(outfile, self.w_rec, fmt='%-8.6f')
            #----------------------------------------
        else:
            w_in_file = np.loadtxt("../files/w_in.dat", delimiter=" ", dtype=dtype)
            w_in_file = w_in_file.reshape(n_in, n_rec)
            self.w_in = self.add_weight(shape=(n_in,n_rec), initializer=w_in_file, name='input_weights', dtype=dtype)
            #----------------------------------------
            w_rec_file = np.loadtxt("../files/w_rec.dat", delimiter=" ", dtype=np.float32)
            w_rec_file = w_rec_file.reshape(n_rec, n_rec)
            self.w_rec = self.add_weight(shape=(n_rec, n_rec), initializer= w_rec_file, name='recurrent_weights', dtype=dtype)
            #----------------------------------------

        self.variable_list = [self.w_in, self.w_rec]

        super().build(input_shape)

    def get_recurrent_weights(self):
      w_rec_var = tf.where(self.disconnect_mask, tf.zeros_like(self.w_rec), self.w_rec)
      return w_rec_var

    def compute_z(self, v, b):
        B = self.b_j0_value + b * self.beta # eq. (8), adaptive_thr -> A_j
        
        if surrograte_type == 'MG':
            v_scaled = (v - B) # v # (v - B) # / self.thr
        else:
            v_scaled = (v - B)  / self.thr
            
        z = SpikeFunction(v_scaled, self.dampening_factor)
        z = z * 1 / self.dt
        return z

    def __call__(self, inputs, states, constants=None):
        self.alpha = tf.exp(-self.dt / self.tau)
        alpha = self.alpha #alpha

        z = states[0]
        v = states[1]
        b = states[2]

        old_z = self.compute_z(v, b) # eq. (9)
        
        # This stop_gradient allows computing e-prop with auto-diff.
        #
        # needed for correct auto-diff computation of gradient for threshold adaptation
        # stop_gradient: forward pass unchanged, gradient is blocked in the backward pass
        if self.stop_gradients:
            z = tf.stop_gradient(z)       
        
        self.rho = tf.exp(-self.dt / self.tau_adaptation) #decay_b -> rho
        
        new_b = self.rho * b + (1-self.rho) * old_z  # eq. (10), decay_b -> rho, b -> a_j

        if len(self.w_in.get_shape().as_list()) == 3:
            i_in = tf.einsum('bi,bij->bj', inputs, self.w_in)
        else:
            i_in = tf.matmul(inputs, self.w_in)


        if len(self.w_rec.get_shape().as_list()) == 3:
            i_rec = tf.einsum('bi,bij->bj', z, self.w_rec)
        else:
            i_rec = tf.matmul(z, self.w_rec)
        i_t =  i_in +  i_rec


        B = self.b_j0_value + new_b * self.beta # eq. (8), adaptive_thr -> A_j
        
        I_reset = z * B * self.dt

        new_v = alpha * v + (1-alpha) * i_t - I_reset       

        # Spike generation
        v_scaled = (new_v - B) # new_v #(new_v - B)
        new_z = SpikeFunction(v_scaled, self.dampening_factor)

        new_state = (new_z, new_v, new_b)
        return [new_z, new_v, new_b], new_state
    
    @tf.function
    def compute_eligibility_traces(self, v_scaled, z_pre, z_post, is_rec):
    
        n_neurons = tf.shape(input=z_post)[2]
        rho = tf.exp(-self.dt / self.tau_adaptation) #self.decay_b
        beta = self.beta
        alpha = tf.exp(-self.dt / self.tau) #self._decay #tau_adaptation
        n_ref = self.n_refractory

        # everything should be time major
        z_pre = tf.transpose(a=z_pre, perm=[1, 0, 2])
        v_scaled = tf.transpose(a=v_scaled, perm=[1, 0, 2])
        z_post = tf.transpose(a=z_post, perm=[1, 0, 2])

        if surrograte_type == 'MG':
            temp = gaussian(v_scaled, mu=0., sigma=lens) * (1. + hight) \
                    - gaussian(v_scaled, mu=lens, sigma=scale * lens) * hight \
                    - gaussian(v_scaled, mu=-lens, sigma=scale * lens) * hight
            psi_no_ref = temp * gamma
        else:
            psi_no_ref = self.dampening_factor / self.b_j0_value * tf.maximum(0., 1. - tf.abs(v_scaled))
            

        # update_refractory = lambda refractory_count, z_post:\
        #     tf.where(z_post > 0,tf.ones_like(refractory_count) * (n_ref - 1),tf.maximum(0, refractory_count - 1))

        # refractory_count_init = tf.zeros_like(z_post[0], dtype=tf.int32)
        # refractory_count = tf.scan(update_refractory, z_post[:-1], initializer=refractory_count_init)
        # refractory_count = tf.concat([[refractory_count_init], refractory_count], axis=0)

        # is_refractory = refractory_count > 0
        # psi = tf.where(is_refractory, tf.zeros_like(psi_no_ref), psi_no_ref)
        psi = psi_no_ref

        # alpha2=alpha/(1-alpha)
        update_epsilon_v = lambda epsilon_v, z_pre: alpha[None, None, :] * epsilon_v + z_pre[:, :, None] #* (1-alpha)
        # epsilon_v_zero = tf.ones((1, 1, n_neurons)) * z_pre[0][:, :, None]
        # epsilon_v = tf.scan(update_epsilon_v, z_pre[1:], initializer=epsilon_v_zero, )
        # epsilon_v = tf.concat([[epsilon_v_zero], epsilon_v], axis=0)
        epsilon_v_zero = tf.zeros((1, 1, n_neurons))* z_pre[0][:, :, None]
        epsilon_v = tf.scan(update_epsilon_v, z_pre, initializer=epsilon_v_zero, )
        
        
        # epsilon_v = epsilon_v * (1-alpha)

        update_epsilon_a = lambda epsilon_a, elems:\
                (rho - (1-rho) * beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi'] * (1-rho) * (1-alpha) # eq.(24)
                # (rho - alpha * (1-rho) * beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi'] * (1-rho) * (1-alpha)  # eq.(24)

        epsilon_a_zero = tf.zeros_like(epsilon_v[0])

        epsilon_a = tf.scan(fn=update_epsilon_a,
                            elems={'psi': psi, 'epsi': epsilon_v, 'previous_epsi':shift_by_one_time_step(epsilon_v)}, initializer=epsilon_a_zero)

        # epsilon_a = tf.scan(fn=update_epsilon_a,
        #                     elems={'psi': psi[:-1], 'epsi': epsilon_v[:-1], 'previous_epsi':shift_by_one_time_step(epsilon_v[:-1])}, initializer=epsilon_a_zero)
        # epsilon_a = tf.concat([[epsilon_a_zero], epsilon_a], axis=0)

        e_trace = psi[:, :, None, :] * (epsilon_v * (1-alpha) - beta * epsilon_a)  # eq. (25)

        # everything should be time major
        e_trace = tf.transpose(a=e_trace, perm=[1, 0, 2, 3])
        epsilon_v = tf.transpose(a=epsilon_v, perm=[1, 0, 2, 3])
        epsilon_a = tf.transpose(a=epsilon_a, perm=[1, 0, 2, 3])
        psi = tf.transpose(a=psi, perm=[1, 0, 2])

        if is_rec:
            identity_diag = tf.eye(n_neurons)[None, None, :, :]
            e_trace -= identity_diag * e_trace
            epsilon_v -= identity_diag * epsilon_v
            epsilon_a -= identity_diag * epsilon_a

        return e_trace, epsilon_v, epsilon_a, psi

    @tf.function
    def compute_loss_gradient(self, learning_signal, z_pre, z_post, v_post, b_post,
                              decay_out=None,zero_on_diagonal=None):
        B = self.b_j0_value + self.beta * b_post # eq.(8): At_j = v_th + beta * at_j 
    
        if surrograte_type == 'MG':
            v_scaled = (v_post - B) # v_post #(v_post - B)
        else:
            v_scaled = (v_post - B)/self.b_j0_value

        
        e_trace, epsilon_v, epsilon_a, psi = self.compute_eligibility_traces(v_scaled, z_pre, z_post, zero_on_diagonal)
        # learning_signal = learning_signal + alpha[None, None, :] * learning_signal
        if decay_out is not None:
            e_trace_time_major = tf.transpose(a=e_trace, perm=[1, 0, 2, 3])
            filtered_e_zero = tf.zeros_like(e_trace_time_major[0])
            filtering = lambda filtered_e, e: filtered_e * decay_out + e #* (1 - decay_out) # eq. (26)
            filtered_e = tf.scan(filtering, e_trace_time_major, initializer=filtered_e_zero)
            filtered_e = tf.transpose(a=filtered_e, perm=[1, 0, 2, 3])
            e_trace = filtered_e

        e_trace = tf.reduce_sum(e_trace, axis=(1))
        gradient = tf.einsum('bj,bij->ij', learning_signal[:,:], e_trace)     
        
        # gradient = tf.matmul(learning_signal, e_trace)
        return gradient, e_trace, epsilon_v, epsilon_a, psi
    