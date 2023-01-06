import tensorflow as tf
import numpy as np
import numpy.random as rd
import json
import os

def einsum_bij_jk_to_bik(a,b):
    try:
        n_b = int(a.get_shape()[0])
    except:
        n_b = tf.shape(input=a)[0]

    try:
        n_i = int(a.get_shape()[1])
    except:
        n_i = tf.shape(input=a)[1]

    try:
        n_j = int(a.get_shape()[2])
    except:
        n_j = tf.shape(input=a)[2]

    try:
        n_k = int(b.get_shape()[1])
    except:
        n_k = tf.shape(input=b)[1]

    a_ = tf.reshape(a,(n_b * n_i,n_j))
    a_b = tf.matmul(a_,b)
    ab = tf.reshape(a_b,(n_b,n_i,n_k))
    return ab

class GSCDataset:
    def __init__(self, n_mini_batch=1, n_in=40, seq_len=101, data_path='../files', shuffle=True, Train = True):
        self.n_mini_batch = n_mini_batch
        self.data_path = data_path
        self.n_in = n_in
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.Train = Train
        # Load features from files
        if self.Train:
            self.feature_stack_train,   self.classes_stack_train   = self.load_data_stack('train')
        self.feature_stack_test,    self.classes_stack_test    = self.load_data_stack('test')
        self.feature_stack_develop, self.classes_stack_develop = self.load_data_stack('dev')
        
        if self.Train:
            self.n_train   = len(self.feature_stack_train)
        self.n_test    = len(self.feature_stack_test)
        self.n_develop = len(self.feature_stack_develop)
        
        if self.Train:
            print('Dataset sizes: test {} \t train {} \t validation {}'.format(self.n_test,self.n_train,self.n_develop))
        
            self.mini_batch_indices = self.generate_mini_batch_selection_list()
            self.current_epoch = 0
            self.index_current_minibatch = 0
        
    def generate_mini_batch_selection_list(self):
        if self.shuffle:
            perm = rd.permutation(self.n_train) #Commented for debuging purpose
        else:
            perm = range(self.n_train)
            
        number_of_batches = self.n_train // self.n_mini_batch
        return np.array_split(perm, number_of_batches) #,validation_set
    
    def load_data_stack(self, dataset):
        path = os.path.join(self.data_path, dataset)
        
        feature_path = os.path.join(path, 'features.dat')
        classes_path = os.path.join(path, 'target_outputs.dat')
        
        feature_stack = np.loadtxt(feature_path, delimiter=" ", dtype=np.float32)
        classes_stack = np.loadtxt(classes_path, delimiter=" ", dtype=np.float32)
        
        return feature_stack, classes_stack
    
    def load_features(self, dataset,selection):
        if dataset == 'train':
            features = self.feature_stack_train[selection]
            classes  = self.classes_stack_train[selection]
        elif dataset == 'test':
            features = self.feature_stack_test[selection]
            classes  = self.classes_stack_test[selection]
        elif dataset == 'develop':
            features = self.feature_stack_develop[selection]
            classes  = self.classes_stack_develop[selection]

        return features, classes
    
    def get_next_training_batch(self):
        features, classes = self.load_features('train', selection=self.mini_batch_indices[self.index_current_minibatch])
        
        # assert (len(features[0]) == self.n_mini_batch * self.seq_len * self.n_in)
        
        # features = features.reshape(self.n_mini_batch, self.seq_len, self.n_in)
        features = features.reshape(features.shape[0], self.seq_len, self.n_in)
        classes_all_timestep = np.ones((1, self.seq_len))
        classes  = classes.reshape(features.shape[0],1)
        # classes = classes * classes_all_timestep

        self.index_current_minibatch += 1
        if self.index_current_minibatch >= len(self.mini_batch_indices):
            self.index_current_minibatch = 0
            self.current_epoch += 1

            #Shuffle the training set after each epoch
            number_of_batches = len(self.mini_batch_indices)
            training_set_indices = np.concatenate(self.mini_batch_indices)
            if self.shuffle:
                training_set_indices = rd.permutation(training_set_indices) #Commented for debuging purpose
            else:
                training_set_indices = training_set_indices
            self.mini_batch_indices = np.array_split(training_set_indices, number_of_batches)

        return features, classes
    
    def get_test_batch(self):
        features, classes = self.load_features('test', np.arange(self.n_test, dtype=np.int))
        features = features.reshape(features.shape[0], self.seq_len, self.n_in)
        classes  = classes.reshape(features.shape[0],1)
        return features, classes

    def get_next_test_batch(self, selection):
        features, classes =  self.load_features('test', selection=selection)
        features = features.reshape(features.shape[0], self.seq_len, self.n_in)
        classes  = classes.reshape(features.shape[0],1)
        return features, classes
    
    def get_validation_batch(self):
        features, classes =  self.load_features('develop', np.arange(self.n_develop, dtype=np.int))
        features = features.reshape(features.shape[0], self.seq_len, self.n_in)
        classes  = classes.reshape(features.shape[0],1)
        return features, classes

    def get_next_validation_batch(self, selection):
        features, classes =  self.load_features('develop', selection=selection)
        features = features.reshape(features.shape[0], self.seq_len, self.n_in)
        classes  = classes.reshape(features.shape[0],1)
        return features, classes
        
    
    
class NumpyAwareEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyAwareEncoder, self).default(obj)



def raster_plot(ax,spikes,linewidth=0.8,**kwargs):

    n_t,n_n = spikes.shape
    event_times,event_ids = np.where(spikes)
    max_spike = 10000
    event_times = event_times[:max_spike]
    event_ids = event_ids[:max_spike]

    for n,t in zip(event_ids,event_times):
        ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)

    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([0, n_t])
    ax.set_yticks([0, n_n])

def strip_right_top_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()



def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes



def generate_click_task_data(batch_size, seq_len, n_neuron, recall_duration, p_group, f0=0.5,
                             n_cues=7, t_cue=100, t_interval=150,
                             n_input_symbols=4):
    t_seq = seq_len
    n_channel = n_neuron // n_input_symbols

    # randomly assign group A and B
    prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
    idx = rd.choice([0, 1], batch_size)
    probs = np.zeros((batch_size, 2), dtype=np.float32)
    # assign input spike probabilities
    probs[:, 0] = prob_choices[idx]
    probs[:, 1] = prob_choices[1 - idx]

    cue_assignments = np.zeros((batch_size, n_cues), dtype=np.int)
    # for each example in batch, draw which cues are going to be active (left or right)
    for b in range(batch_size):
        cue_assignments[b, :] = rd.choice([0, 1], n_cues, p=probs[b])

    # generate input nums - 0: left, 1: right, 2:recall, 3:background noise
    input_nums = 3*np.ones((batch_size, seq_len), dtype=np.int)
    input_nums[:, :n_cues] = cue_assignments
    input_nums[:, -1] = 2

    # generate input spikes
    input_spike_prob = np.zeros((batch_size, t_seq, n_neuron))
    d_silence = t_interval - t_cue
    for b in range(batch_size):
        for k in range(n_cues):
            # input channels only fire when they are selected (left or right)
            c = cue_assignments[b, k]
            # reverse order of cues
            #idx = sequence_length - int(recall_cue) - k - 1
            idx = k
            input_spike_prob[b, d_silence+idx*t_interval:d_silence+idx*t_interval+t_cue, c*n_channel:(c+1)*n_channel] = f0

    # recall cue
    input_spike_prob[:, -recall_duration:, 2*n_channel:3*n_channel] = f0
    # background noise
    input_spike_prob[:, :, 3*n_channel:] = f0/4.
    input_spikes = generate_poisson_noise_np(input_spike_prob)

    # generate targets
    target_mask = np.zeros((batch_size, seq_len), dtype=np.bool)
    target_mask[:, -1] = True
    target_nums = np.zeros((batch_size, seq_len), dtype=np.int)
    target_nums[:, :] = np.transpose(np.tile(np.sum(cue_assignments, axis=1) > int(n_cues/2), (seq_len, 1)))

    return input_spikes, input_nums, target_nums, target_mask

