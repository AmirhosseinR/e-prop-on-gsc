import numpy as np
import numpy.random
import tensorflow as tf
from time import time

from tools import GSCDataset
from alif_eligibility_propagation import CustomALIF2, exp_convolve

from absl import app
from absl import flags
from absl import logging
import sys

FLAGS = flags.FLAGS
##
# flags.DEFINE_string('dataset', '../files_2CH_100TS', 'Path to dataset to use')
flags.DEFINE_string('dataset', '../files', 'Path to dataset to use') # Also change files/w_'out/in/rec'.dat
flags.DEFINE_bool('shuffle', False, 'Shuffle the training set after each epoch')
flags.DEFINE_bool('wr_init_tofile', True, 'If True, write init values of variables to files, esle read from files')
flags.DEFINE_bool('do_plot', False, 'interactive plots during training')
flags.DEFINE_bool('print_shape', False, 'interactive plots during training')

flags.DEFINE_integer('n_batch', 10, 'batch size of the testing set')
flags.DEFINE_integer('n_out', 12, 'number of output neurons (number of target curves)')
flags.DEFINE_integer('n_in', 40, 'number of input units')
flags.DEFINE_integer('n_ch', 2, 'number of input channel')
flags.DEFINE_integer('n_regular', 0, 'number of regular spiking units in the recurrent layer.')
flags.DEFINE_integer('n_adaptive', 120, 'number of adaptive spiking units in the recurrent layer')
flags.DEFINE_integer('n_ref', 0, 'Number of refractory steps')

flags.DEFINE_integer('n_iter', 30, 'number of iterations')
flags.DEFINE_integer('seq_len', 100, 'number of time steps per sequence')
flags.DEFINE_integer('print_every', 1, 'print statistics every K iterations')

flags.DEFINE_float('dampening_factor', 0.5, 'dampening factor to stabilize learning in RNNs')
flags.DEFINE_float('dt', 1., '(ms) simulation step')
flags.DEFINE_float('thr', 0.01, 'threshold at which the LSNN neurons spike (in arbitrary units)')
flags.DEFINE_float('tau_a', 150, 'Adaptation time constant')
flags.DEFINE_float('tau_v', 5, 'Membrane time constant of recurrent neurons')
flags.DEFINE_float('tau_out', 10, 'Mikolov: tau for PSP decay at output')
flags.DEFINE_float('beta', 0.184, 'Scaling constant of the adaptive threshold')

flags.DEFINE_bool('readout_bias', False, 'Use bias variable in readout')
flags.DEFINE_bool('eprop', True, 'Use e-prop to train network (BPTT if False)') # True
flags.DEFINE_string('eprop_impl', 'hardcoded', '["autodiff", "hardcoded"] Use tensorflow for computing e-prop '
                                                     'updates or implement equations directly')

flags.DEFINE_float('adam_epsilon', 1e-5, '')
flags.DEFINE_float('l2', 0.05e-5, 'l2 regularization')
flags.DEFINE_float('lr_init', 0.01, '') #0.01
flags.DEFINE_float('lr_decay', .3, '')
flags.DEFINE_integer('lr_decay_every', 6, 'Decay every')
##
FLAGS(sys.argv)

opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr_init, epsilon=FLAGS.adam_epsilon) # , beta1=momentum 

# For use in C file
rho_c = np.exp(-1/FLAGS.tau_a)
alpha_c = np.exp(-1/FLAGS.tau_v)
print('rho: ', rho_c)
print('alpha: ', alpha_c)
print('1-rho: ', 1-rho_c)
print('1-alpha: ', 1-alpha_c)
print('(1-alpha)x(1-rho): ', (1-alpha_c)*(1-rho_c))
#**********************************************
lr = tf.Variable(FLAGS.lr_init, dtype=tf.float32, trainable=False, name="LearningRate")
#**********************************************
# Experiment parameters
dt = FLAGS.dt  # time step in ms
taua = FLAGS.tau_a
tauv = FLAGS.tau_v
#############################################
#          Build the Network                #
#############################################
def get_cell():
    thr_new  = FLAGS.thr
    beta_new = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])

    return CustomALIF2(n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tauv,
                      dt=FLAGS.dt, tau_adaptation=taua, beta=beta_new, thr=thr_new,
                      dampening_factor=FLAGS.dampening_factor,
                      n_refractory=FLAGS.n_ref,
                      stop_gradients=FLAGS.eprop is not None,
                      wr_init_tofile=FLAGS.wr_init_tofile)    

cell_f = get_cell()
cell_f.build([FLAGS.n_batch, None, FLAGS.n_in*FLAGS.n_ch])
lsnn = tf.keras.layers.RNN(cell_f,return_sequences=True, dtype=tf.float32)
#############################################
#     Build the readout weights             #
#############################################
n_neurons = (FLAGS.n_regular + FLAGS.n_adaptive)
decay = np.exp(-FLAGS.dt / FLAGS.tau_out, dtype=np.float32)  # output layer filtered_z decay, chose value between 15 and 30ms as for tau_v

# After processing the data, this object loads it and prepare it.
print("Reading Dataset ...")
dataset = GSCDataset(n_mini_batch=FLAGS.n_batch, n_in=FLAGS.n_in*FLAGS.n_ch, seq_len=FLAGS.seq_len, data_path=FLAGS.dataset, shuffle=FLAGS.shuffle)
print("Dataset is loaded.")

#compute middle term of eq. (11)
#**********************************************
if FLAGS.wr_init_tofile:
    w_out = tf.Variable(np.random.randn(n_neurons, FLAGS.n_out) / np.sqrt(n_neurons), name='out_weight', dtype=tf.float32)
    #----------------------------------------
    # save to file
    with open('../files/w_out.dat', 'w') as outfile:
        np.savetxt(outfile, w_out, fmt='%-8.6f')
    #----------------------------------------
else:    
    w_out = np.loadtxt("../files/w_out.dat", delimiter=" ", dtype=np.float32)
    w_out = w_out.reshape(n_neurons, FLAGS.n_out)
    w_out = tf.Variable(w_out, name='out_weight', dtype=tf.float32)
#**********************************************
if FLAGS.readout_bias:
    b_out = tf.Variable(np.zeros(FLAGS.n_out), dtype=tf.float32, name="OutBias")


variables = cell_f.trainable_variables + [w_out]
#############################################
#          Create LOSS function             #
#############################################
@tf.function
def classification_loss(classes, filtered_z, w_out):
    output = tf.einsum('btj,jk->btk', filtered_z, w_out)

    if FLAGS.readout_bias:
        output += b_out

    output_mean = tf.reduce_mean(input_tensor=output, axis=1)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes, logits=output_mean)
    loss = tf.reduce_mean(input_tensor=loss)

    output_prediction = tf.argmax(input=output_mean, axis=-1)
    is_correct = tf.equal(classes, output_prediction)
    is_correct_float = tf.cast(is_correct, dtype=tf.float32)
    accuracy  = tf.reduce_mean(input_tensor=is_correct_float)

    output_softmax = tf.nn.softmax(output_mean, axis=-1)
    target_outputs = tf.one_hot(tf.reshape(classes, [tf.shape(input=classes)[0]]), FLAGS.n_out) # tf.one_hot(classes, FLAGS.n_out)

    output_error = output_softmax - target_outputs


    #overall_loss is used as true_gradients for BPTT
    overall_loss = loss #overall_loss is equal to the dreg_loss_dw_in and dreg_loss_dw_rec

    if FLAGS.l2 > 0:
        losses_l2 = [tf.reduce_sum(input_tensor=tf.square(w)) for w in variables]
        overall_loss += FLAGS.l2 * tf.reduce_sum(input_tensor=losses_l2)

    return overall_loss, loss, output_error, accuracy, output_prediction
#############################################
#        Compute gradients                  #
#############################################
@tf.function
def compute_gradients(cell_f, inputs, z, filtered_z, v, b, output_error, w_out, loss, decay):

    # variables = cell_f.trainable_variables + [w_out]

    with tf.GradientTape() as tape:
        # We say which variables are tracked with back-prop through time (BPTT)
        [tape.watch(v) for v in variables]

        if FLAGS.eprop and FLAGS.eprop_impl == 'hardcoded':           
            div_ls = tf.shape(input=filtered_z)[0] * tf.shape(input=filtered_z)[1] # tf.shape(filtered_z)[1] 
            div_ls = tf.cast(div_ls, tf.float32)

            learning_signals = tf.einsum('bk,jk->bj', output_error, w_out)/div_ls # eq. (4) w_out

            # e-traces for input synapses
            n_rec = FLAGS.n_regular + FLAGS.n_adaptive
            
            grad_in_forward, e_trace_in, epsilon_v_in, epsilon_a_in, psi_in = cell_f.compute_loss_gradient(learning_signals[:,:n_rec], 
                                                                        inputs, z[:,:,:n_rec], v[:,:,:n_rec], b[:,:,:n_rec],
                                                                        zero_on_diagonal=False, decay_out=decay)
            
            if FLAGS.l2 > 0:
                grad_in_forward += 2 * FLAGS.l2 *  cell_f.w_in                
            
            # e-traces for recurrent synapses
            z_previous_step = tf.concat([tf.zeros_like(z[:, 0])[:, None], z[:, :-1]], axis=1)   

            grad_rec_forward, e_trace_rec, epsilon_v_rec, epsilon_a_rec, psi_rec  = cell_f.compute_loss_gradient(learning_signals[:,:n_rec], 
                                                            z_previous_step[:,:,:n_rec], z[:,:,:n_rec], v[:,:,:n_rec], b[:,:,:n_rec],
                                                            zero_on_diagonal=True,decay_out=decay)

            if FLAGS.l2 > 0:
                grad_rec_forward += 2 * FLAGS.l2 *  cell_f.w_rec
            
            grad_out = tf.reduce_mean(input_tensor=output_error[:, None, None, :] * filtered_z[:, :, :, None], axis=(0, 1))

            if FLAGS.l2 > 0:
                grad_out += 2 * FLAGS.l2 *  w_out
            
            # concatenate all gradients
            eprop_gradients = [grad_in_forward, grad_rec_forward, grad_out]
        else:
            # This automatically computes the correct gradients in tensor flow
            learning_signals = tf.zeros_like(z)
            eprop_gradients  = tape.gradients(overall_loss, variables)
            assert eprop_gradients ,"No gradients have been computed, grads is an empty list: {}, variables are {}".format(eprop_gradients,variables)
    grads_and_vars   = [(g, v) for g, v in zip(eprop_gradients, variables)]
    opt.apply_gradients(grads_and_vars)

# Loss list to store the loss over itertaions
loss_list = []
accuracy_list = []
valid_loss_list = []
valid_acc_list  = []
# dictionary of tensors that we want to compute simultaneously (most of them are just computed for plotting)

total_parameters = 0

for variable in variables:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim
    total_parameters += variable_parameters
print("______________________________________________")
print("")
print("TOTAL NUM OF PARAMETERS = ", total_parameters)
print("______________________________________________")
print("")
#############################################
#         Train the model                   #
#############################################
def batch_to_feed_dict(batch):
    features_np, classes_np  = batch
    
    std = features_np.std(axis=(0,1), keepdims=True)
    features_np = features_np/std

    classes_np = classes_np.reshape([-1])
    classes_np = classes_np.astype(np.int64)
    
    return features_np, classes_np


def compute_result(type="validation"):
    assert type in ["validation", "test"]
    total_batch_size = dataset.n_develop if type == "validation" else dataset.n_test
    n_minibatch = total_batch_size // FLAGS.n_batch
    mini_batch_sizes = [FLAGS.n_batch for _ in range(n_minibatch)]
    if total_batch_size - (n_minibatch * FLAGS.n_batch) != 0:
        mini_batch_sizes = mini_batch_sizes + [total_batch_size - (n_minibatch * FLAGS.n_batch)]
        n_minibatch += 1

    collect_loss     = [None] * n_minibatch
    collect_accuracy = [None] * n_minibatch

    for idx, mb_size in enumerate(mini_batch_sizes):
        selection = np.arange(mb_size)
        selection = selection + np.ones_like(selection) * idx * FLAGS.n_batch

        if type == "validation":
            data = dataset.get_next_validation_batch(selection)
        elif type == "test":
            data = dataset.get_next_test_batch(selection)

        inputs, classes = batch_to_feed_dict(data)
        z, _, _ = lsnn(inputs)
        filtered_z = exp_convolve(z, decay)
        overall_loss, _, _, accuracy, _ = classification_loss(classes, filtered_z, w_out)
        
        collect_loss[idx]     = overall_loss
        collect_accuracy[idx] = accuracy


    loss_result     = np.mean(collect_loss)
    accuracy_result = np.mean(collect_accuracy)
    return loss_result, accuracy_result

t_train = 0
epoch_last_iteration = -1
epoch_loss = 0.                       # Defines a cost related to an epoch
epoch_accuracy = 0.
Sigma_minibatch_loss = 0.
num_minibatches = int(dataset.n_train / FLAGS.n_batch) # number of minibatches of size minibatch_size in the train set

best_acc = 0
train_acc = 0
k_iter = 0

while dataset.current_epoch <= FLAGS.n_iter:
    # print("**********************************************")
    is_new_epoch = epoch_last_iteration == dataset.current_epoch - 1
    epoch_last_iteration = dataset.current_epoch
    
    lr_temp = lr
    if FLAGS.lr_decay_every > 0 and np.mod(dataset.current_epoch, FLAGS.lr_decay_every) == 0 and dataset.current_epoch>0 and is_new_epoch:
        if lr_temp > 1e-5:
            lr = lr * FLAGS.lr_decay
            opt.lr.assign(lr)
        print('Decay learning rate: {:.2g}'.format(lr))

    # train
    t0 = time()

    inputs, classes = batch_to_feed_dict(dataset.get_next_training_batch())
    z, v, b = lsnn(inputs)
    filtered_z = exp_convolve(z, decay)
    overall_loss, loss, output_error, accuracy, output_prediction = classification_loss(classes, filtered_z, w_out)
    compute_gradients(cell_f, inputs, z, filtered_z, v, b, output_error, w_out, loss, decay)
    # Each time we run this tensorflow operation a weight update is applied
    k_iter += 1

    t_train = time() - t0
    
    minibatch_loss     = overall_loss
    minibatch_accuracy = accuracy
    epoch_loss  += minibatch_loss  / num_minibatches
    epoch_accuracy += minibatch_accuracy / num_minibatches

    Sigma_minibatch_loss += minibatch_loss
            
    if np.mod(k_iter, 1000) == 0: # 500
        print('''prediction {}\nclasses    {}\n''' .format(output_prediction, classes))
        # print('loss_batch: ', Sigma_minibatch_loss)

    if is_new_epoch: # is_new_epoch: # True: #
        print('--------------------------------') 
        print('n_epoch: ', dataset.current_epoch)
        
        # Run the simulation
        t0 = time()

        # results_values = sess.run(results_tensors, feed_dict=feed_dict)
        t_valid = time() - t0
                    
        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max

        av = tf.reduce_mean(input_tensor=z, axis=(0, 1)) / dt # eq. (5) -supp
        firing_rate_stats = get_stats(av * 1000)

        print('''firing rate (Hz)  min {:.0f} \t max {:.0f} \t average {:.0f} +- std {:.0f} (averaged over batches and time)'''.format(
                 firing_rate_stats[0], firing_rate_stats[1], firing_rate_stats[2], firing_rate_stats[3]))

    if is_new_epoch and dataset.current_epoch>0:
        print ("Loss after epoch %i: %f"  % (dataset.current_epoch, epoch_loss))
        print ("Accuracy after epoch %i: %f" % (dataset.current_epoch, epoch_accuracy))

        train_acc = epoch_accuracy
        
        loss_list.append(epoch_loss)
        accuracy_list.append(epoch_accuracy)
        
        np.save('result/train_loss_list', loss_list)
        np.save('result/train_acc_list', accuracy_list)
        
        epoch_loss = 0.                       # Defines a cost related to an epoch
        epoch_accuracy = 0.

        Sigma_minibatch_loss = 0.

    if is_new_epoch and dataset.current_epoch > 0 and np.mod(dataset.current_epoch, 1) == 0:
        loss, accuracy = compute_result("validation")
        valid_loss_list.append(loss)
        valid_acc_list.append(accuracy)
        print('loss {:.3g} (valid),   Accuracy: {:.3g} (valid)'.format(loss, accuracy))
        valid_acc = accuracy
        if valid_acc>=best_acc: #and train_acc>0.92:
            best_acc = valid_acc
            print('\n                                ***Best valid accuracy: {:.4g}***\n'.format(best_acc*100))
            print('Saving weights...') 
            with open('result/w_in.dat', 'w') as outfile:
                    np.savetxt(outfile, cell_f.w_in, fmt='%-8.8f')
            with open('result/w_rec.dat', 'w') as outfile:
                    np.savetxt(outfile, cell_f.w_rec, fmt='%-8.8f')
            with open('result/w_out.dat', 'w') as outfile:
                    np.savetxt(outfile, w_out, fmt='%-8.8f')
            print('--------------------------------') 
        loss, accuracy = compute_result("test")
        print('loss {:.3g} (test),   Accuracy: {:.3g} (test)'.format(loss, accuracy))
        np.save('result/valid_loss_list', valid_loss_list)
        np.save('result/valid_acc_list', valid_acc_list)
        print('--------------------------------') 
 

