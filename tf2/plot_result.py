import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt

# plt.ion()

train_acc_list  = np.load('../result/train_acc_list.npy')
train_loss_list = np.load('../result/train_loss_list.npy')
valid_acc_list = np.load('../result/valid_acc_list.npy')
valid_loss_list = np.load('../result/valid_loss_list.npy')



# plot the cost
fig, ax_list = plt.subplots(nrows=1, ncols=1, figsize=(9.5, 7))
ax = ax_list

ax.set_xlabel('iterations')
ax.set_ylabel('Loss/Accuracy')

ax.plot(np.arange(1, len(train_loss_list)+1), np.squeeze(train_loss_list), linestyle='dashed', label='Train Loss', alpha=0.8, marker='.')
ax.plot(np.arange(1, len(valid_loss_list)+1), np.squeeze(valid_loss_list), linestyle='solid',  label='Valid Loss', alpha=0.8, marker='.')
# ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax.plot(np.arange(1, len(train_acc_list)+1), np.squeeze(train_acc_list), linestyle='dashed', label='Train Accuracy', alpha=0.8, marker='.')
ax.plot(np.arange(1, len(valid_acc_list)+1), np.squeeze(valid_acc_list), linestyle='solid',  label='Valid Accuracy', alpha=0.8, marker='.')

ax.set_ylim([0,1])
ax.set_xlim([1,30])
ax.legend()
minor_ticks = np.arange(0, 1, 0.05)
major_ticks = np.arange(0, 1, 0.1)
ax.set_yticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.grid(which='minor', axis='y', linestyle=':')
ax.grid(which='major', axis='y', linewidth=1)

# ax2.legend()
ax.set_title('E-PROP + LSNN + MG')
plt.savefig('../result/loss_list_eprop_BI_RNN_LSNN.png', bbox_inches='tight')
plt.show()  


# plt.ioff()


