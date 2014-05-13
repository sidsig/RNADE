from RNADE import RNADE
from datasets import Dataset
#import mocap_data
from SGD import SGD_Optimiser
import pdb
import numpy
import cPickle
import theano
import scipy.io
import h5py


#rnade.params = [W,b_alpha,V_alpha,b_mu,V_mu,b_sigma,V_sigma,activation_rescaling]

##mocap dataset
# A = scipy.io.loadmat('data/MOCAP')
# data = A['batchdata']

# new_data = numpy.zeros(data.shape)
# new_data[:] = data
#pdb.set_trace()

# batch_size = 100
# num_examples = 100
# train_data = mocap_data.sample_train_seq(batch_size)
# for i in xrange(1,num_examples):
#     train_data = numpy.vstack((train_data,mocap_data.sample_train_seq(batch_size)))
# numpy.random.shuffle(train_data)

#red wine dataset
dataset_path = '/homes/sss31/PhD/rnade_release/red_wine.hdf5'
h5 = h5py.File(dataset_path,'r')
new_data = h5['all']['data'][:]
num_examples = new_data.shape[0]
numpy.random.shuffle(new_data)
#nade parameters
n_visible = new_data.shape[-1]
n_hidden = 50
n_components = 2
hidden_act = 'sigmoid'
l2 = 0.001
train_dataset = Dataset([new_data],100)

#Experiment with theano grads
rnade = RNADE(n_visible,n_hidden,n_components,hidden_act=hidden_act,l2=l2)
rnade.build_fprop()
optimiser = SGD_Optimiser(rnade.params,[rnade.v],[rnade.cost],momentum=True,patience=500,custom_grads=False,)
optimiser.train(train_dataset,valid_set=None,learning_rate=0.001,mom_rate=0.9,num_epochs=1000,save=False,
            lr_update=False,update_type='linear',start=2)

#Experiment with custom grads
# rnade = RNADE(n_visible,n_hidden,n_components,hidden_act=hidden_act,l2=l2)
# rnade.build_fprop_two()
# optimiser = SGD_Optimiser(rnade.params,[rnade.v],[rnade.cost],momentum=True,patience=500,custom_grads=True,custom_grad_dict=rnade.grads)
# optimiser.train(train_dataset,valid_set=None,learning_rate=0.001,num_epochs=1000,save=False,
#             lr_update=True,update_type='linear',start=2)