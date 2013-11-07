import csv
import numpy
import os
import math
import sys
import time

from sklearn import preprocessing

import pylab as pl

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from DBN_prediction import DBN
from DBN_prediction_GRBM import DBN_GRBM

def load_data_from_csv(dataset_filename, sli_win_size):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset
	'''

	#############
	# LOAD DATA #
	#############
 
	# Load the dataset
	dataset = numpy.genfromtxt(dataset_filename, delimiter=',')

	for x in xrange(10):
		print dataset[x,:]	
	# with open(dataset_filename, 'rb') as f:
	# 	reader = csv.reader(f)
	# 	for row in reader:
	# 		print row
	
	#print 'datasetset rows: %i columns: %i' % (dataset.shape[0], dataset.shape[1])
	# print dataset[0,:]
	# print dataset[1,:]
	#max_v = dataset[:,0].max(axis=0)
	#min_v = dataset[:,0].min(axis=0)

	flowGroup = range(sli_win_size)
	#flowGroup.extend([sli_win_size+5, sli_win_size+6, sli_win_size+7])
	flowGroup.extend([sli_win_size+6, sli_win_size+7])

	max_v = dataset[:,flowGroup].max()
	min_v = dataset[:,flowGroup].min()

	scale_factors_y = (min_v, max_v)

	print max_v
	print min_v

	f_v = dataset[0,0]
	first_val_sc = (f_v - min_v)/(max_v - min_v)

	print f_v
	print first_val_sc

	for i in flowGroup:
		dataset[:, i] = (dataset[:,i] - min_v) /(max_v - min_v)
	
	# dataset[:, 0] = (dataset[:,0] - min_v) /(max_v - min_v)
	# dataset[:, 6] = (dataset[:,6] - min_v) /(max_v - min_v)
	# dataset[:, 7] = (dataset[:,7] - min_v) /(max_v - min_v)
	# dataset[:, 8] = (dataset[:,8] - min_v) /(max_v - min_v)

	dataset[:, sli_win_size + 0] = (dataset[:,sli_win_size + 0] - dataset[:,sli_win_size + 0].min(axis=0)) /(dataset[:,sli_win_size + 0].max(axis=0) - dataset[:,sli_win_size + 0].min(axis=0))
	dataset[:, sli_win_size + 1] = (dataset[:,sli_win_size + 1] - dataset[:,sli_win_size + 1].min(axis=0)) /(dataset[:,sli_win_size + 1].max(axis=0) - dataset[:,sli_win_size + 1].min(axis=0))
	dataset[:, sli_win_size + 2] = (dataset[:,sli_win_size + 2] - dataset[:,sli_win_size + 2].min(axis=0)) /(dataset[:,sli_win_size + 2].max(axis=0) - dataset[:,sli_win_size + 2].min(axis=0))
	dataset[:, sli_win_size + 3] = dataset[:,sli_win_size + 3]/ 100.
	dataset[:, sli_win_size + 4] = (dataset[:,sli_win_size + 4] - dataset[:,sli_win_size + 4].min(axis=0)) /(dataset[:,sli_win_size + 4].max(axis=0) - dataset[:,sli_win_size + 4].min(axis=0))
	dataset[:, sli_win_size + 5] = (dataset[:,sli_win_size + 5] - dataset[:,sli_win_size + 5].min(axis=0)) /(dataset[:,sli_win_size + 5].max(axis=0) - dataset[:,sli_win_size + 5].min(axis=0))

	for x in xrange(10):
		print dataset[x,:]

	#min_max_scaler = preprocessing.MinMaxScaler((0, 1))
	##dataset = preprocessing.scale(dataset)
	##dataset = min_max_scaler.fit_transform(dataset)
	#min_max_scaler.fit(dataset)

	# print 'Scale:'
	# print  min_max_scaler.scale_  
	# print 'Min:'
	# print  min_max_scaler.min_
	# print 'Params:'
	# print min_max_scaler.get_params()

	#dataset = min_max_scaler.transform(dataset)

	train_set_rows = math.trunc(dataset.shape[0] * 0.7)
	valid_set_rows = math.trunc(dataset.shape[0] * 0.15)
	test_set_rows = dataset.shape[0] - train_set_rows - valid_set_rows

	begin_row = 0
	end_row = train_set_rows-1
	train_set_a = dataset[begin_row:end_row, :]
	numpy.random.shuffle(train_set_a)
	train_set_x_a = dataset[:, :train_set_a.shape[1]-1]
	train_set_y_a = dataset[:, train_set_a.shape[1]-1:].flatten()

	#train_set_x_a = dataset[begin_row:end_row, :dataset.shape[1]-1]
	#print 'P: train_set_x rows: %i columns: %i' % (train_set_x.shape[0], train_set_x.shape[1])
	#print train_set_x[0,:]
	#train_set_y_a = dataset[begin_row:end_row, dataset.shape[1]-1:].flatten()
	#print 'P: train_set_y rows: %i columns: %i' % (train_set_y.shape[0], train_set_y.shape[1])
	# print train_set_y.shape[0]
	# print train_set_y.shape[1]
	#print train_set_y[0,0]

	begin_row = end_row
	end_row = end_row + valid_set_rows
	valid_set_x_a = dataset[begin_row:end_row, :dataset.shape[1]-1]
	valid_set_y_a = dataset[begin_row:end_row, dataset.shape[1]-1:].flatten()

	begin_row = end_row
	end_row = end_row + test_set_rows
	test_set_x_a = dataset[begin_row:end_row, :dataset.shape[1]-1]
	test_set_y_a = dataset[begin_row:end_row, dataset.shape[1]-1:].flatten()

	train_set = (train_set_x_a, train_set_y_a)
	valid_set = (valid_set_x_a, valid_set_y_a)
	test_set = (test_set_x_a, test_set_y_a)

	print 'train_set_x rows: %i columns: %i' % (train_set[0].shape[0], train_set[0].shape[1])
	print 'train_set_y rows: %i' % train_set[1].shape[0]

	print 'valid_set_x rows: %i columns: %i' % (valid_set[0].shape[0], valid_set[0].shape[1])
	print 'valid_set_y rows: %i' % valid_set[1].shape[0]

	print 'test_set_x rows: %i columns: %i' % (test_set[0].shape[0], test_set[0].shape[1])
	print 'test_set_y rows: %i' % test_set[1].shape[0]
	#print test_set[0][0,:]

	for x in xrange(10):
		dataT_x, dataT_y = train_set
		print dataT_x[x,:]
		print dataT_y[x]	

	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#witch row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	def shared_dataset(data_xy, borrow=True):
		""" Function that loads the dataset into shared variables

		The reason we store our dataset in shared variables is to allow
		Theano to copy it into the GPU memory (when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch everytime
		is needed (the default behaviour if the data is not in a shared
		variable) would lead to a large decrease in performance.
		"""
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ``floatX`` as well
		# (``shared_y`` does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn't make sense) therefore instead of returning
		# ``shared_y`` we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, shared_y

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	#print 'Theano train_set_y %i' % len(train_set_y)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
			(test_set_x, test_set_y)]
	return rval, scale_factors_y

def run_dbn_tfp(finetune_lr=0.3, pretraining_epochs=200,
			 pretrain_lr=0.005, k=1, training_epochs=1000,
			 batch_size=30):

	datasets, scale_factors_y = load_data_from_csv('C:\\Users\\vivanac\\Desktop\\SkyDrive\\England\\DeepLearningTutorials\\data\\dataset_10_7_3.csv', 10);

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test = test_set_x.get_value(borrow=True).shape[0]
	n_valid = valid_set_x.get_value(borrow=True).shape[0]
	print 'n_train_batches %i' % n_train_batches

	# numpy random generator
	numpy_rng = numpy.random.RandomState(123)
	print '... building the model'
	# construct the Deep Belief Network
	dbn = DBN_GRBM(numpy_rng=numpy_rng, n_ins=17,
			  hidden_layers_sizes=[34, 34],
			  n_outs=1)

	#########################
	# PRETRAINING THE MODEL #
	#########################
	print '... getting the pretraining functions'
	pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
												batch_size=batch_size,
												k=k)

	print '... pre-training the model'
	start_time = time.clock()
	## Pre-train layer-wise
	for i in xrange(dbn.n_layers):
		# go through pretraining epochs
		for epoch in xrange(pretraining_epochs):
			# go through the training set
			c = []
			for batch_index in xrange(n_train_batches):
				# if i == 0:
				# 	c.append(pretraining_fns[i](index=batch_index,
				# 							lr=0.005))
				# else:
				c.append(pretraining_fns[i](index=batch_index,
											lr=pretrain_lr))
			print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
			print numpy.mean(c)

	end_time = time.clock()
	print >> sys.stderr, ('The pretraining code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))

	########################
	# FINETUNING THE MODEL #
	########################

	# get the training, validation and testing function for the model
	print '... getting the finetuning functions'
	train_fn, validate_model, test_model, test_model_rmse, evaluation_model = dbn.build_finetune_functions(
				datasets=datasets, batch_size=batch_size,
				learning_rate=finetune_lr)

	#print 'Number of finetune_functions %i' % train_fn.size[0]

	print '... finetunning the model'
	# early-stopping parameters
	patience = 8 * n_train_batches  # look as this many examples regardless
	patience_increase = 4.    # wait this much longer when a new best is
							  # found
	improvement_threshold = 0.9995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(n_train_batches, patience / 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	best_params = None
	best_validation_loss = numpy.inf
	test_score = 0.
	test_rmse = 0.
	start_time = time.clock()

	done_looping = False
	epoch = 0

	print 'patience %i' % patience
	print 'n_train_batches %i' % n_train_batches
	print 'validation freq %i' % validation_frequency

	while (epoch < training_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			#print 'minibatch_index %i' % minibatch_index
			minibatch_avg_cost = train_fn(minibatch_index)
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:

				validation_losses = validate_model()
				this_validation_loss = numpy.sum(validation_losses) / n_valid
				print('epoch %i, minibatch %i/%i, validation error %f %%' % \
					  (epoch, minibatch_index + 1, n_train_batches,
					   this_validation_loss * 100.))

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:

					#improve patience if loss improvement is good enough
					if (this_validation_loss < best_validation_loss *
						improvement_threshold):
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = test_model()
					test_score = numpy.sum(test_losses) / n_test
					print(('epoch %i, minibatch %i/%i, test error of '
						   'best model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))

					test_losses_rmse = test_model_rmse()
					test_score_rmse = numpy.sqrt(numpy.sum(test_losses_rmse)/n_test)
					print(('epoch %i, minibatch %i/%i, test RMSE of '
						   'best model %f') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score_rmse))

			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print(('Optimization complete with best validation score of %f %%,'
		   'with test performance %f %% and RMSE %f') %
				 (best_validation_loss * 100., test_score * 100., test_score_rmse))
	print >> sys.stderr, ('The fine tuning code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time)
											  / 60.))

	#run whole test set and save data to csv
	rmse_val, mape_val, y_pred, y_target = evaluation_model()

	print rmse_val
	print mape_val

	min_scale, max_scale = scale_factors_y
	y_pred = y_pred*(max_scale - min_scale) + min_scale
	y_target = y_target*(max_scale - min_scale) + min_scale
	#y_shared = test_set_y

	#y_pred_shared = theano.shared(numpy.asarray(y_pred, dtype=theano.config.floatX), borrow=True)
	rmse_val2 = numpy.sqrt(numpy.sum(numpy.square(y_pred - y_target))/n_test)
	mape_val2 = numpy.sum(numpy.abs((y_pred - y_target)/y_target))/n_test
	
	print rmse_val2
	print mape_val2
	#print y_pred.shape[0]
	#print y_shared.get_value(borrow=True).shape[0]
	#print y_pred_shared.get_value(borrow=True).shape[0]

	#y_dataset = T.concatenate([y_shared, y_pred_shared], axis=0)

	#y_dataset = numpy.concatenate((y_target, y_pred), axis=0)	
	y_dataset = numpy.column_stack((y_target.flatten(),y_pred.flatten()))

	numpy.savetxt('C:\\Users\\vivanac\\Desktop\\SkyDrive\\England\\DeepLearningTutorials\\data\\result.csv', y_dataset, fmt='%f',delimiter=",")

	pl.plot(range(n_test), y_pred)
	pl.plot(range(n_test), y_target)
	pl.show()

if __name__ == '__main__':
	run_dbn_tfp()