import sys

sys.path.append("../../..")

from utility_classes.time_logger import TimeLogger as logger
from utility_classes.time_logger import TimeLoggerCheckpointSaverListener as checkpoint_listener
from utility_classes.timeliner import TimeLiner as timeliner

import pickle

import tensorflow as tf
import tensorflow.contrib.keras as tfk
from tensorflow.python.ops import array_ops
from tensorflow.python.client import timeline

import numpy as np
import argparse

# instead of scipy, use PIL directly to save images
try:
    import PIL
    def imsave(filename, data):
        PIL.Image.fromarray(data.astype(np.uint8)).save(filename)
    have_imsave = True
except ImportError:
    have_imsave = False
    
import h5py as h5
import os
import time

# limit tensorflow spewage to just warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

use_nvtx = False
if (use_nvtx):
  import cupy.cuda.nvtx as nvtx
else:
  class nvtx_dummy:
    def RangePush(self, name, color):
      pass
    def RangePop(self):
      pass
  nvtx = nvtx_dummy()

#horovod, yes or no?
horovod=True
try:
    import horovod.tensorflow as hvd
except:
    horovod = False

#import helpers
from tiramisu_helpers import *

#GLOBAL CONSTANTS
image_height =  768 
image_width = 1152


def conv(x, nf, sz, wd, stride=1):
    return tf.layers.conv2d(inputs=x, filters=nf, kernel_size=sz, strides=(stride,stride),
                            padding='same', data_format='channels_first',
                            kernel_initializer= tfk.initializers.he_uniform(),
                            bias_initializer=tf.initializers.zeros(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=wd)
                            )


def dense_block(n, x, growth_rate, p, wd, training, bn=False, filter_sz=3):

    added = []
    for i in range(n):
        if bn:
            with tf.name_scope("conv_bn_relu%i"%i) as scope:
                b = conv(x, growth_rate, sz=filter_sz, wd=wd)
                b = tf.layers.batch_normalization(b, axis=1, training=training)
                b = tf.nn.relu(b)
                if p: b = tf.layers.dropout(b, rate=p, training=training)
        else:
            with tf.name_scope("conv_relu%i"%i) as scope:
                b = conv(x, growth_rate, sz=filter_sz, wd=wd)
                b = tf.nn.relu(b)
                if p: b = tf.layers.dropout(b, rate=p, training=training)

        x = tf.concat([x, b], axis=1) #was axis=-1. Is that correct?
        added.append(b)

    return x, added


def transition_dn(x, p, wd, training, bn=False):
    if bn:
        with tf.name_scope("conv_bn_relu") as scope:
            b = conv(x, x.get_shape().as_list()[1], sz=1, wd=wd, stride=2) #was [-1]. Filters are at 1 now.
            b = tf.layers.batch_normalization(b, axis=1, training=training)
            b = tf.nn.relu(b)
            if p: b = tf.layers.dropout(b, rate=p, training=training)
    else:
        with tf.name_scope("conv_relu") as scope:
            b = conv(x, x.get_shape().as_list()[1], sz=1, wd=wd, stride=2)
            b = tf.nn.relu(b)
            if p: b = tf.layers.dropout(b, rate=p, training=training)
    return b


def down_path(x, nb_layers, growth_rate, p, wd, training, bn=False, filter_sz=3):

    skips = []
    for i,n in enumerate(nb_layers):
        with tf.name_scope("DB%i"%i):
            x, added = dense_block(n, x, growth_rate, p, wd, training=training, bn=bn, filter_sz=filter_sz)
            skips.append(x)
        with tf.name_scope("TD%i"%i):
            x = transition_dn(x, p=p, wd=wd, training=training, bn=bn)

    return skips, added


def reverse(a): 
	return list(reversed(a))


def transition_up(added,wd,training):
    x = tf.concat(added,axis=1) 
    _, ch, r, c = x.get_shape().as_list()
    x = tf.layers.conv2d_transpose(inputs=x,strides=(2,2),kernel_size=(3,3),
				   padding='same', data_format='channels_first', filters=ch,
				   kernel_initializer=tfk.initializers.he_uniform(),
				   bias_initializer=tf.initializers.zeros(),
                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=wd)
                   )
    return x 
    
	
def up_path(added,skips,nb_layers,growth_rate,p,wd,training,bn=False,filter_sz=3):
    for i,n in enumerate(nb_layers):
        x = transition_up(added,wd,training)
        x = tf.concat([x,skips[i]],axis=1) #was axis=-1. Is that correct?
        x, added = dense_block(n,x,growth_rate,p,wd,training=training,bn=bn,filter_sz=filter_sz)
    return x


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def create_tiramisu(nb_classes, img_input, height, width, nc, loss_weights, nb_dense_block=6, 
                    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0., training=True, batchnorm=False, dtype=tf.float16, filter_sz=3, comm_rank=-1):
    create_tiramisu_timer_logger = logger(comm_rank, "Create Tiramisu", -1, True)
    create_tiramisu_timer_logger.start_timer()

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    with tf.variable_scope("tiramisu", custom_getter=float32_variable_storage_getter):

        with tf.variable_scope("conv_input") as scope:
            x = conv(img_input, nb_filter, sz=filter_sz, wd=wd)
            if batchnorm:
                x = tf.layers.batch_normalization(x, axis=1, training=training)
            x = tf.nn.relu(x)
            if p: x = tf.layers.dropout(x, rate=p, training=training)

        with tf.name_scope("down_path") as scope:
            skips,added = down_path(x, nb_layers, growth_rate, p, wd, training=training, bn=batchnorm, filter_sz=filter_sz)
        
        with tf.name_scope("up_path") as scope:
            x = up_path(added, reverse(skips[:-1]),reverse(nb_layers[:-1]), growth_rate, p, wd, training=training, bn=batchnorm, filter_sz=filter_sz)

        with tf.name_scope("conv_output") as scope:
            x = conv(x,nb_classes,sz=1,wd=wd)
            if p: x = tf.layers.dropout(x, rate=p, training=training)
            _,f,r,c = x.get_shape().as_list()
        #x = tf.reshape(x,[-1,nb_classes,image_height,image_width]) #nb_classes was last before
        x = tf.transpose(x,[0,2,3,1]) #necessary because sparse softmax cross entropy does softmax over last axis

    create_tiramisu_timer_logger.end_timer()
    return x, tf.nn.softmax(x)



def create_dataset(h5ir, datafilelist, batchsize, num_epochs, comm_size, comm_rank, dtype, shuffle=False):
    create_dataset_timer_logger = logger(comm_rank, "Create Dataset", -1, True)
    create_dataset_timer_logger.start_timer()

    if comm_size > 1:
        # use an equal number of files per shard, leaving out any leftovers
        per_shard = len(datafilelist) // comm_size
        sublist = datafilelist[0:per_shard * comm_size]
        dataset = tf.data.Dataset.from_tensor_slices(sublist)
        dataset = dataset.shard(comm_size, comm_rank)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(datafilelist)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(map_func=lambda dataname: tuple(tf.py_func(h5ir.read, [dataname], [dtype, tf.int32, dtype])),
                          num_parallel_calls = 4)
    dataset = dataset.prefetch(16)
    # make sure all batches are equal in size
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batchsize))
    dataset = dataset.repeat(num_epochs)

    create_dataset_timer_logger.end_timer()

    return dataset


#                                       label predict color
colormap = np.array([[[  0,  0,  0],  #   0      0     black
                      [255,  0,255],  #   0      1     purple
                      [  0,255,255]], #   0      2     cyan
                     [[  0,255,  0],  #   1      0     green
                      [128,128,128],  #   1      1     grey
                      [255,255,  0]], #   1      2     yellow
                     [[255,  0,  0],  #   2      0     red
                      [  0,  0,255],  #   2      1     blue
                      [255,255,255]], #   2      2     white
                     ])


# Timeline related functions
# TODO: Encapsulate in a different class later
def init_timeline_configs(enable_tf_timeline, trace_level, min_timeline_step, max_timeline_step):
    options = None
    run_metadata = None
    many_runs_timeline = None

    if enable_tf_timeline:
        options = tf.RunOptions(trace_level=trace_level)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = timeliner()

    return options, run_metadata, many_runs_timeline, min_timeline_step, max_timeline_step


def update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline, train_steps=-1,
                             min_step=-1, max_step=-1):
    if enable_tf_timeline and train_steps >= min_step and train_steps <= max_step:
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.update_timeline(chrome_trace)


#main function
def main(input_path, blocks, weights, image_dir, checkpoint_dir, trn_sz, learning_rate, loss_type, fs_type, opt_type, batch, batchnorm, num_epochs, dtype, chkpt, filter_sz, growth, disable_training, enable_tf_timeline):
    options = None
    run_metadata = None
    many_runs_timeline = None

    timeline_trace_fp = open("timeline_trace.pickle", "wb")

    options, run_metadata, many_runs_timeline, min_timeline_step, max_timeline_step = \
        init_timeline_configs(enable_tf_timeline, tf.RunOptions.FULL_TRACE, 5, 6)

    global_time_logger = logger(-1, "Global Total Time", -1, True)
    global_time_logger.start_timer()

    #init horovod

    initialization_timer_logger = logger(-1, "Initialize Horovod", -1, True)
    initialization_timer_logger.start_timer()

    nvtx.RangePush("init horovod", 1)
    comm_rank = 0
    comm_local_rank = 0
    comm_size = 1
    comm_local_size = 1
    if horovod:
        hvd.init()
        comm_rank = hvd.rank()
        comm_local_rank = hvd.local_rank()
        comm_size = hvd.size()
        #not all horovod versions have that implemented
        try:
            comm_local_size = hvd.local_size()
        except:
            comm_local_size = 1
        if comm_rank == 0:
            print("Using distributed computation with Horovod: {} total ranks".format(comm_size,comm_rank))
    nvtx.RangePop() # init horovod

    initialization_timer_logger.set_rank(int(comm_rank))
    initialization_timer_logger.end_timer()

    global_time_logger.set_rank(int(comm_rank))

    #parameters
    channels = [0,1,2,10]
    per_rank_output = False
    loss_print_interval = 1
    
    #session config

    initialization_timer_logger.start_timer(comm_rank, "Configure Session")

    sess_config=tf.ConfigProto(inter_op_parallelism_threads=6, #1
                               intra_op_parallelism_threads=1, #6
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(comm_local_rank)

    initialization_timer_logger.end_timer()

    #get data

    initialization_timer_logger.start_timer(comm_rank, "Get Data")

    training_graph = tf.Graph()
    if comm_rank == 0:
        print("Loading data...")
    trn_data, val_data, tst_data = load_data(input_path, trn_sz, comm_rank)
    if comm_rank == 0:    
        print("Shape of trn_data is {}".format(trn_data.shape[0]))
        print("done.")

    initialization_timer_logger.end_timer()

    #print some stats
    if comm_rank==0:
        print("Learning Rate: {}".format(learning_rate))
        print("Num workers: {}".format(comm_size))
        print("Local batch size: {}".format(batch))
        if dtype == tf.float32:
            print("Precision: {}".format("FP32"))
        else:
            print("Precision: {}".format("FP16"))
        print("Batch normalization: {}".format(batchnorm))
        print("Blocks: {}".format(blocks))
        print("Growth rate: {}".format(growth))
        print("Filter size: {}".format(filter_sz))
        print("Channels: {}".format(channels))
        print("Loss type: {}".format(loss_type))
        print("Loss weights: {}".format(weights))
        print("Optimizer type: {}".format(opt_type))
        print("Num training samples: {}".format(trn_data.shape[0]))
        print("Num validation samples: {}".format(val_data.shape[0]))

    io_training_time_logger = logger(comm_rank, "IO and Training", -1, True)
    io_training_time_logger.start_timer()

    with training_graph.as_default():
        nvtx.RangePush("TF Init", 3)
        #create readers
        trn_reader = h5_input_reader(input_path, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, comm_rank=comm_rank)
        val_reader = h5_input_reader(input_path, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, comm_rank=comm_rank)
        #create datasets
        if fs_type == "local":
            trn_dataset = create_dataset(trn_reader, trn_data, batch, num_epochs, comm_local_size, comm_local_rank, dtype, shuffle=True)
            val_dataset = create_dataset(val_reader, val_data, batch, 1, comm_local_size, comm_local_rank, dtype, shuffle=False)
        else:
            trn_dataset = create_dataset(trn_reader, trn_data, batch, num_epochs, comm_size, comm_rank, dtype, shuffle=True)
            val_dataset = create_dataset(val_reader, val_data, batch, 1, comm_size, comm_rank, dtype, shuffle=False)
        
        #create iterators
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, (dtype, tf.int32, dtype),
                                                       ((batch, len(channels), image_height, image_width),
                                                        (batch, image_height, image_width),
                                                        (batch, image_height, image_width))
                                                       )
        next_elem = iterator.get_next()
        
        #create init handles
        #trn
        trn_iterator = trn_dataset.make_initializable_iterator()
        trn_handle_string = trn_iterator.string_handle()
        trn_init_op = iterator.make_initializer(trn_dataset)
        #val
        val_iterator = val_dataset.make_initializable_iterator()
        val_handle_string = val_iterator.string_handle()
        val_init_op = iterator.make_initializer(val_dataset)

        #set up model
        logit, prediction = create_tiramisu(3, next_elem[0], image_height, image_width, len(channels), loss_weights=weights, nb_layers_per_block=blocks, p=0.2, wd=1e-4, dtype=dtype, batchnorm=batchnorm, growth_rate=growth, filter_sz=filter_sz, comm_rank=comm_rank)
        
        #set up loss
        labels_one_hot = tf.cast(tf.contrib.layers.one_hot_encoding(next_elem[1], 3), dtype=dtype)
        loss = None
        if loss_type == "weighted":
            loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_one_hot, logits=logit, weights=next_elem[2])
        elif loss_type == "focal":
            loss = focal_loss(onehot_labels=labels_one_hot, logits=logit, alpha=1., gamma=2.)
        else:
            raise ValueError("Error, loss type {} not supported.",format(loss_type))
        if horovod:
            loss_avg = hvd.allreduce(tf.cast(loss, tf.float32))
        else:
            loss_avg = tf.identity(loss)

        #set up global step
        global_step = tf.train.get_or_create_global_step()

        #set up optimizer
        if opt_type.startswith("LARC"):
            if comm_rank==0:
                print("Enabling LARC")
            train_op = get_larc_optimizer(opt_type.split("-")[1], loss, global_step, learning_rate, LARC_mode="clip", LARC_eta=0.002, LARC_epsilon=1./16000.)
        else:
            train_op = get_optimizer(opt_type, loss, global_step, learning_rate)
        #set up streaming metrics
        iou_op, iou_update_op = tf.metrics.mean_iou(labels=next_elem[1],
                                                    predictions=tf.argmax(prediction, axis=3),
                                                    num_classes=3,
                                                    weights=None,
                                                    metrics_collections=None,
                                                    updates_collections=None,
                                                    name="iou_score")
        iou_reset_op = tf.variables_initializer([ i for i in tf.local_variables() if i.name.startswith('iou_score/') ])

        if horovod:
            iou_avg = hvd.allreduce(iou_op)
        else:
            iou_avg = tf.identity(iou_op)

        #compute epochs and stuff:
        if fs_type == "local":
            num_samples = trn_data.shape[0] // comm_local_size
        else:
            num_samples = trn_data.shape[0] // comm_size
        #num_steps_per_epoch = num_samples // batch
        num_steps_per_epoch = 10
        num_steps = num_epochs*num_steps_per_epoch
        if per_rank_output:
            print("Rank {} does {} steps per epoch".format(comm_rank, num_steps_per_epoch))
        
        #hooks
        #these hooks are essential. regularize the step hook by adding one additional step at the end
        hooks = [tf.train.StopAtStepHook(last_step=num_steps+1)]
        #bcast init for bcasting the model after start
        init_bcast = hvd.broadcast_global_variables(0)
        #initializers:
        init_op =  tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        
        #checkpointing
        if comm_rank == 0:
            checkpoint_save_freq = num_steps_per_epoch * 2
            checkpoint_saver = tf.train.Saver(max_to_keep = 1000)
            listener = checkpoint_listener(comm_rank, True)
            hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq,
                                                      saver=checkpoint_saver, listeners=[listener]))
            #create image dir if not exists
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
        
        ##DEBUG
        ##summary
        #if comm_rank == 0:
        #    print("write graph for debugging")
        #    tf.summary.scalar("loss",loss)
        #    summary_op = tf.summary.merge_all()
        #    #hooks.append(tf.train.SummarySaverHook(save_steps=num_steps_per_epoch, summary_writer=summary_writer, summary_op=summary_op))
        #    with tf.Session(config=sess_config) as sess:
        #        sess.run([init_op, init_local_op])
        #        #create iterator handles
        #        trn_handle = sess.run(trn_handle_string)
        #        #init iterators
        #        sess.run(trn_init_op, feed_dict={handle: trn_handle, datafiles: trn_data, labelfiles: trn_labels})
        #        #summary:
        #        sess.run(summary_op, feed_dict={handle: trn_handle})
        #        #summary file writer
        #        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        ##DEBUG
        

        #start session
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            #initialize
            sess.run([init_op, init_local_op])

            #restore from checkpoint:
            if comm_rank == 0:
                load_model(sess, checkpoint_saver, checkpoint_dir, comm_rank)
            #broadcast loaded model variables
            sess.run(init_bcast)

            #create iterator handles
            trn_handle, val_handle = sess.run([trn_handle_string, val_handle_string], options=options,
                                               run_metadata=run_metadata)

            update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline)

            #init iterators
            sess.run(trn_init_op, feed_dict={handle: trn_handle}, options=options, run_metadata=run_metadata)

            update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline)

            sess.run(val_init_op, feed_dict={handle: val_handle}, options=options, run_metadata=run_metadata)

            update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline)

            nvtx.RangePop()  # TF Init

            # do the training
            epoch = 1
            step = 1
            train_loss = 0.
            nvtx.RangePush("Training Loop", 4)
            nvtx.RangePush("Epoch", epoch)
            start_time = time.time()

            training_loop_timer_logger = logger(comm_rank, "Training Loop", -1, True)
            training_loop_timer_logger.start_timer()

            train_steps = 0
            while not (sess.should_stop()):
                #training loop
                try:
                    training_iteration_time_logger = logger(comm_rank, "Training Iteration", epoch, True)
                    training_iteration_time_logger.start_timer()

                    nvtx.RangePush("Step", step)

                    if disable_training:
                        train_steps = sess.run([global_step], feed_dict={handle: trn_handle}, options=options,
                                               run_metadata=run_metadata)

                        update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline, train_steps[0],
                                                 min_timeline_step, max_timeline_step)

                        train_steps_in_epoch = train_steps[0] % num_steps_per_epoch

                        # do the validation phase
                        if train_steps_in_epoch == 0:
                            eval_steps = 0
                            while True:
                                try:
                                    sess.run([next_elem[1]], feed_dict={handle: val_handle}, options=options,
                                             run_metadata=run_metadata)

                                    update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline)

                                    eval_steps += 1
                                except tf.errors.OutOfRangeError:
                                    sess.run(val_init_op, feed_dict={handle: val_handle}, options=options,
                                             run_metadata=run_metadata)

                                    update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline)

                                    break

                    else:
                        # construct feed dict
                        _, train_steps, tmp_loss = sess.run([train_op,
                                                             global_step,
                                                             (loss if per_rank_output else loss_avg)],
                                                            feed_dict={handle: trn_handle}, options=options,
                                                            run_metadata=run_metadata)

                        update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline, train_steps,
                                                 min_timeline_step, max_timeline_step)

                        step_trace_fp = open("train_step_trace_" + str(epoch) + str(step) +
                                             str(time.time()) + ".pickle", "wb")

                        pickle.dump(run_metadata, step_trace_fp)

                        train_steps_in_epoch = train_steps%num_steps_per_epoch
                        train_loss += tmp_loss
                        nvtx.RangePop() # Step
                        step += 1

                        #print step report
                        eff_steps = train_steps_in_epoch if (train_steps_in_epoch > 0) else num_steps_per_epoch
                        if (train_steps % loss_print_interval) == 0:
                            if per_rank_output:
                                print("REPORT: rank {}, training loss for step {} (of {}) is {}, time {}".format(comm_rank, train_steps, num_steps, train_loss/eff_steps,time.time()-start_time))
                            else:
                                if comm_rank == 0:
                                    print("REPORT: training loss for step {} (of {}) is {}, time {}".format(train_steps, num_steps, train_loss/eff_steps,time.time()-start_time))

                        #do the validation phase
                        if train_steps_in_epoch == 0:
                            end_time = time.time()
                            #print epoch report
                            train_loss /= num_steps_per_epoch
                            if per_rank_output:
                                print("COMPLETED: rank {}, training loss for epoch {} (of {}) is {}, time {} s".format(comm_rank, epoch, num_epochs, train_loss, time.time() - start_time))
                            else:
                                if comm_rank == 0:
                                    print("COMPLETED: training loss for epoch {} (of {}) is {}, time {} s".format(epoch, num_epochs, train_loss, time.time() - start_time))

                            #evaluation loop
                            eval_loss = 0.
                            eval_steps = 0
                            nvtx.RangePush("Eval Loop", 7)
                            timeline_help_count = 0
                            while True:
                                try:
                                    #construct feed dict
                                    _, tmp_loss, val_model_predictions, val_model_labels = sess.run([iou_update_op,
                                                                                                     (loss if per_rank_output else loss_avg),
                                                                                                     prediction,
                                                                                                     next_elem[1]],
                                                                                                    feed_dict={handle: val_handle},
                                                                                                    options=options,
                                                                                                    run_metadata=run_metadata)

                                    update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline,
                                                             timeline_help_count,
                                                             min_timeline_step, max_timeline_step)

                                    step_trace_fp = open("validation_step_trace_" + str(epoch) + str(step) +
                                                         str(time.time()) + ".pickle", "wb")

                                    pickle.dump(run_metadata, step_trace_fp)

                                    timeline_help_count += 1

                                    #print some images
                                    if comm_rank == 0:
                                        if have_imsave:
                                            imsave(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'
                                                   +str(eval_steps)+'_rank'+str(comm_rank)+'.png',np.argmax(val_model_predictions[0,...],axis=2)*100)
                                            imsave(image_dir+'/test_label_epoch'+str(epoch)+'_estep'
                                                   +str(eval_steps)+'_rank'+str(comm_rank)+'.png',val_model_labels[0,...]*100)
                                            imsave(image_dir+'/test_combined_epoch'+str(epoch)+'_estep'
                                                   +str(eval_steps)+'_rank'+str(comm_rank)+'.png',colormap[val_model_labels[0,...],np.argmax(val_model_predictions[0,...],axis=2)])
                                        else:
                                            np.save(image_dir+'/test_pred_epoch'+str(epoch)+'_estep'
                                                    +str(eval_steps)+'_rank'+str(comm_rank)+'.npy',np.argmax(val_model_predictions[0,...],axis=2)*100)
                                            np.save(image_dir+'/test_label_epoch'+str(epoch)+'_estep'
                                                    +str(eval_steps)+'_rank'+str(comm_rank)+'.npy',val_model_labels[0,...]*100)

                                    eval_loss += tmp_loss
                                    eval_steps += 1
                                except tf.errors.OutOfRangeError:
                                    eval_steps = np.max([eval_steps,1])
                                    eval_loss /= eval_steps
                                    if per_rank_output:
                                        print("COMPLETED: rank {}, evaluation loss for epoch {} (of {}) is {}".format(comm_rank, epoch, num_epochs, eval_loss))
                                    else:
                                        if comm_rank == 0:
                                            print("COMPLETED: evaluation loss for epoch {} (of {}) is {}".format(epoch, num_epochs, eval_loss))
                                    if per_rank_output:
                                        iou_score = sess.run(iou_op)

                                        print("COMPLETED: rank {}, evaluation IoU for epoch {} (of {}) is {}".format(comm_rank, epoch, num_epochs, iou_score))
                                    else:
                                        iou_score = sess.run(iou_avg)

                                        if comm_rank == 0:
                                            print("COMPLETED: evaluation IoU for epoch {} (of {}) is {}".format(epoch, num_epochs, iou_score))
                                    sess.run(iou_reset_op)

                                    sess.run(val_init_op, feed_dict={handle: val_handle}, options=options,
                                             run_metadata=run_metadata)

                                    update_timeline_in_range(enable_tf_timeline, run_metadata, many_runs_timeline)

                                    step_trace_fp = open("validation_step_trace_out." + str(time.time()) +
                                                         "pickle", "wb")

                                    pickle.dump(run_metadata, step_trace_fp)

                                    break
                            nvtx.RangePop() # Eval Loop

                    if enable_tf_timeline:
                        many_runs_timeline.save('Timeliner_output.json')

                    # reset counters
                    epoch += 1
                    train_loss = 0.
                    step = 0

                    nvtx.RangePop()  # Epoch
                    nvtx.RangePush("Epoch", epoch)

                    training_iteration_time_logger.end_timer()

                except tf.errors.OutOfRangeError:
                    break

            nvtx.RangePop() # Epoch
            nvtx.RangePop() # Training Loop

            training_loop_timer_logger.end_timer()

    if enable_tf_timeline:
        many_runs_timeline.save('Timeliner_output.json')
        pickle.dump(run_metadata, timeline_trace_fp)

    io_training_time_logger.end_timer()
    global_time_logger.end_timer()

if __name__ == '__main__':
    argparse_timer_logger = logger(-1, "Parse Arguments", -1, True)
    argparse_timer_logger.start_timer()

    AP = argparse.ArgumentParser()
    AP.add_argument("--lr",default=1e-4,type=float,help="Learning rate")
    AP.add_argument("--blocks",default=[3,3,4,4,7,7,10],type=int,nargs="*",help="Number of layers per block")
    AP.add_argument("--output",type=str,default='output',help="Defines the location and name of output directory")
    AP.add_argument("--chkpt",type=str,default='checkpoint',help="Defines the location and name of the checkpoint file")
    AP.add_argument("--chkpt_dir",type=str,default='checkpoint',help="Defines the location and name of the checkpoint file")
    AP.add_argument("--trn_sz",type=int,default=-1,help="How many samples do you want to use for training? A small number can be used to help debug/overfit")
    AP.add_argument("--frequencies",default=[0.982,0.00071,0.017],type=float, nargs='*',help="Frequencies per class used for reweighting")
    AP.add_argument("--loss",default="weighted",choices=["weighted","focal"],type=str, help="Which loss type to use. Supports weighted, focal [weighted]")
    AP.add_argument("--datadir",type=str,help="Path to input data")
    AP.add_argument("--fs",type=str,default="local",help="File system flag: global or local are allowed [local]")
    AP.add_argument("--optimizer",type=str,default="LARC-Adam",help="Optimizer flag: Adam, RMS, SGD are allowed. Prepend with LARC- to enable LARC [LARC-Adam]")
    AP.add_argument("--epochs",type=int,default=150,help="Number of epochs to train")
    AP.add_argument("--batch",type=int,default=1,help="Batch size")
    AP.add_argument("--use_batchnorm",action="store_true",help="Set flag to enable batchnorm")
    AP.add_argument("--dtype",type=str,default="float32",choices=["float32","float16"],help="Data type for network")
    AP.add_argument("--filter-sz",type=int,default=3,help="Convolution filter size")
    AP.add_argument("--growth",type=int,default=16,help="Channel growth rate per layer")
    AP.add_argument("--disable_training", help="Disable training for test purpose", action='store_true')
    AP.add_argument("--enable_tf_timeline", help="Enable Timeline module for tracing TF workflow", action='store_true')
    parsed = AP.parse_args()

    #play with weighting
    weights = [1./x for x in parsed.frequencies]
    weights /= np.sum(weights)

    # convert name of datatype into TF type object
    dtype=getattr(tf, parsed.dtype)

    argparse_timer_logger.end_timer()

    #invoke main function
    main(input_path=parsed.datadir,blocks=parsed.blocks,weights=weights,image_dir=parsed.output,
         checkpoint_dir=parsed.chkpt_dir, trn_sz=parsed.trn_sz, learning_rate=parsed.lr, loss_type=parsed.loss,
         fs_type=parsed.fs, opt_type=parsed.optimizer, num_epochs=parsed.epochs, batch=parsed.batch,
         batchnorm=parsed.use_batchnorm, dtype=dtype, chkpt=parsed.chkpt, filter_sz=parsed.filter_sz,
         growth=parsed.growth, disable_training=parsed.disable_training, enable_tf_timeline=parsed.enable_tf_timeline)
