#!/usr/bin/env python
# -*- coding: utf-8 -*-
import h5py
# import warnings
# with warnings.catch_warnings():
#    warnings.filterwarnings("ignore",category=FutureWarning)
#    import h5py
from data_loader import TurnPredictionDataset
from lstm_model import LSTMPredictor
from torch.nn.utils import clip_grad_norm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from copy import deepcopy

from os import mkdir
from os.path import exists
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import f1_score, roc_curve, confusion_matrix
import time as t
import pickle
import platform
from sys import argv
import json
from random import randint
import os
import distro 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import feature_vars as feat_dicts

# %% data set select
data_set_select = 0  # 0 for maptask, 1 for mahnob, 2 for switchboard
if data_set_select == 0:
    #    train_batch_size = 878
    train_batch_size = 128
    test_batch_size = 1
else:
    train_batch_size = 128
    # train_batch_size = 256
    #    train_batch_size = 830 # change this
    test_batch_size = 1

# %% Batch settings
alpha = 0.99  # smoothing constant
init_std = 0.5
momentum = 0
test_batch_size = 1  # this should stay fixed at 1 when using slow test because the batches are already set in the data loader

# sequence_length = 800
# dropout = 0
# Roddy used 3s context window; our data: 0.2-38s for BC
prediction_length = 1  # (3 seconds of prediction)
shuffle = True
num_layers = 1
onset_test_flag = True
annotations_dir = './data/extracted_annotations/voice_activity/'

proper_num_args = 2
print('Number of arguments is: ' + str(len(argv)))

if not (len(argv) == proper_num_args):
    # %% Single run settings (settings when not being called as a subprocess)
    no_subnets = True
    feature_dict_list = feat_dicts.gemaps_50ms_dict_list 

    hidden_nodes_master = 50
    hidden_nodes_acous = 50
    hidden_nodes_visual = 0
    sequence_length = 600  # (10 seconds of TBPTT)
    learning_rate = 0.01
    freeze_glove_embeddings = False
    grad_clip_bool = False # turn gradient clipping on or off
    grad_clip = 1.0 # try values between 0 and 1
    init_std = 0.5

    num_epochs = 1500
    slow_test = True
    early_stopping = True
    patience = 10
    
    l2_dict = {
        'emb': 0.0001,
        'out': 0.000001,
        'master': 0.00001,
        'acous': 0.00001,
        'visual': 0.}

    """note: for applying dropout on models with subnets the 'master_in' dropout probability is not used 
    and the dropout for the output of the appropriate modality is used."""

    dropout_dict = {
        'master_out': 0.,
        'master_in': 0, # <- this doesn't affect anything when there are subnets
        'acous_in': 0.25,
        'acous_out': 0.25,
        'visual_in': 0,
        'visual_out': 0.
    }

    results_dir = './results'
    if not(os.path.exists(results_dir)):
        os.mkdir(results_dir)
    train_list_path = './data/splits/training_AA.txt'
    test_list_path = './data/splits/testing_AA.txt'
    

    use_date_str = True
    detail = '_'
    if 'dev' in train_list_path:
        detail = 'dev' + detail
    # import feature_vars as feat_dicts

    # %% Settings

    for feat_dict in feature_dict_list:
        detail += feat_dict['short_name'] + '_'
    if no_subnets:
        detail += 'no_subnet_'

    name_append = detail + \
                  '_m_' + str(hidden_nodes_master) + \
                  '_a_' + str(hidden_nodes_acous) + \
                  '_v_' + str(hidden_nodes_visual) + \
                  '_lr_' + str(learning_rate)[2:] + \
                  '_l2e_' + str(l2_dict['emb'])[2:] + \
                  '_l2o_' + str(l2_dict['out'])[2:] + \
                  '_l2m_' + str(l2_dict['master'])[2:] + \
                  '_l2a_' + str(l2_dict['acous'])[2:] + \
                  '_l2v_' + str(l2_dict['visual'])[2:] + \
                  '_dmo_'+str(dropout_dict['master_out'])[2:] + \
                  '_dmi_'+str(dropout_dict['master_in'])[2:] + \
                  '_dao_'+str(dropout_dict['acous_out'])[2:] + \
                  '_dai_'+str(dropout_dict['acous_in'])[2:] + \
                  '_dvo_' + str(dropout_dict['visual_out'])[2:] + \
                  '_dvi_' + str(dropout_dict['visual_in'])[2:] + \
                  '_seq_' + str(sequence_length) + \
                  '_frg_' + str(str(int(freeze_glove_embeddings)))[0]
                  # '_grc_' + str(grad_clip)[2:]
    print(name_append)

else:
    # use the input arguments as parameters
    grad_clip_bool = False
    json_dict = json.loads(argv[1])
    locals().update(json_dict)
    # print features:
    feature_print_list = list()
    for feat_dict in feature_dict_list:
        for feature in feat_dict['features']:
            feature_print_list.append(feature)
    print_list = ' '.join(feature_print_list)
    print('Features being used: ' + print_list)
    print('Early stopping: ' + str(early_stopping))


lstm_settings_dict = {
    'no_subnets': no_subnets,
    'hidden_dims': {
        'master': hidden_nodes_master,
        'acous': hidden_nodes_acous,
        'visual': hidden_nodes_visual,
    },
    'uses_master_time_rate': {},
    'time_step_size': {},
    'is_irregular': {},
    'layers': num_layers,
    'dropout': dropout_dict,
}

# %% Get OS type and whether to use cuda or not
#plat = platform.linux_distribution()[0]
plat = linux_distro = distro.like()   # change the above line to this due to the python version 
my_node = platform.node()

use_cuda = torch.cuda.is_available()

print('Use CUDA: ' + str(use_cuda))

if use_cuda:
    #    torch.cuda.device(randint(0,1))
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = True
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor
    p_memory = True

# %% Data loaders (change this to adapt to the speaker space)
t1 = t.time()

# training set data loader
print('feature dict list:', feature_dict_list)
train_dataset = TurnPredictionDataset(feature_dict_list, annotations_dir, train_list_path, sequence_length,
                                      prediction_length, 'train', data_select=data_set_select)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=0,
                              drop_last=True, pin_memory=p_memory)
feature_size_dict = train_dataset.get_feature_size_dict()

if slow_test:
    # slow test loader
    test_dataset = TurnPredictionDataset(feature_dict_list, annotations_dir, test_list_path, sequence_length,
                                         prediction_length, 'test', data_select=data_set_select)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                 pin_memory=p_memory)
else:
    # quick test loader
    test_dataset = TurnPredictionDataset(feature_dict_list, annotations_dir, test_list_path, sequence_length,
                                         prediction_length, 'train', data_select=data_set_select)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=False)

lstm_settings_dict = train_dataset.get_lstm_settings_dict(lstm_settings_dict)
print('time taken to load data: ' + str(t.time() - t1))

# %% Load list of test files
test_file_list = list(pd.read_csv(test_list_path, header=None, dtype=str)[0])
train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])

# %% helper funcs
data_select_dict = {0: ['f', 'g'],
                    1: ['c1', 'c2'],
                    2: ['A', 'B']}
time_label_select_dict = {0: 'frame_time',  # gemaps
                          1: 'timestamp'}  # openface


def plot_person_error(name_list, data, results_key='barchart'):
    y_pos = np.arange(len(name_list))
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.barh(y_pos, data, align='center', alpha=0.5)
    plt.yticks(y_pos, name_list, fontsize=5)
    plt.xlabel('mean abs error per time frame', fontsize=7)
    plt.xticks(fontsize=7)
    plt.title('Individual Error')
    plt.savefig(results_dir + '/' + result_dir_name + '/' + results_key + '.pdf')


def perf_plot(results_save, results_key):
    # results_dict, dict_key
    plt.figure()
    plt.plot(results_save[results_key])
    p_max = np.round(np.max(np.array(results_save[results_key])), 4)
    p_min = np.round(np.min(np.array(results_save[results_key])), 4)
    #    p_last = np.round(results_save[results_key][-1],4)
    plt.annotate(str(p_max), (np.argmax(np.array(results_save[results_key])), p_max))
    plt.annotate(str(p_min), (np.argmin(np.array(results_save[results_key])), p_min))
    #    plt.annotate(str(p_last), (len(results_save[results_key])-1,p_last))
    plt.title(results_key + name_append, fontsize=6)
    plt.xlabel('epoch')
    plt.ylabel(results_key)
    plt.savefig(results_dir + '/' + result_dir_name + '/' + results_key + '.pdf')


# %% Loss functions
loss_func_L1 = nn.L1Loss()
loss_func_L1_no_reduce = nn.L1Loss(reduce=False)
# loss_func_MSE = nn.MSELoss()
# loss_func_MSE_no_reduce = nn.MSELoss(reduce=False)

# add the pos_weight based on the proportion
#pos_weight = torch.ones([1])
loss_func_BCE = torch.nn.BCELoss()
# add sigmoid activation internally
loss_func_BCE_Logit = nn.BCEWithLogitsLoss()



# %% Test function
def test():
    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    predicted_val = list()
    true_val = list()
    losses_mse, losses_l1 = [], []
    model.eval()
    # setup results_dict
    results_lengths = test_dataset.get_results_lengths()
    for file_name in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            # create new arrays for the results
            results_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])
            losses_dict[file_name + '/' + g_f] = np.zeros([results_lengths[file_name], prediction_length])

    for batch_indx, batch in enumerate(test_dataloader):

        model_input = []
        
        # check out this part; torch.squeeze: remove all the 1 out
        for b_i, bat in enumerate(batch):
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(torch.squeeze(bat, 0).transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(torch.squeeze(bat, 0).type(dtype)).transpose(0, 2).transpose(1, 2))
        
        
        y_test = Variable(torch.squeeze(batch[4].type(dtype), 0))
        
        
        info_test = batch[-1]
        batch_length = int(info_test['batch_size'])
        if batch_indx == 0:
            model.change_batch_size_reset_states(batch_length)
        else:
            if slow_test:
                model.change_batch_size_no_reset(batch_length)
            else:
                model.change_batch_size_reset_states(batch_length)

        out_test = model(model_input)
        out_test = torch.transpose(out_test, 0, 1)
        
        
        # convert to binary class to prepare for f-scores
        
        threshold = torch.tensor([0.5])
        predicted = ((out_test>threshold).float()*1).numpy()
        true = ((y_test.transpose(0, 1)>threshold).float()*1).numpy()
        # convert 2d to 1d array
        predicted = predicted.flatten()
        true = true.flatten()
        predicted_val.append(predicted)
        true_val.append(true)
        
        '''
        # use majority vote for the next 3 seconds
        temp = results.numpy()
        temp_lst = []
        temp = np.sum(results.numpy(), axis=2).tolist()
        for i in temp:
            for j in i:
                if j > 30:
                    temp_lst.append(0)
                else:
                    temp_lst.append(1)
            predicted_vals.append(temp_lst)
        #predicted = [1 if x > 0.5 else 0 for x in out_test]
        true_vals = list()
        #predicted_class.append(results[0])
        '''
        
        
        
        if test_dataset.set_type == 'test':
            file_name_list = [info_test['file_names'][i][0] for i in range(len(info_test['file_names']))]
            gf_name_list = [info_test['g_f'][i][0] for i in range(len(info_test['g_f']))]
            time_index_list = [info_test['time_indices'][i][0] for i in range(len(info_test['time_indices']))]
        else:
            file_name_list = info_test['file_names']
            gf_name_list = info_test['g_f']
            time_index_list = info_test['time_indices']

        # Should be able to make other loss calculations faster
        # Too many calls to transpose as well. Should clean up loss pipeline
        y_test = y_test.permute(2, 0, 1)
        loss_no_reduce = loss_func_L1_no_reduce(out_test, y_test.transpose(0, 1))
        
        
        for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                 gf_name_list,
                                                                 time_index_list,
                                                                 range(batch_length)):

            results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = out_test[
                batch_indx].data.cpu().numpy()
            losses_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = loss_no_reduce[
                batch_indx].data.cpu().numpy()
        
        loss = loss_func_BCE(F.sigmoid(out_test), y_test.transpose(0, 1))
        # loss = loss_func_BCE_Logit(out_test,y_test.transpose(0,1))
        losses_test.append(loss.data.cpu().numpy())
        batch_sizes.append(batch_length)

        loss_l1 = loss_func_L1(out_test, y_test.transpose(0, 1))
        losses_l1.append(loss_l1.data.cpu().numpy())
        
        
    # get weighted mean
    # normalize with the batch size(after sigmoid)
    loss_weighted_mean = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_test))) / np.sum(batch_sizes)
    loss_weighted_mean_l1 = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_l1))) / np.sum(batch_sizes)
    #    loss_weighted_mean_mse = np.sum( np.array(batch_sizes)*np.squeeze(np.array(losses_mse))) / np.sum( batch_sizes )
    
    # get f-score after all batches
    true_vals = [item for sublist in true_val  for item in sublist]
    predicted_vals = [item for sublist in predicted_val for item in sublist]
    f_score = f1_score(true_vals, predicted_vals, average='weighted')
    
    for conv_key in test_file_list:
        # append .g and .f results
        results_dict[conv_key + '/' + data_select_dict[data_set_select][1]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][1]]).reshape(-1, prediction_length)
        results_dict[conv_key + '/' + data_select_dict[data_set_select][0]] = np.array(
            results_dict[conv_key + '/' + data_select_dict[data_set_select][0]]).reshape(-1, prediction_length)
        
    
    # get error per person
    bar_chart_labels = []
    bar_chart_vals = []
    for conv_key in test_file_list:
        #        for g_f in ['g','f']:
        for g_f in data_select_dict[data_set_select]:
            losses_dict[conv_key + '/' + g_f] = np.array(losses_dict[conv_key + '/' + g_f]).reshape(-1,
                                                                                                    prediction_length)
            bar_chart_labels.append(conv_key + '_' + g_f)
            bar_chart_vals.append(np.mean(losses_dict[conv_key + '/' + g_f]))

    results_save['test_losses'].append(loss_weighted_mean)
    results_save['test_losses_l1'].append(loss_weighted_mean_l1)
    results_save['f_scores'].append(f_score)
    
    #    results_save['test_losses_mse'].append(loss_weighted_mean_mse)

    indiv_perf = {'bar_chart_labels': bar_chart_labels,
                  'bar_chart_vals': bar_chart_vals}
    results_save['indiv_perf'].append(indiv_perf)
    

# %% Init model
# model = LSTMPredictor(feature_size_dict, hidden_nodes, num_layers,train_batch_size,sequence_length,prediction_length,train_dataset.get_embedding_info(),dropout=dropout)
embedding_info = train_dataset.get_embedding_info()

model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, feature_size_dict=feature_size_dict,
                      batch_size=train_batch_size, seq_length=sequence_length, prediction_length=prediction_length,
                      embedding_info=embedding_info)

model.weights_init(init_std)

optimizer_list = []

optimizer_list.append( optim.Adam( model.out.parameters(), lr=learning_rate, weight_decay=l2_dict['out'] ) )
for embed_inf in embedding_info.keys():
    if embedding_info[embed_inf]:
        for embedder in embedding_info[embed_inf]:
            if embedder['embedding_use_func'] or (embedder['use_glove'] and not(lstm_settings_dict['freeze_glove'])):
                optimizer_list.append(
                    optim.Adam( model.embedding_func.parameters(), lr=learning_rate, weight_decay=l2_dict['emb'] )
                                      )

for lstm_key in model.lstm_dict.keys():
    optimizer_list.append(optim.Adam(model.lstm_dict[lstm_key].parameters(), lr=learning_rate, weight_decay=l2_dict[lstm_key]))



results_save = dict()

results_save['train_losses'], results_save['test_losses'], results_save['indiv_perf'], results_save[
    'test_losses_l1'],results_save['f_scores'] = [], [], [], [], []


# %% Training
for epoch in range(0, num_epochs):
    model.train()
    t_epoch_strt = t.time()
    loss_list = []
    model.change_batch_size_reset_states(train_batch_size)

    if onset_test_flag:
        # setup results_dict
        train_results_dict = dict()
        #            losses_dict = dict()
        train_results_lengths = train_dataset.get_results_lengths()
        for file_name in train_file_list:
            #            for g_f in ['g','f']:
            for g_f in data_select_dict[data_set_select]:
                # create new arrays for the results
                train_results_dict[file_name + '/' + g_f] = np.zeros(
                    [train_results_lengths[file_name], prediction_length])
                train_results_dict[file_name + '/' + g_f][:] = np.nan
    for batch_indx, batch in enumerate(train_dataloader):
        # b should be of form: (x,x_i,v,v_i,y,info)
        model.init_hidden()
        model.zero_grad()
        model_input = []

        model_input = []

        for b_i, bat in enumerate(batch):
            if len(bat) == 0:
                model_input.append(bat)
            elif (b_i == 1) or (b_i == 3):
                model_input.append(bat.transpose(0, 2).transpose(1, 2).numpy())
            elif (b_i == 0) or (b_i == 2):
                model_input.append(Variable(bat.type(dtype)).transpose(0, 2).transpose(1, 2))

        y = Variable(batch[4].type(dtype).transpose(0, 2).transpose(1, 2))
        info = batch[5]
        model_output_logits = model(model_input)
        
        
        
        # model_output_logits = model(model_input[0],model_input[1],model_input[2],model_input[3])

        # loss = loss_func_BCE(F.sigmoid(model_output_logits), y)
        # pay attention to the sigmoid func processing here!
        
        
        loss = loss_func_BCE_Logit(model_output_logits,y)
        loss_list.append(loss.cpu().data.numpy())
        loss.backward()
        #        optimizer.step()
        if grad_clip_bool:
            clip_grad_norm(model.parameters(), grad_clip)
        for opt in optimizer_list:
            opt.step()
        if onset_test_flag:
            file_name_list = info['file_names']
            gf_name_list = info['g_f']
            time_index_list = info['time_indices']
            train_batch_length = y.shape[1]
            # model_output = torch.transpose(model_output,0,1)
            model_output = torch.transpose(model_output_logits, 0, 1)
            for file_name, g_f_indx, time_indices, batch_indx in zip(file_name_list,
                                                                     gf_name_list,
                                                                     time_index_list,
                                                                     range(train_batch_length)):
                #                train_results_dict[file_name+'/'+g_f_indx[0]][time_indices[0]:time_indices[1]] = model_output[batch_indx].data.cpu().numpy()
                train_results_dict[file_name + '/' + g_f_indx][time_indices[0]:time_indices[1]] = model_output[
                    batch_indx].data.cpu().numpy()
    
    # stores the BCE loss func results
    results_save['train_losses'].append(np.mean(loss_list))
    # %% Test model
    t_epoch_end = t.time()
    model.eval()
    test()
    model.train()
    t_total_end = t.time()
    #        torch.save(model,)
    print(
        '{0} \t Test_loss: {1}\t Train_Loss: {2} \t FScore: {3}  \t Train_time: {4} \t Test_time: {5} \t Total_time: {6}'.format(
            epoch + 1,
            np.round(results_save['test_losses'][-1], 4),
            np.round(np.float64(np.array(loss_list).mean()), 4),
            np.around(results_save['f_scores'][-1], 4),
            np.round(t_epoch_end - t_epoch_strt, 2),
            np.round(t_total_end - t_epoch_end, 2),
            np.round(t_total_end - t_epoch_strt, 2)))
    if (epoch + 1 > patience) and \
            (np.argmin(np.round(results_save['test_losses'], 4)) < (len(results_save['test_losses']) - patience)):
        print('early stopping called at epoch: ' + str(epoch + 1))
        break

# %% Output plots and save results
if use_date_str:
    result_dir_name = t.strftime('%Y%m%d%H%M%S')[3:]
    result_dir_name = result_dir_name + name_append+'_loss_'+str(results_save['test_losses'][np.argmin(np.round(results_save['test_losses'], 4))])[2:6]
else:
    result_dir_name = name_append

if not (exists(results_dir)):
    mkdir(results_dir)

if not (exists(results_dir + '/' + result_dir_name)):
    mkdir(results_dir + '/' + result_dir_name)

results_save['learning_rate'] = learning_rate
# results_save['l2_reg'] = l2_reg
results_save['l2_master'] = l2_dict['master']
results_save['l2_acous'] = l2_dict['acous']
results_save['l2_visual'] = l2_dict['visual']

results_save['hidden_nodes_master'] = hidden_nodes_master
results_save['hidden_nodes_visual'] = hidden_nodes_visual
results_save['hidden_nodes_acous'] = hidden_nodes_acous


perf_plot(results_save, 'train_losses')
perf_plot(results_save, 'test_losses')
perf_plot(results_save, 'test_losses_l1')
perf_plot(results_save, 'f_scores') #!!! need to set this beforehand

plt.close('all')
plot_person_error(results_save['indiv_perf'][-1]['bar_chart_labels'],
                  results_save['indiv_perf'][-1]['bar_chart_vals'], 'barchart')
plt.close('all')
pickle.dump(results_save, open(results_dir + '/' + result_dir_name + '/results.p', 'wb'))
torch.save(model.state_dict(), results_dir + '/' + result_dir_name + '/model.p')
if len(argv) == proper_num_args:
    json.dump(argv[1], open(results_dir + '/' + result_dir_name + '/settings.json', 'w'), indent=4, sort_keys=True)


