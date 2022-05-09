# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:07:18 2022

Prepare the data 
"""
import pandas as pd
import os
from pydub import AudioSegment
import numpy as np
import audiofile
import opensmile


###########
#BC labels#
###########

# # prepare BC labels
# BC_oppor = pd.read_csv('BC_opportunity.csv')
# BC_oppor = BC_oppor.loc[BC_oppor['opportunity'] == 'BC']
# # read each file
# path = 'D:\\course_material\\thesis\\BC\\modeling\\temporal_model\\data\\signals\\dialogues_mono'

#input: start
def seg_timeframe(start,end,interval):
    frameTimes = np.arange(start,end,interval)
    frameTimes = frameTimes.tolist()
    frameTimes_lst = []
    for i in frameTimes:
        new_ele = round(i, 2)
        frameTimes_lst.append(new_ele)
    return frameTimes_lst
    
def add_BC(df,file_candi,interval):
    n = 0 
    new = []
    while n < file_candi.shape[0]:
        # adjust the onset and offset to unify the original annotation file
        file_candi['onset.1'][n]/interval
        # onset: include preceding frame; offset: include preceding frame
        onset = file_candi['onset.1'][n] - file_candi['onset.1'][n] % interval       
        offset = file_candi['offset.1'][n] + (interval - file_candi['offset.1'][n] % interval)     
        frameTimes_new_lst = seg_timeframe(onset,offset,interval)
        new.append(frameTimes_new_lst)
        frame_BC = [item for sublist in new for item in sublist]
        n +=1    
    # match the BC part to the original annotation
    n = 0
    while n < df['frameTimes'].shape[0]:
        for time in frame_BC:
            if df['frameTimes'][n] == time:
               df['val'][n] = 1
        n += 1
    return df    
    
# input: the annotation file with all BC; the folder path with all the audios
# output: the BC label file 
def get_BC(BC_oppor,path,interval):
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        audio = AudioSegment.from_file(filename)
        audio_length = audio.duration_seconds
        # create the dataframe
        frameTimes_lst = seg_timeframe(0, audio_length ,interval)
        df = pd.DataFrame(frameTimes_lst, columns = ['frameTimes'])
        # set 0 as the proto file
        df['val'] = 0
        Name = filename[:-4] + '.csv'
        if Name.split('.')[1] == 'f':
            # find the match BC in the file
            file_candi = BC_oppor.loc[(BC_oppor['fileindex'] == Name.split('.')[0]) & ((BC_oppor['participant'] == 'Adult1')|(BC_oppor['participant'] == 'Parent'))]
            file_candi = file_candi.reset_index()
            result = add_BC(df,file_candi,interval)
            
        elif Name.split('.')[1] == 'g':
            # find the match BC in the file
            file_candi = BC_oppor.loc[(BC_oppor['fileindex'] == Name.split('.')[0]) & ((BC_oppor['participant'] == 'Adult2')|(BC_oppor['participant'] == 'Child'))]
            file_candi = file_candi
            result = add_BC(df,file_candi,interval)
        else:
            print(Name)
        result.to_csv(Name,index=False)
    return result


def get_proto(BC_oppor,path,interval):

    folder = os.fsencode(path)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        audio = AudioSegment.from_file(filename)
        audio_length = audio.duration_seconds
        # create the dataframe
        frameTimes_lst = seg_timeframe(0, audio_length ,interval)
        df = pd.DataFrame(frameTimes_lst, columns = ['frameTimes'])
        df['val'] = 0
        Name = filename[:-4] + '.csv'
        if Name.split('.')[1] == 'f':
            # find the match BC in the file(participants are listeners)
            file_candi = BC_oppor.loc[(BC_oppor['fileindex'] == Name.split('.')[0]) & ((BC_oppor['participant'] == 'Adult1')|(BC_oppor['participant'] == 'Parent'))]
            file_candi = file_candi.reset_index()
            #result = add_BC(df,file_candi,interval)
            n = 0 
            new = []
            while n < file_candi.shape[0]:
                # adjust the onset and offset to unify the original annotation file
                file_candi['onset.1'][n]/interval
                # onset: include preceding frame; offset: include preceding frame
                onset = file_candi['onset.1'][n] - file_candi['onset.1'][n] % interval       
                offset = file_candi['offset.1'][n] + (interval - file_candi['offset.1'][n] % interval)     
                frameTimes_new_lst = seg_timeframe(onset,offset,interval)
                new.append(frameTimes_new_lst)
                frame_BC = [item for sublist in new for item in sublist]
                n +=1    
            # match the BC part to the original annotation
            n = 0
            while n < df['frameTimes'].shape[0]:
                for time in frame_BC:
                    if df['frameTimes'][n] == time:
                       df['val'][n] = 1
                n += 1
        elif Name.split('.')[1] == 'g':
            # find the match BC in the file
            file_candi = BC_oppor.loc[(BC_oppor['fileindex'] == Name.split('.')[0]) & ((BC_oppor['participant'] == 'Adult2')|(BC_oppor['participant'] == 'Child'))]
            file_candi = file_candi.reset_index()
            #result = add_BC(df,file_candi,interval)
            n = 0 
            new = []
            while n < file_candi.shape[0]:
                # adjust the onset and offset to unify the original annotation file
                file_candi['onset.1'][n]/interval
                # onset: include preceding frame; offset: include preceding frame
                onset = file_candi['onset.1'][n] - file_candi['onset.1'][n] % interval       
                offset = file_candi['offset.1'][n] + (interval - file_candi['offset.1'][n] % interval)     
                frameTimes_new_lst = seg_timeframe(onset,offset,interval)
                new.append(frameTimes_new_lst)
                frame_BC = [item for sublist in new for item in sublist]
                n +=1    
            # match the BC part to the original annotation
            n = 0
            while n < df['frameTimes'].shape[0]:
                for time in frame_BC:
                    if df['frameTimes'][n] == time:
                       df['val'][n] = 1
                n += 1
        else:
            print(Name)
        df.to_csv(Name,index=False)
        print('finish extracted' + Name)
    return df
# get_BC(BC_oppor,path,0.05)


#################
#Visual features#
#################

# get visual features based on manual annotation(sampling rate: 10ms; 50ms)
# add visual features to the one-hot embeddings
# input: dataframe; file candidate; time frame length; a list of extracted features
def add_fea(df,file_candi,interval,tag):    
    n = 0 
    new = []
    while n < file_candi.shape[0]:
        # adjust the onset and offset to unify the original annotation file
        file_candi['onset.1'][n]/interval
        # onset: include preceding frame; offset: include preceding frame
        onset = round(file_candi['onset.1'][n] - file_candi['onset.1'][n] % interval,2)
        offset = round(file_candi['offset.1'][n] + (interval - file_candi['offset.1'][n] % interval))     
        new.append([onset,offset])
        n +=1    
    # match the BC part to the original annotation
    for pair in new:
        df.loc[((df['frameTimes'] >= pair[0]) & (df['frameTimes'] <= pair[1])),tag] = 1
    return df
    

# input: the annotation file with all BC; the folder path with all the audios
# output: the BC label file 
def get_fea(All_visual,path,interval,tag_lst):
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        Name = filename[:-4] + '.csv'
        audio = AudioSegment.from_file(path + '/' + filename)
        audio_length = audio.duration_seconds
        # create the dataframe
        frameTimes_lst = seg_timeframe(0, audio_length ,interval)
        df = pd.DataFrame(frameTimes_lst, columns = ['frameTimes'])
        # set 0 as the proto file
        for tag in tag_lst:
            df[tag] = 0
        try:   
            if Name.split('.')[1] == 'f':
                # find the match features in the file
                for tag in tag_lst:
                    # this only helps to select the target file
                    file_candi = All_visual.loc[(All_visual['category'] == tag)&(All_visual['fileindex'] == Name.split('.')[0]) & ((All_visual['participant'] == 'Adult1')|(All_visual['participant'] == 'Parent'))]
                    file_candi = file_candi.reset_index() 
                    result = add_fea(df,file_candi,interval,tag)
                    
            elif Name.split('.')[1] == 'g':
                # find the match features in the file
                for tag in tag_lst:
                    # this only helps to select the target file
                    file_candi = All_visual.loc[(All_visual['category'] == tag)&(All_visual['fileindex'] == Name.split('.')[0]) & ((All_visual['participant'] == 'Adult2')|(All_visual['participant'] == 'Child'))]
                    file_candi = file_candi.reset_index() 
                    result = add_fea(df,file_candi,interval,tag)
            else:
                print(Name)
            
            # cluster nod and smile features
            result['nod'] = result['Nod'] + result['NodR'] + result['NodF']
            result['Smile'] = result['S1'] + result['S2'] 
            nod_lst = []
            for i in result['nod'].tolist():
                if i == 0:
                    nod_lst.append(0)
                else:
                    nod_lst.append(1)
            smile_lst = []
            for i in result['Smile'].tolist():
                if i == 0:
                    smile_lst.append(0)
                else:
                    smile_lst.append(1)
            result['Nod'] = nod_lst
            result['Smile'] = smile_lst
            final = result.drop(['nod', 'NodR','NodF','S1','S2'], axis=1)
            
            temp = './data/extracted_annotations/visual/'
            file_path = temp + str(interval) + '/' + Name
            final.to_csv(file_path,index=False)
        
        except:   
            print(Name)
    return final

#read file 
All_visual = pd.read_csv('All_visual.csv')
tag_lst = ['LS','Nod','NodR','NodF','HShake','S1','S2','Laugh','Frown','Raised','Forward','Backward']
path = './data/signals/dialogues_mono'
get_fea(All_visual,path,0.05,tag_lst)
get_fea(All_visual,path,0.01,tag_lst)

#################
#Verbal features#
#################
# sampling rates: 50ms(About 4133 out of 40750 words are not longer than 50ms)
# asynchronous(these are the same as Roddy,2018)

def add_verbal(df,file_candi,interval,tag):    
    n = 0 
    new = []
    while n < file_candi.shape[0]:
        # adjust the onset and offset to unify the original annotation file
        file_candi['Global_start'][n]/interval
        # onset: include preceding frame; offset: include preceding frame
        onset = round(file_candi['Global_start'][n] - file_candi['Global_start'][n] % interval,2)
        offset = round(file_candi['Global_end'][n] + (interval - file_candi['Global_end'][n] % interval))     
        new.append([onset,offset])
        n +=1    
    # match the BC part to the original annotation
    for pair in new:
        df.loc[((df['frameTimes'] >= pair[0]) & (df['frameTimes'] <= pair[1])),tag] = 1
    return df

def get_verbal(All_visual,interval):
    path = './data/signals/dialogues_mono'
    tag_lst = list(dict.fromkeys(All_visual['POS'])) 
    
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        Name = filename[:-4] + '.csv'
        audio = AudioSegment.from_file(path + '/' + filename)
        audio_length = audio.duration_seconds
        # create the dataframe
        frameTimes_lst = seg_timeframe(0, audio_length ,interval)
        df = pd.DataFrame(frameTimes_lst, columns = ['frameTimes'])
        temp = './data/extracted_annotations/verbal/'
        file_path = temp + str(interval) + '/' + Name
        # set 0 as the proto file
        for tag in tag_lst:
            df[tag] = 0
        try:   
            if Name.split('.')[1] == 'f':
                
                # find the match features in the file
                for tag in tag_lst:
                    # this only helps to select the target file
                    file_candi = All_visual.loc[(All_visual['POS'] == tag)&(All_visual['Filename'] == Name.split('.')[0]) & ((All_visual['Speaker'] == 'Adult1')|(All_visual['Speaker'] == 'Parent'))]
                    file_candi = file_candi.reset_index() 
                    try:
                        result = add_verbal(df,file_candi,interval,tag)
                        result.to_csv(file_path,index=False)
                    except:
                        pass
            
                    
            elif Name.split('.')[1] == 'g':
                
                # find the match features in the file
                for tag in tag_lst:
                    
                    # this only helps to select the target file
                    file_candi = All_visual.loc[(All_visual['POS'] == tag)&(All_visual['Filename'] == Name.split('.')[0]) & ((All_visual['Speaker'] == 'Adult2')|(All_visual['Speaker'] == 'Child'))]
                    file_candi = file_candi.reset_index() 
                    try:
                        result = add_verbal(df,file_candi,interval,tag)
                        result.to_csv(file_path,index=False)  
                    except:
                        pass
             
        except:   
            print(Name)
    return result

# asynchronous POS features
# put the file directory under the temporal_model folder
def get_async(marker):
    filename_lst = list(dict.fromkeys(All_visual['Filename'])) 
    tag_lst = list(dict.fromkeys(All_visual['POS'])) 
    for filename in filename_lst:
        
        if marker == 'f':
            file_candi_f = All_visual.loc[(All_visual['Filename'] == Name.split('.')[0]) & ((All_visual['Speaker'] == 'Adult1')|(All_visual['Speaker'] == 'Parent'))]
        else:
            file_candi_f = All_visual.loc[(All_visual['Filename'] == Name.split('.')[0]) & ((All_visual['Speaker'] == 'Adult2')|(All_visual['Speaker'] == 'Child'))]
        # sort the file based on start time
        file_candi_f = file_candi_f[['Global_start','POS']]
        file_sorted_f = file_candi_f.sort_values(by=['Global_start'])
        # convert label into one-hot encodings
        # add the proto zeros
        for tag in tag_lst:
            file_sorted_f[tag] = 0
        # convert the label into 1
        for tag in tag_lst:
            # loop the dataframe candidate
            file_sorted_f.loc[(file_sorted_f['POS']==tag),tag] = 1    
        final = file_sorted_f.drop(['POS'], axis=1)
        
        # add the start/end row if necessary
        # get audio duration
        audio = AudioSegment.from_file('./data/signals/dialogues_mono/' + filename + '.' + marker + '.wav')
        audio_length = audio.duration_seconds
        
        if final.iloc[:1]['Global_start'].tolist()[0] > 0:
            d = pd.DataFrame(0, index=np.arange(1), columns=final.columns)
            final = pd.concat([d,final])
        
        if final.iloc[-1:]['Global_start'].tolist()[0] < audio_length:
            end = pd.DataFrame(0, index=np.arange(1), columns=final.columns)
            end['Global_start'] = audio_length
            final = pd.concat([final,end])   
        
        # output the csv file
        final.to_csv('./data/extracted_annotations/verbal/async/' + filename + '.' + marker + '.csv')
    return final


# check the length of each conversation
transcription = pd.read_csv('transcription.csv')
def describe_transcription(phonemes,aspect): 
    # inspect the dataset
    # types and numbers of phonemes
    phoneme_content = phonemes[aspect]
    phoneme_type = list(dict.fromkeys(phoneme_content))
    length = []
    for i in phoneme_type:
        specific = phonemes[phonemes[aspect] == i]        
        info = specific['Length'].sum()
        result = [i,info]
        length.append(result)
    return length

results = describe_transcription(transcription,'Filename')


################
#Vocal features#
################

# go for the gemaps dataset

# input: filepath; sampling rate
def extract_eGeMAPS(file,CueStart,timewindow):
    signal, sampling_rate = audiofile.read(file,duration=timewindow,offset=CueStart,always_2d=True)
    # We set up a feature extractor for functionals of a pre-defined feature set.
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b,feature_level=opensmile.FeatureLevel.Functionals)
    result = smile.process_signal(signal,sampling_rate)
    return result

def concat_eGeMAPS(sample_rate):
    
    if sample_rate == 0.05:
        
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
    timepoint_file = pd.read_csv('./data/signals/gemaps_features_50ms/AA-AN-DL.f.csv') 
    file = './data/signals/dialogues_mono/AA-AN-DL.f.wav'
    eGeMAPS_all = pd.DataFrame()
    for start in timepoint_file['frameTime'].tolist():
        eGeMAPS = extract_eGeMAPS(file,start,0.05)
        eGeMAPS_all = pd.concat([eGeMAPS_all, eGeMAPS])

# problem here: the segment is so short that they are filled with NANs    
timepoint_file = pd.read_csv('./data/signals/gemaps_features_50ms/AA-AN-DL.f.csv') 
file = './data/signals/dialogues_mono/AA-AN-DL.f.wav'
eGeMAPS_all = pd.DataFrame()
for start in timepoint_file['frameTime'].tolist():
    eGeMAPS = extract_eGeMAPS(file,start,0.05)
    eGeMAPS_all = pd.concat([eGeMAPS_all, eGeMAPS])




