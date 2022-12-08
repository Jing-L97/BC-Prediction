# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:07:18 2022

Prepare the data 
"""
import pandas as pd
import os
from pydub import AudioSegment
import numpy as np
#import audiofile
#import opensmile
import pickle
encoding = 'utf-8'

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
            #file_candi = file_candi
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


# !!! Note: the features are from the speaker!
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
            if Name.split('.')[1] == 'g':
                # find the match features in the file
                for tag in tag_lst:
                    # this only helps to select the target file
                    file_candi = All_visual.loc[(All_visual['category'] == tag)&(All_visual['fileindex'] == Name.split('.')[0]) & ((All_visual['participant'] == 'Adult1')|(All_visual['participant'] == 'Parent'))]
                    file_candi = file_candi.reset_index() 
                    result = add_fea(df,file_candi,interval,tag)
                    
            elif Name.split('.')[1] == 'f':
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

# get POS tags
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
            if Name.split('.')[1] == 'g':
                
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
                    
            elif Name.split('.')[1] == 'f':
                
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

# surprisal value


interval = 0.05
All_visual = pd.read_csv('Embedding.csv')
path = './data/extracted_annotations/frame/'


folder = os.fsencode(path)
for file in os.listdir(folder):
    filename = os.fsdecode(file)
    Name = filename[:-4] + '.csv'
    df = pd.read_csv(path + filename)
    temp = './data/extracted_annotations/embedding/'
    #file_path = temp + str(interval) + '/' + Name
    file_path = temp + Name
    
    if Name.split('.')[1] == 'f':
                trial = pd.DataFrame()
                file_candi = All_visual.loc[(All_visual['Filename'] == Name.split('.')[0]) & ((All_visual['Speaker'] == 'Adult1')|(All_visual['Speaker'] == 'Parent'))]
                file_candi = file_candi.reset_index() 
                n = 0 
                
                while n < file_candi.shape[0]:
                    # adjust the onset and offset to unify the original annotation file
                    file_candi['Global_start'][n]/interval
                    # onset: include preceding frame; offset: include preceding frame
                    onset = round(file_candi['Global_start'][n] - file_candi['Global_start'][n] % interval,2)
                    offset = round(file_candi['Global_end'][n] + (interval - file_candi['Global_end'][n] % interval))     
                    selected = df.loc[((df['frameTimes'] >= onset) & (df['frameTimes'] <= offset))]
                    selected['Embedding'] = file_candi['Embedding'][n]
                    
                    trial = pd.concat([trial,selected])
                    n +=1 
                # remove duplicated time frames
                # try:
                #     trial.drop_duplicates(subset='frameTime', keep="last")
                # except:
                #     pass
                # add 0 for the missing values
                surprisal_lst = []
                m = 0
                while m < df.shape[0]:
                    try:
                        G_Surprisal = trial.loc[trial['frameTimes'] == df['frameTimes'][m],'Embedding'].item()
                    except:
                        G_Surprisal = 0
                    surprisal_lst.append(G_Surprisal)
                    m += 1
                #only preserve the surprisal column
                df['Embedding'] = surprisal_lst
                df1 = df[['frameTimes', 'Embedding']]
                df1.to_csv(file_path,index=False)
                
    
    
 
        elif Name.split('.')[1] == 'g':
                trial = pd.DataFrame()
                file_candi = All_visual.loc[(All_visual['Filename'] == Name.split('.')[0]) & ((All_visual['Speaker'] == 'Adult1')|(All_visual['Speaker'] == 'Parent'))]
                file_candi = file_candi.reset_index() 
                
                try:
                    
                    n = 0 
                    
                    while n < file_candi.shape[0]:
                        # adjust the onset and offset to unify the original annotation file
                        file_candi['Global_start'][n]/interval
                        # onset: include preceding frame; offset: include preceding frame
                        onset = round(file_candi['Global_start'][n] - file_candi['Global_start'][n] % interval,2)
                        offset = round(file_candi['Global_end'][n] + (interval - file_candi['Global_end'][n] % interval))     
                        selected = df.loc[((df['frameTimes'] >= onset) & (df['frameTimes'] <= offset))]
                        selected['G_Surprisal'] = file_candi['Surprisal'][n]
                        
                        trial = pd.concat([trial,selected])
                        n +=1 
                    # remove duplicated time frames
                    trial.drop_duplicates(subset='frameTime', keep="last")
                    # add 0 for the missing values
                    surprisal_lst = []
                    m = 0
                    while m < df.shape[0]:
                        try:
                            G_Surprisal = trial.loc[trial['frameTimes'] == df['frameTimes'][m],'G_Surprisal']
                        except:
                            G_Surprisal = 0
                        surprisal_lst.append(G_Surprisal)
                        m += 1
                    #only preserve the surprisal column
                    df['G_Surprisal'] = surprisal_lst
                    df1 = df[['frameTimes', 'G_Surprisal']]
                    df1.to_csv(file_path,index=False)
                except:
                    print(filename)          
                
        elif Name.split('.')[1] == 'f':
                trial = pd.DataFrame()
                # this only helps to select the target file
                file_candi = All_visual.loc[(All_visual['Filename'] == Name.split('.')[0]) & ((All_visual['Speaker'] == 'Adult2')|(All_visual['Speaker'] == 'Child'))]
                file_candi = file_candi.reset_index() 
                try:
                    
                    n = 0 
                    new = []
                    while n < file_candi.shape[0]:
                        # adjust the onset and offset to unify the original annotation file
                        file_candi['Global_start'][n]/interval
                        # onset: include preceding frame; offset: include preceding frame
                        onset = round(file_candi['Global_start'][n] - file_candi['Global_start'][n] % interval,2)
                        offset = round(file_candi['Global_end'][n] + (interval - file_candi['Global_end'][n] % interval))     
                        selected = df.loc[((df['frameTimes'] >= onset) & (df['frameTimes'] <= offset))]
                        selected['G_Surprisal'] = file_candi['Surprisal'][n]
                        trial = pd.concat([trial,selected])
                        n +=1    
                    
                    # remove duplicated time frames
                    trial.drop_duplicates(subset='frameTime', keep="last")
                    # add 0 for the missing values
                    surprisal_lst = []
                    m = 0
                    while m < df.shape[0]:
                        try:
                            G_Surprisal = trial.loc[trial['frameTimes'] == df['frameTimes'][m],'G_Surprisal']
                        except:
                            G_Surprisal = 0
                        surprisal_lst.append(G_Surprisal)
                        m += 1
                    #only preserve the surprisal column
                    df['G_Surprisal'] = surprisal_lst
                    df1 = df[['frameTimes', 'G_Surprisal']]
                    df1.to_csv(file_path,index=False)
                except:
                    print(filename)              
         
    except:   
        print(Name)



def get_embedding(All_visual,interval):
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
        temp = './data/extracted_annotations/embedding/'
        file_path = temp + str(interval) + '/' + Name
        # set 0 as the proto file
        for tag in tag_lst:
            df[tag] = 0
        try:   
            if Name.split('.')[1] == 'g':
                
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
                    
            elif Name.split('.')[1] == 'f':
                
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



# change the embedding settings
embedding = pd.read_csv('Embedding.csv')
embedding_candi = embedding['Embedding'].tolist()
final = []
n = 0 
while n < len(embedding_candi): 
    each = []
    for i in embedding_candi:
        try:
            num = float(i)
            each.append(num)
        except:
            each.append(0)
    final.append(each)
    n += 1





# asynchronous POS features
# put the file directory under the temporal_model folder
def get_async(marker):
    filename_lst = list(dict.fromkeys(All_visual['Filename'])) 
    tag_lst = list(dict.fromkeys(All_visual['POS'])) 
    for filename in filename_lst:
        
        if marker == 'g':
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

# DialoGPT embeddings
open_file = open(file_name, "rb")
gpt_embed_table = pickle.load(
    open('word embedding.pickle', 'rb'))
vector_lst = []
for i in gpt_embed_table:
    try: 
        vector = i.tolist()        
    except:
        vector = 'invalid'
    vector_lst.append(vector)

word = pd.read_csv('word.csv')
word['Embedding'] = vector_lst
# remove the rows with invalid vectors
word_final = word[word.Embedding != 'invalid']
word_final.to_csv('Embedding.csv')


def match_phase():
    # match the time stages with the transcription
    transcription = pd.read_csv('transcription.csv')
    # read the transcription 
    result = []
    annot_folder = os.fsencode('./phases/AA')
    phases = pd.DataFrame()
    for file in os.listdir(annot_folder):
        name = file.decode(encoding)
        # select the corresponding file
        selected_file = transcription.loc[transcription['Filename']==name[:-4]]
        
        # sort the selected_file by start points
        selected_file_sorted = selected_file.sort_values(by=['Start'])
        selected_file_reindexed = selected_file_sorted.reset_index()
        phase_lst = []
        # map the corresponding stages with the original file
        # chunk the stage column of the annotated one
        phase = pd.read_csv('./phases/AA/' + name)
        phase_type = list(dict.fromkeys(phase['Phases']))
        for i in phase_type:
            selected_phase = phase.loc[phase['Phases']==i]
            start = selected_file_reindexed.loc[selected_file_reindexed['UtteranceName']==selected_phase[:1]['UtteranceName'].item()].index.tolist()[0]
            end = selected_file_reindexed.loc[selected_file_reindexed['UtteranceName']==selected_phase[-1:]['UtteranceName'].item()].index.tolist()[0]
            interval = end - start + 1
            stage = [i] * interval
            phase_lst.append(stage)
        # flatten the list
        phase_lst = [item for sublist in phase_lst for item in sublist]
        if len(phase_lst) < selected_file_reindexed.shape[0]:
            difference = selected_file_reindexed.shape[0] - len(phase_lst)
            added = ['POSTGAME'] * difference
            for i in added:
                phase_lst.append(i)
        elif len(phase_lst) > selected_file_reindexed.shape[0]:
            phase_lst = phase_lst[:selected_file_reindexed.shape[0]]
        selected_file_reindexed['Stage'] = phase_lst
        title = './surprisal1/' + name
        selected_file_reindexed.to_csv(title)
        
        
        try:
            selected_file_reindexed['Stage'] = phase_lst
            title = './surprisal1/' + name
            selected_file_reindexed.to_csv(title)
        except:
            print(name)
        
    
match_phase()

# read the file content
annot_folder = os.fsencode('./annot/CA')
phases = pd.DataFrame()
name_lst = []
for file in os.listdir(annot_folder):
    name = file.decode(encoding,'ignore')
    name_lst.append(name)

stage_lst = ['PREGAME','GAME','POSTGAME']
child = pd.DataFrame()
annotation = pd.DataFrame()
for name in name_lst:   
    child_temp = pd.read_csv('./annot/CA/'+ name,encoding = "ISO-8859-1")
    n = 0
    role_lst = []
    while n < child_temp.shape[0]:
        if child_temp['Roles'][n] == '0':
            if (child_temp['Speaker'][n] == 'Adult1') or (child_temp['Speaker'][n] == 'Parent'):
                role_lst.append('I')
            else:
                role_lst.append('R')
        elif child_temp['Roles'][n] == 'SPEAKER1':
            if (child_temp['Speaker'][n] == 'Adult1') or (child_temp['Speaker'][n] == 'Parent'):
                role_lst.append('R')
            else:
                role_lst.append('I')
                
        elif child_temp['Roles'][n] == 'SPEAKER2':
            if (child_temp['Speaker'][n] == 'Adult1') or (child_temp['Speaker'][n] == 'Parent'):
                role_lst.append('I')
            else:
                role_lst.append('R')
        else:
            role_lst.append('invalid')
            
        n += 1
    child_temp['Floor'] = role_lst
    child = pd.concat([child,child_temp])            

Adult = child
Annotated = pd.concat([child,Adult])   

# append the speaker role to the entropy file
n = 0
floor_lst = []
while n < local_entro.shape[0]:
    utt_name = local_entro['UtteranceName'][n]
    try:
        floor = Annotated.loc[Annotated['UtteranceName']==utt_name,'Floor'].tolist()[0]
        floor_lst.append(floor)
    except:
        floor_lst.append('R')
        print(utt_name)
    n += 1
local_entro['Floor'] = floor_lst    
initiator = local_entro.loc[local_entro['Floor']== 'I']
responder = local_entro.loc[local_entro['Floor']== 'R']

local_entro.to_csv('Local.csv')
global_entro.to_csv('Global.csv')
    # for stage in stage_lst: 
    #     stage_candi = child_temp.loc[child_temp['Stage']==stage]
    #     position_lst = list(range(1,len(stage_candi['Stage'].tolist()) + 1))
    #     stage_candi['Position'] = position_lst
    #     #sort the file
    #     child = pd.concat([child,stage_candi])
    
# add the position list
child.to_csv('Local_stage.csv')

annot_lst = annotation['Stage'].tolist()
child['Stage'] = annot_lst

# resample the file
data = pd.read_csv('Verbal.csv')
# balance the frame data
name_type = list(dict.fromkeys(data['filename']))
nonBC_final = pd.DataFrame()
for name in name_type:
    BC_no = data.loc[(data['filename']==name) & (data['class']=='BC')].shape[0]
    nonBC = data.loc[(data['filename']==name) & (data['class']=='nonBC')].head(BC_no)
    nonBC_final = pd.concat([nonBC_final,nonBC])
BC = data.loc[data['class']=='BC']
final = pd.concat([nonBC_final,BC])
final.to_csv('Verbal1.csv')

################
#Vocal features#
################r
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



