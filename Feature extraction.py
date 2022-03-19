# -*- coding: utf-8 -*-
"""
Extract vocal features and prepare other features for training

@author: Jing Liu
"""
import pandas as pd
import os
import audiofile
import opensmile
import re
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment
import math
import numpy as np
from scipy.io import wavfile
import noisereduce as nr
#from data_preparation import *
import spacy

##############
#import files#
##############
data = pd.read_csv('BC_oppor.csv')
whole_data = pd.read_csv('AllData.csv')
eGeMAPS_all = pd.read_csv('eGeMAPS_all_3.csv')
visual_no_result= pd.read_csv('visual_no.csv') 
visual_dur_result= pd.read_csv('visual_dur.csv') 

# add the extracted features to the original dataset
def add_features(data,features):
    data = data.reset_index()
    features = features.reset_index()
    new_set = pd.concat([data,features], axis=1)
    new_set.drop(['index'], axis=1)
    return new_set

################
#Vocal features#
################

# audio pre-processing: reduce noise
def reduce_noise(audio_name):
    rate, data = wavfile.read(audio_name)
    reduced_noise = nr.reduce_noise(y = data, sr=rate, thresh_n_mult_nonstationary=2,stationary=False)
    # export the "clean" data
    wavfile.write(audio_name, rate, reduced_noise)
    return reduced_noise

# extract fea based on the corresponding speakers
path = 'C:/Users/Crystal/OneDrive/Desktop/converted'
folder = os.fsencode(path)
for file in os.listdir(folder):
    filename = os.fsdecode(file)
    reduce_noise(filename)   

# Feature 1: eGeMAPS feature set (esp: HNR; MFCC;energy) remember to set the sampling rate!
def extract_eGeMAPS(file,CueStart,timewindow):
    signal, sampling_rate = audiofile.read(file,duration=timewindow,offset=CueStart,always_2d=True)
    # We set up a feature extractor for functionals of a pre-defined feature set.
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.Functionals)
    result = smile.process_signal(signal,sampling_rate)
    return result

# iterate each BC response and the corresponding audio file(file_location: the audio file)
# concatenate features to original dataset
def concate_eGeMAPS(file,timewindow): 
    n = 0
    eGeMAPS_all = pd.DataFrame()
    while n < file.shape[0]:   
        # get the audio file name
        audio_name = file['filename'][n][:-15] + '.wav'
        # get the timewindow
        # if the starting point is smaller than the timewindow, than start from the beginning
        start_point = file['onset.1'][n]-timewindow
        # entry the speaker's audio folder
        folder_name = get_speaker(data['participant'][n])
        # extract the corresponding acoustic fea
        audio = './' + folder_name + '/' + audio_name
        eGeMAPS = extract_eGeMAPS(audio,start_point,timewindow)
        eGeMAPS_all = pd.concat([eGeMAPS_all, eGeMAPS])
        n += 1 
    eGeMAPS_all.to_csv('eGeMAPS_cleaned_3.csv',index=False) 
    return eGeMAPS_all

eGeMAPS_all = concate_eGeMAPS(data,3)


# Feature 2: intonation features: use ToBI structure
# transciprtion pre-processing: into the same line
# load the transcription file as the dataframe
def clean_trans(file):
    # remove speaker tag
    original_txt = pd.read_csv(file,delimiter='\:\s',on_bad_lines='skip')
    # remove blank rows
    nonempty = original_txt.dropna()
    # assign a header to the target column
    nonempty.columns = [*nonempty.columns[:-1], 'body']
    # only extract the target column
    body_lst = nonempty['body'].to_list()
    body_str = ' '.join([str(item)for item in body_lst])
    # remove the explanation/time intervals in '[]'
    cleaned = re.sub(r'(\[*\S+\])|(\(\S+\))|(\[\S+\s\S+\])|(\[\S+\s\S+\s\S+\])','', body_str)
    text_file = open('clean_'+ file, "w")
    #write string to file
    text_file.write(cleaned)
    #close file
    text_file.close()
    return cleaned

# walk the folder to get the final results
path = 'D:/course material/Master thesis/BC/data & modeling/dataset/Transcriptions'
folder = os.fsencode(path)
for file in os.listdir(folder):
    filename = os.fsdecode(file)
    clean_trans(filename)   
    
    
# naive one: calculate the slope here
# get the peak
# types: flat; rising; falling; rising-falling; falling-rising
# A threshold for pitch change: 0.2–0.3% between 250–4000 HZ and temporal info
# systematic maturational changes in auditory processing for school-age (4–10 years) children (Shafer et al., 2000)

# convert into the categories(from pitch to pitch change perception)-> perceovable
def pitchContour(voiceID):
    sound = parselmouth.Sound(voiceID) 
    pitch = call(sound, "To Pitch", 0.0,75,600)
    raw_pitch_values = pitch.selected_array['frequency']
    pitch_values = raw_pitch_values[raw_pitch_values != 0]
    # get the proper interval as the temporal threshold for pitch change perception
    # for now it is 100ms(the communicative lit, 2021)
    interval = math.ceil(len(pitch_values)/30)
    raw_result = [sum(x)/len(x) for x in (pitch_values[k:k+interval] for k in range(0,len(pitch_values),interval))]
    # To ensure to get 10 averaged pitch values
    if len(raw_result) > 30:
        redundent = len(raw_result)-30
        result = raw_result[redundent:len(raw_result)]
    elif len(raw_result) < 30:
        filler_number = len(raw_result)-30
        filler = pitch_values[filler_number]
        result = np.append(raw_result, filler)
    else:
        result = raw_result
    # get   Value =5* (log(x)-log(min))/(log(max)-log(min))
    maxPitch = np.amax(pitch_values)
    minPitch = np.amin(pitch_values)
    unified_pitch_lst = []
    for i in result:
        unified_pitch = 5* (math.log10(i)-math.log10(minPitch))/(math.log10(maxPitch)-math.log10(minPitch))
        floated_pitch = float(unified_pitch)
        unified_pitch_lst.append(floated_pitch)
    return unified_pitch_lst

def segment_audio(data):
    n = 0
    while n < data.shape[0]: 
        if (data['onset.1'][n] < 3):
            start_point = 0
        else:
            start_point = data['onset.1'][n]-3
        # get the audio file name
        audio_name = data['filename'][n][:-15] + '.wav'
        # get the timewindow
        start = start_point * 1000
        end = ['onset.1'][n] * 1000
        recording = AudioSegment.from_wav(audio_name)
        sample = recording[start:end]
        name = str(n) + '.wav'
        sample.export(name, format="wav")
        n+=1
    return sample

    
def extract_pitch_change(file,start_point,timewindow):
    sound = parselmouth.Sound(file)
    # load the normalized key timepoints
    pitch = pitchContour(sound)
    # get the peak point
    peak = max(pitch)
    # get pitch change range for two parts
    start = peak - pitch[0]
    end = peak - pitch[-1]
    # get the emerging point of the peak
    peak_index = pitch.index(peak)
    # two conditions: just ignore the starting/ending point if the duration is too short/long
    if (peak_index > 20 and start > 1):
        intonation = 'rise' 
    elif (peak_index < 10 and end > 1):
        intonation = 'fall'
    elif (peak_index > 10 and peak_index < 20 and start > 1 and end > 1):
        intonation = 'risefall'
    else:
        intonation = 'flat'
    # temporal for pitch change
    return intonation

def concate_intonation(file,timewindow): 
    n = 0
    intonation_all = []
    while n < data.shape[0]:   
        # get the audio file name
        audio_name = data['filename'][n][:-15] + '.wav'
        # get the timewindow
        if data['onset.1'][n]<timewindow:
            start_point = 0
        else:
            start_point = data['onset.1'][n]-timewindow
        segment_audio(audio_name,start_point,data['onset.1'][n])
        intonation = extract_pitch_change('trial.wav',start_point,timewindow)
        intonation_all.append(intonation)
        n += 1 
    intonation_all.to_csv('intonation_all.csv',index=False) 
    return intonation_all
    
# input: file with the starting points; audio files' path; output: dataframe/list with the intonation category
intonation_all = concate_intonation(data,3)  
    
    
#################
#Visual features#
#################
# organize the visual features
# loop each datapoint, try to get the speaker's turn
# this is for each data point
# input: info of each datapoint
def extract_visual(whole_data,BC,n): 
    result_no = []
    result_dur = []
    parti = BC['participant'][n]
    filename = BC['filename'][n]
    # potential visual cue list from the speaker
    tag_lst = ['LS','Nods','HShake','S1','S2','Laugh','Frown','Raised']
    # get the speaker's tag
    speaker = get_speaker(parti)
    # filter the desired visual data from the whole data
    for tag in tag_lst:   
        visual = whole_data.loc[((whole_data['participant'] == speaker) & (whole_data['filename'] == filename) & (whole_data['category'] == tag))]
        # get the behaviors based on the time interval
        # make sure the target behavior occurs in the given time window
        start_point = BC['onset.1'][n]-3
        end_point = BC['onset.1'][n]
        final = visual.loc[(((visual['onset.1'] >= start_point) & (visual['onset.1'] <= end_point)) | ((visual['offset.1'] >= start_point) & (visual['offset.1'] <= end_point))| ((visual['onset.1'] <= start_point) & (visual['offset.1'] >= end_point)))]    
        # calculate the number of occurence in the given time window
        number = final.shape[0]
        # calculate the duration of the behavior in the given time window
        duration = final['duration.1'].sum()
        result_no.append(number)
        result_dur.append(duration)
    df_no = pd.DataFrame(result_no).T
    df_dur = pd.DataFrame(result_dur).T
    # return a dataframe with all the listed features
    return df_no, df_dur


# loop BC dataframe
n = 0
visual_all_no = pd.DataFrame()
visual_all_dur = pd.DataFrame()
while n < data.shape[0]:   
    # get each datapoint's preceding visual cues
    visual_no, visual_dur = extract_visual(whole_data,data,n)
    visual_all_no = pd.concat([visual_all_no, visual_no])
    visual_all_dur = pd.concat([visual_all_dur, visual_dur])
    n += 1 
    
# add headers to the dataframe
visual_no_result = visual_all_no.rename(columns={0:'LS',1:'Nods',2:'HShake',3:'S1',4:'S2',5:'Laugh',6:'Frown',7:'Raised'})
visual_dur_result = visual_all_dur.rename(columns={0:'LS',1:'Nods',2:'HShake',3:'S1',4:'S2',5:'Laugh',6:'Frown',7:'Raised'})
headers =  ['LS','Nods','HShake','S1','S2','Laugh','Frown','Raised']
visual_no_result.to_csv('visual_no.csv',index=False) 
visual_dur_result.to_csv('visual_dur.csv',index=False)  
# concatenate the results    
new_set = add_features(data,eGeMAPS_all)    
final_set_no = add_features(new_set,visual_no_result)
final_set_dur = add_features(new_set,visual_dur_result)   
final_set_no.to_csv('final_set_no.csv',index=False) 
final_set_dur.to_csv('final_set_dur.csv',index=False) 


#################
#Verbal features#
#################

# POS tags
def get_POS(transcription,language):
    if language == 'English':
        nlp = spacy.load("en_core_web_lg")
        transcription = transcription[(transcription['Filename'] == 'CA-BO-IO') | (transcription['Filename'] == 'AA-BO-CM')]
    elif language == 'French':
        nlp = spacy.load("fr_core_news_lg")
        transcription = transcription[(transcription['Filename'] != 'CA-BO-IO') & (transcription['Filename'] != 'AA-BO-CM')]
    utt = transcription['Text'].tolist()
    n = 0
    info_utt = []
    transcription = transcription.values.tolist()
    transcription = pd.DataFrame(transcription, columns = ['index1','index2','Filename','UtteranceName','Speaker', 'Text','Start','End','Length'])
    new = transcription[['Filename','UtteranceName','Speaker', 'Text','Start','End','Length']]
    while n < len(utt): 
        utt_name = new['UtteranceName'][n]
        file_name = new['Filename'][n]
        doc = nlp(utt[n])
        for token in doc:
            word = token.text
            tag = token.pos_
            info_utt.append([file_name,utt_name,word,tag])
        n+=1
    utt_frame = pd.DataFrame(info_utt, columns = ['Filename','UtteranceName','Word','POS'])
    return utt_frame


def append_POS(transcription,Words,language):
    utt_frame = get_POS(transcription,language)      
    if language == 'French':
        Words = Words[(Words['Filename'] != 'CA-BO-IO') & (Words['Filename'] != 'AA-BO-CM')]
    if language == 'English':
        Words = Words[(Words['Filename'] == 'CA-BO-IO') | (Words['Filename'] == 'AA-BO-CM')]
    utt_frame['standard'] = utt_frame['UtteranceName']+utt_frame['Word']
    Words['standard'] = Words['UtteranceName']+Words['Word']  
    utt_frame_lst = utt_frame['standard'].tolist()
    # get the matching word from FA document
    result = Words.loc[Words['standard'].isin(utt_frame_lst)] 
    result1 = result[['Word', 'Start', 'End','Length','UtteranceName','Speaker', 
                     'Filename','Global_start','Global_end','standard']].values.tolist()
    # match the POS tags to the word dataframe
    suspicious = []
    POS_lst = []
    final = pd.DataFrame()
    n = 0
    while n < len(result1):
        utterance = result1[n][-1]
        index = utt_frame_lst.index(utterance)
        selected = utt_frame.iloc[[index]]
        if selected.shape[0] > 1:
            suspicious.append(selected)
        else:  
            # get POS list
            final = pd.concat([selected,final])
            new = selected.values.tolist()
            POS = new[0][-2]
            POS_lst.append(POS)
        n+=1   
    # append the POS column to the original dataset
    result['POS'] = POS_lst
    final = result[['Filename','UtteranceName','Speaker','Word','Start','End','Global_start','Global_end','Length','POS']]
    return final

Words = pd.read_csv('word.csv')
transcription = pd.read_csv('transcription.csv')
utt_frame = pd.read_csv('POS_candi.csv')


result_eng = append_POS(transcription,Words,'English')
result_eng.to_csv('result_eng.csv')
result_fr = append_POS(transcription,Words,'French')
result_eng = pd.read_csv('result_eng.csv')
result_fr = pd.read_csv('result_fr.csv')
final = pd.concat([result_fr,result_eng])
final.to_csv('POS.csv')
POS_lst = final['POS'].tolist()
#get type of POS
POS_type = list(dict.fromkeys(POS_lst))



