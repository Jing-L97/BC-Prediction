# -*- coding: utf-8 -*-
"""
Automatic segmentation of audio files(based on startpoint of each utterance)

@author: Jing Liu
"""

import re
from pydub import AudioSegment
import pandas as pd

# segment audio file based on transcription(for improvement: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/326)
# input: raw transcriptions with events(in the list form); output: a cleaned transcription with texts(in the list form)
def remove_event(raw):
    body_lst = []
    for i in raw: 
        # clean other annotations
        cleaned = re.sub(r'(\[\S+\s\S+\])|(\[\S+\])|(\[\S+\s\S+\s\S+\])','', i)
        body_lst.append(cleaned)
    return body_lst

# convert the hh:mm:ss time into miliseconds; generate the endtime based on the start of the next utterance
# input: start time list;length of the  output: start time list; endtime list 
def convert_time(start_candi,audio_length):
    time_lst = []
    for i in start_candi: 
        time_str = i[1:-1]
        # convert hh:mm:ss into milliseconds
        h,m,s = time_str.split(':')
        millisecond = (int(m) * 60 + int(float(s)))*1000
        time_lst.append(millisecond)
    # add the end points as the beginning of the next utterance
    end_lst = time_lst[1:]
    end_lst.append(audio_length)
    return time_lst,end_lst

# Version 1: For the starting point and text in the same line 
def extract_time_v1(transcription):
    audio_name = transcription[:-4] + '.wav'  
    audio = AudioSegment.from_file(audio_name)
    audio_length = audio.duration_seconds
    original_txt = pd.read_csv(transcription,delimiter='\:\s',header=None, on_bad_lines='skip')
    # remove blank rows
    nonempty = original_txt.dropna()
    # split the column into speaker label and start points
    nonempty[['Start_candi', 'body']] = nonempty[1].str.split(' ', 1, expand=True)
    # speaker label
    start_candi = nonempty['Start_candi'].to_list()
    time_lst,end_lst = convert_time(start_candi,audio_length)
    nonempty['Start'] = time_lst
    # add the end points as the beginning of the next utterance
    end_lst = time_lst[1:]
    end_lst.append(audio_length)
    nonempty['End'] = end_lst
    body_candi = nonempty[1].to_list()
    body_lst = remove_event(body_candi)
    # Remove the redundant columns
    nonempty.drop('Start_candi', inplace=True, axis=1)
    nonempty.drop(1, inplace=True, axis=1)
    # print out the transcription files
    n = 0
    while n < len(body_lst):
       speaker = nonempty[0][n]
       name = str(n) + '_' + speaker + '_' + transcription 
       text_file = open(name, "w", encoding='utf-8')
       content = str(nonempty['body'][n])
       text_file.write(content)
       text_file.close()    
       n += 1
    return nonempty


# Version 2: For the starting point and text in the same line 
def extract_time_v2(transcription):
    original_txt = pd.read_csv(transcription,delimiter='\:\s',header=None, on_bad_lines='skip')
    # convert the index into column
    reindex = original_txt.reset_index()
    # reset the column headers
    rename = reindex.rename({'index': 'speaker', 0: 'body'}, axis=1)
    time = rename[rename['body'].isna()]
    # extract the start timepoints
    start_candi = time['speaker'].to_list()
    nonempty = rename.dropna()
    time_lst,end_lst = convert_time(start_candi,15)
    nonempty = nonempty.copy()
    nonempty['Start'] = time_lst
    nonempty['End'] = end_lst
    body_candi = nonempty['body'].to_list()
    body_lst = remove_event(body_candi)
    nonempty['body_cleaned'] = body_lst
    # remove the non-existing transcription(problem: empty strings are not removed)
    nonempty = nonempty.dropna()
    # Remove the redundant columns
    nonempty.drop('body', inplace=True, axis=1)
    # print out the transcription files
    Speaker_lst = nonempty['speaker'].to_list()
    n = 0
    while n < len(body_lst):
        speaker = Speaker_lst[n]
        name = str(n) + '_' + speaker + '_' + transcription
        text_file = open(name, "w", encoding='utf-8')
        nonempty = nonempty.copy()
        content = body_lst[n]
        text_file.write(content)
        text_file.close()    
        n += 1
    return nonempty

def segment_audio(transcription):
    # get the audio file name
    audio_name = transcription[:-4] + '.wav'  
    data = extract_time_v2(transcription)
    n = 0
    Start_lst = data['Start'].to_list()
    End_lst = data['End'].to_list()
    Speaker_lst = data['speaker'].to_list()
    # loop the dataframe
    while n < data.shape[0]: 
        start = Start_lst[n]
        end = End_lst[n]
        recording = AudioSegment.from_wav(audio_name)
        segment = recording[start:end]
        speaker = Speaker_lst[n]
        name = str(n) + '_' + speaker + '_' + transcription[:-4] + '.wav'
        segment.export(name, format="wav")
        n+=1
    return segment


def main():
    segment_audio('AA-MG-CH.txt')

if __name__ == "__main__":
    main()

