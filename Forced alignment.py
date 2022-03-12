# -*- coding: utf-8 -*-
"""
Automatic segmentation of audio files
Conduct the MAUS forced aligner(https://www.bas.uni-muenchen.de/Bas/BasMAUS.html)
The FA part is inspired by @WilliamNHavard's paper on sentence-aligned Spoken utterances 

input: the folder path containing orthographic transcription and the corresponding audios 
output: the textgrids generated by MAUS

@author: Jing Liu
"""

import re
from pydub import AudioSegment
import pandas as pd
import os
import os.path as osp
import requests
from lxml import etree

# Two different formats of timepoints
# Format v1: timepoints are in different lines from content
def read_transcription_one(transcription):
    original_txt = pd.read_csv(transcription,delimiter='(\:\s)|(\]\s)',header=None, on_bad_lines='skip')
    # remove unnecessary lines
    original_txt.drop(1, inplace=True, axis=1)
    original_txt.drop(2, inplace=True, axis=1)
    original_txt.drop(4, inplace=True, axis=1)
    original_txt.drop(5, inplace=True, axis=1)
    # reset the column headers
    rename = original_txt.rename({0: 'Speaker',3: 'Start_candi',6: 'body'}, axis=1)
    # convert time
    start_candi = rename['Start_candi'].to_list()
    time_lst = []
    for i in start_candi: 
        time_str = i[1:]
        # convert hh:mm:ss into milliseconds
        h,m,s = time_str.split(':')
        millisecond = (int(m) * 60 + int(float(s)))*1000
        time_lst.append(millisecond)  
    return rename,time_lst

# Format v2: timepoints are in different lines from content
def read_transcription_two(transcription):
    original_txt = pd.read_csv(transcription,delimiter='\:\s',header=None, on_bad_lines='skip')
    # convert the index into column
    reindex = original_txt.reset_index()
    # reset the column headers
    rename = reindex.rename({'index': 'speaker', 0: 'body'}, axis=1)
    time = rename[rename['body'].isna()]
    # extract the start timepoints
    start_candi = time['speaker'].to_list()
    nonempty = rename.dropna()   
    time_lst = []
    for i in start_candi: 
        time_str = i[1:-1]
        # convert hh:mm:ss into milliseconds
        h,m,s = time_str.split(':')
        millisecond = (int(m) * 60 + int(float(s)))*1000
        time_lst.append(millisecond)
    return nonempty,time_lst

# append start and end timepoints to the dataframe
def append_time(nonempty,audio_length,time_lst):
    # add startpoints to the whole dataframe
    speaker_lst = nonempty['Speaker'].to_list()
    # remove whitespace in a string
    speakers = []
    for i in speaker_lst:
        renewed = i.replace(" ", "")
        speakers.append(renewed)
    speaker_type = list(dict.fromkeys(speakers))
    nonempty = nonempty.copy()
    nonempty['Start'] = time_lst
    nonempty['Speaker'] = speakers
    # create an empty dataframe
    new = pd.DataFrame()
    for i in speaker_type:
        time_i = nonempty.loc[nonempty['Speaker'] == i]
        # append the endpoints
        speaker_start = time_i['Start'].to_list()
        # add the end points as the beginning of the same speaker's next utterance
        time_i = time_i.copy()
        end_lst = speaker_start[1:]
        end_lst.append(audio_length)
        time_i['End'] = end_lst
        new = pd.concat([new, time_i])
    return new

# remove event description in the transcription 
def remove_event(raw,new):
    body_lst = []
    for i in raw: 
        # clean other annotations
        cleaned = re.sub(r'(\[\S+\s\S+\])|(\[\S+\])|(\[\S+\s\S+\s\S+\])','', i)
        body_lst.append(cleaned)
    # add the string length as a seperate column
    len_lst = []
    for i in body_lst:
        number = len(i.split())
        len_lst.append(number)
    new['Text'] = body_lst
    new['Length'] = len_lst
    new = new[new['Length'] != 0]
    return new

 
def extract_time(transcription,version):
    # get audio length
    audio_name = transcription[:-4] + '.wav'  
    audio = AudioSegment.from_file(audio_name)
    audio_length = audio.duration_seconds * 10000
    # read transcription and convert timepoints
    if version == 'one':
        nonempty,time_lst = read_transcription_one(transcription)
    elif version == 'two':
        nonempty,time_lst = read_transcription_two(transcription)
    new = append_time(nonempty,audio_length,time_lst)
    body_candi = new['body'].to_list()
    # remove other event markers and remove the non-existing transcription
    new = remove_event(body_candi)
    if version == 'one': 
        new.drop('Start_candi', inplace=True, axis=1)
    elif version == 'two':
        # Remove the redundant columns
        new.drop('body', inplace=True, axis=1)
        new.drop('speaker', inplace=True, axis=1)
    # print out the transcription files
    Speaker_lst = new['Speaker'].to_list()
    n = 0
    while n < new.shape[0]:
        body_lst = new['Text'].to_list()
        speaker = Speaker_lst[n]
        name = str(n) + '_' + speaker + '_' + transcription
        text_file = open(name, "w", encoding='utf-8')
        new = new.copy()
        content = body_lst[n]
        text_file.write(content)
        text_file.close()    
        n += 1
    # add additional line of filename and adust the order of columns
    new['Filename'] = transcription
    new = new.reindex(['Filename','Speaker','Text','Start','End','Length'], axis=1)
    return new


# segment audios according to different speakers
# segment audio file based on transcription
# input: path of raw transcriptions and audios; 
# output: a set of cleaned transcriptions and audios segmented on the utterance level 

def segment_audio(path):
    # get the filename list
    # loop the input folder to get the filename
    file_lst = []
    folder = os.fsencode(path)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            file_lst.append(filename) 
    # output a backup file with all the trnascription information
    new = pd.DataFrame()
    for transcription in file_lst:
        try:
            # get the audio file name
            audio_name = transcription[:-4] + '.wav'  
            data = extract_time(transcription,'one')
            n = 0
            Start_lst = data['Start'].to_list()
            End_lst = data['End'].to_list()
            Speaker_lst = data['Speaker'].to_list()
            # loop the dataframe
            n = 0
            while n < data.shape[0]: 
                start = Start_lst[n]
                end = End_lst[n]
                # segment seperate files based on speakers
                speaker = Speaker_lst[n]
                name = str(n) + '_' + speaker + '_' + audio_name
                # only export audio segmentations of conversational participants
                try:
                    recording = AudioSegment.from_wav(speaker+ '_' + audio_name)
                    segment = recording[start:end]               
                    segment.export(name, format="wav")       
                except:
                    print(name)
                n+=1
        except:
            print(transcription)
        new = pd.concat([new, data])
    return new


# Forced alignment based on MAUS(VAD has been incorporated in the pipeline)
# The audios are already mono-channel
def perform_FA(outpath):
    # loop the input folder to get the filename
    file_lst = []
    folder = os.fsencode(outpath)
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            name = filename[:-4]
            file_lst.append(name) 

    # log function
    def write_log(filename, stage):
        with open(osp.join(outpath, 'alignment_log.txt'), 'a') as log_file:
            log_file.write('{}\t{}\n'.format(filename, stage))
    
    
    # build request
    url = 'https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic'
    data = {r'LANGUAGE': 'fra-FR', r'OUTFORMAT': r'TextGrid'}
    
    # loop the file in the filelist
    for filename in file_lst:
        audio_file = filename + '.wav'
        text_file = filename + '.txt'
        files = {r'TEXT': open(osp.join(outpath, text_file), 'rb'),
                     r'SIGNAL': open(osp.join(outpath, audio_file), 'rb')}
        print('Sending request ...')
        r = requests.post(url, files=files, data=data)
        print('Processing results ...')
        
        if r.status_code == 200:
            root = etree.fromstring(r.text)
            success = root.find('success').text
            download_url = root.find('downloadLink').text
        
            if success != 'false':
                request_download = requests.get(download_url, stream=True)
                if request_download.status_code == 200:
                    try:
                        textgrid_file = filename + '.TextGrid'
                        with open(osp.join(outpath, textgrid_file), 'wb') as f:
                            f.write(request_download.content)
                        print('Complete Alignment')
                    except:
                        write_log(filename, 'FAIL Write TextGrid')
                        print('{} [{}]: {} FAIL Write TextGrid')
                        pass
                else:
                    write_log(filename, 'FAIL Download TextGrid')
                    print('{} [{}]: {} FAIL Download TextGrid')
            else:
                write_log(filename, 'FAIL Alignment')
                print(r.text)
                print('{} [{}]: {} FAIL Alignment')
        else:
            write_log(filename, 'FAIL Alignment Request')
            print('{} [{}]: {} FAIL Alignment Request')



def main():
    path = 'Path\\to\\your\\dataset'
    segment_audio(path)
    perform_FA(path)

if __name__ == "__main__":
    main()





