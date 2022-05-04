# -*- coding: utf-8 -*-
"""
Automatic segmentation of audio files
Conduct the MAUS forced aligner(https://www.bas.uni-muenchen.de/Bas/BasMAUS.html)
The FA part is inspired by @WilliamNHavard's paper on sentence-aligned Spoken utterances 

@author: Jing Liu
"""

import re
from pydub import AudioSegment
import pandas as pd
import os
import os.path as osp
import requests
from lxml import etree
import textgrid
import shutil

# segment audio file based on transcription(for improvement: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/326)
# input: raw transcriptions with events(in the list form); output: a cleaned transcription with texts(in the list form)
def remove_event(raw):
    body_lst = []
    for i in raw: 
        # clean other annotations
        cleaned = re.sub(r'(\[\S+\s\S+\])|(\[\S+\])|(\[\S+\s\S+\s\S+\])|(\[ Rires\.\])|(\(\S+\))','', i)
        body_lst.append(cleaned)
    # add the string length as a seperate column
    len_lst = []
    for i in body_lst:
        number = len(i.split())
        len_lst.append(number)
    return body_lst,len_lst

# convert the hh:mm:ss time into miliseconds; generate the endtime based on the start of the next utterance
# input: start time list;length of the  output: start time list; endtime list 

# convert the hh:mm:ss time into miliseconds; generate the endtime based on the start of the next utterance
# input: start time list;length of the  output: start time list; endtime list 

def convert_time(rename,audio_length):
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
    # add startpoints to the whole dataframe
    speaker_lst = nonempty['speaker'].to_list()
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

# Version 1: For the starting point and text in the same line 
def extract_time_v1(transcription):
    audio_name = transcription[:-4] + '.wav'  
    audio = AudioSegment.from_file(audio_name)
    audio_length = audio.duration_seconds * 1000
    original_txt = pd.read_csv(transcription,delimiter='(\:\s)|(\]\s)',header=None, on_bad_lines='skip')
    
     # remove unnecessary lines
    original_txt.drop(1, inplace=True, axis=1)
    original_txt.drop(2, inplace=True, axis=1)
    original_txt.drop(4, inplace=True, axis=1)
    original_txt.drop(5, inplace=True, axis=1)
    # reset the column headers
    nonempty = original_txt.rename({0: 'Speaker',3: 'Start_candi',6: 'body'}, axis=1)
    # convert time
    start_candi = nonempty['Start_candi'].to_list()
    time_lst = []
    for i in start_candi: 
        time_str = i[1:]
        # convert hh:mm:ss into milliseconds
        h,m,s = time_str.split(':')
        millisecond = (int(m) * 60 + int(float(s)))*1000
        time_lst.append(millisecond)  
    speaker_lst = nonempty['Speaker'].to_list()
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
    
    body_candi = new['body'].to_list()
    # remove other event markers
    body_lst,len_lst = remove_event(body_candi)
    new['Text'] = body_lst
    new['Length'] = len_lst
    # remove the non-existing transcription
    new = new[new['Length'] != 0]
    # Remove the redundant columns
    new.drop('body', inplace=True, axis=1)
    new.drop('Start_candi', inplace=True, axis=1)
    # print out the transcription files
    Speaker_lst = new['Speaker'].to_list()
    utterance_lst = []
    n = 0
    while n < new.shape[0]:
        body_lst = new['Text'].to_list()
        speaker = Speaker_lst[n]
        name = str(n) + '_' + speaker + '_' + transcription[:-4]
        utterance_lst.append(name)
        text_file = open(name, "w", encoding='utf-8')
        new = new.copy()
        content = body_lst[n]
        text_file.write(content)
        text_file.close()    
        n += 1
    new['Filename'] = transcription
    new['UtteranceName'] = utterance_lst
    # reorder the column names
    data = new.reindex(['Filename','UtteranceName','Speaker','Text','Start','End','Length'], axis=1)
    data['Start'] = data['Start']
    data['End'] = data['End']
    return data


# Version 2: For the starting point and text in different lines 
def extract_time_v2(transcription):
    audio_name = transcription[:-4] + '.wav'  
    audio = AudioSegment.from_file(audio_name)
    audio_length = audio.duration_seconds * 1000
    original_txt = pd.read_csv(transcription,delimiter='\:\s',header=None, on_bad_lines='skip')
    # convert the index into column
    reindex = original_txt.reset_index()
    # reset the column headers
    rename = reindex.rename({'index': 'speaker', 0: 'body'}, axis=1)
    # set the start and end time
    new = convert_time(rename,audio_length)
    body_candi = new['body'].to_list()
    # remove other event markers
    body_lst,len_lst = remove_event(body_candi)
    new['Text'] = body_lst
    new['Length'] = len_lst
    # remove the non-existing transcription
    new = new[new['Length'] != 0]
    # Remove the redundant columns
    new.drop('body', inplace=True, axis=1)
    new.drop('speaker', inplace=True, axis=1)
    # print out the transcription files
    Speaker_lst = new['Speaker'].to_list()
    utterance_lst = []
    n = 0
    while n < new.shape[0]:
        body_lst = new['Text'].to_list()
        speaker = Speaker_lst[n]
        name = str(n) + '_' + speaker + '_' + transcription[:-4]
        utterance_lst.append(name)
        text_file = open(name, "w", encoding='utf-8')
        new = new.copy()
        content = body_lst[n]
        text_file.write(content)
        text_file.close()    
        n += 1
    new['Filename'] = transcription[:-4]
    new['UtteranceName'] = utterance_lst
    # reorder the column names
    data = new.reindex(['Filename','UtteranceName','Speaker','Text','Start','End','Length'], axis=1)
    data['Start'] = data['Start']
    data['End'] = data['End']
    return data


# Version 3: For the starting point and text in the same line 
def extract_time(transcription):
    audio_name = transcription[:-4] + '.wav'  
    audio = AudioSegment.from_file(audio_name)
    audio_length = audio.duration_seconds * 1000
    original_txt = pd.read_csv(transcription,delimiter='(\:\s)|(\]\s)',header=None, on_bad_lines='skip')
    
     # remove unnecessary lines
    original_txt.drop(1, inplace=True, axis=1)
    original_txt.drop(2, inplace=True, axis=1)
    original_txt.drop(4, inplace=True, axis=1)
    original_txt.drop(5, inplace=True, axis=1)
    # reset the column headers
    nonempty = original_txt.rename({3: 'Speaker',0: 'Start_candi',6: 'body'}, axis=1)
    
    # convert hh:mm:ss into seconds
    start_candi = nonempty['Start_candi'].to_list()
    time_lst = []
    for i in start_candi: 
        time_str = i[1:]
        # convert hh:mm:ss into milliseconds
        h,m,s = time_str.split(':')
        millisecond = (int(m) * 60 + int(float(s)))*1000
        time_lst.append(millisecond)  
    speaker_lst = nonempty['Speaker'].to_list()
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
    
    body_candi = new['body'].to_list()
    # remove other event markers
    body_lst,len_lst = remove_event(body_candi)
    new['Text'] = body_lst
    new['Length'] = len_lst
    # remove the non-existing transcription
    new = new[new['Length'] != 0]
    # Remove the redundant columns
    new.drop('body', inplace=True, axis=1)
    new.drop('Start_candi', inplace=True, axis=1)
    # print out the transcription files
    Speaker_lst = new['Speaker'].to_list()
    utterance_lst = []
    n = 0
    while n < new.shape[0]:
        body_lst = new['Text'].to_list()
        speaker = Speaker_lst[n]
        name = str(n) + '_' + speaker + '_' + transcription
        utterance_lst.append(name)
        text_file = open(name, "w", encoding='utf-8')
        new = new.copy()
        content = body_lst[n]
        
        text_file.write(content)
        text_file.close()    
        n += 1
    new['Filename'] = transcription[:-4]
    new['UtteranceName'] = utterance_lst
    # reorder the column names
    data = new.reindex(['Filename','UtteranceName','Speaker','Text','Start','End','Length'], axis=1)
    data['Start'] = data['Start']
    data['End'] = data['End']
    data.to_csv('Utterance.csv')
    return data


# segment audios according to different speakers
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
            data = extract_time(transcription)
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
            new = pd.concat([new, data]) 
        except:
            print(transcription)
        
    return new


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
    # to change langauge setting, see: https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/WebServices/BAS_Webservices.cmdi.xml
    
    url = 'https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic'
    data = {r'LANGUAGE': 'eng-US', r'OUTFORMAT': r'TextGrid'}
    
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
                        print('{} [{}]: {} OK')
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



# parse the textgrid to get the start and end points of each word
def parse_Interval(IntervalObject):
    start_time = ""
    end_time = ""
    P_name = ""
    ind = 0
    str_interval = str(IntervalObject)
    # print(str_interval)
    for ele in str_interval:
        if ele == "(":
            ind = 1
        if ele == " " and ind == 1:
            ind = 2
        if ele == "," and ind == 2:
            ind = 3
        if ele == " " and ind == 3:
            ind = 4

        if ind == 1:
            if ele != "(" and ele != ",":
                start_time = start_time + ele
        if ind == 2:
            end_time = end_time + ele
        if ind == 4:
            if ele != " " and ele != ")":
                P_name = P_name + ele

    st = float(start_time)
    et = float(end_time)
    pn = P_name
    return [pn,st,et]

# get a list of all the phonemes
def parse_textgrid(filename,layername):
    tg = textgrid.TextGrid.fromFile(filename)
    list_words = tg.getList(layername)
    words_list = list_words[0]
    final = pd.DataFrame()
    for ele in words_list:
        pho = parse_Interval(ele)
        #target_lst.append(pho)
        data = pd.DataFrame(pho).T
        final = pd.concat([data,final])
    # remove silence in the dataframe
    final = final.rename({0: 'Word',1: 'Start',2: 'End'}, axis=1)
    final['Length'] = final['End'] - final['Start']
    final = final[final.Word != 'None']
    utterance_candi = filename.split('.')
    final['UtteranceName'] = utterance_candi[0]
    speaker_candi = filename.split('_')
    speaker = speaker_candi[1]
    audioname = speaker_candi[-1]
    name = audioname.split('.')
    final['Speaker'] = speaker
    final['Filename'] = name[0]
    return final


# read and parse textgrid files within the filepath
# input: folder path containing all the textgrids
# output: a dataframe comtaining word, start/end point, duration

def get_duration(path,layername,transcription):  
    # remove suffix of utterancename
    temp = transcription['UtteranceName'].tolist()
    UtteranceNew = []
    for i in temp:
        new_name = i[:-4]
        UtteranceNew.append(new_name)
    transcription['UtteranceName'] = UtteranceNew
    
    folder = os.fsencode(path)
    final = pd.DataFrame()
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith('TextGrid'):
            data = parse_textgrid(filename,layername)        
            # change the duration scale
            # match via utterance name
            global_start = transcription.loc[transcription['UtteranceName'] == data['UtteranceName'].tolist()[0], 'Start'].tolist()[0]
            renewed_start = data['Start']/1000 + global_start
            renewed_end = data['End']/1000 + global_start
            data['Global_start'] = renewed_start
            data['Global_end'] = renewed_end
            final = pd.concat([data,final])
        final.to_csv('word.csv', index=False)    
    return final

def move_unaligned(path, transcription,word):
    whole_utt = transcription['UtteranceName'].tolist()
    #duplicate = [item for item, count in collections.Counter(whole_utt).items() if count > 1]
    aligned_utterances = word['UtteranceName'].tolist()
    aligned_utterances = pd.DataFrame (aligned_utterances, columns = ['utterances'])
    #duplicate2 = [item for item, count in collections.Counter(aligned_utterances).items() if count > 1]
    whole_utt = pd.DataFrame (whole_utt, columns = ['utterances'])
    label = pd.concat([aligned_utterances, whole_utt]).drop_duplicates(keep=False)    
    file = label['utterances'].tolist()        
    # move the unaligned files to a different folder
    original_path = path + '\\aligned'
    new_path = path + '\\temp'
    missing = []
    for i in file:
        # move txt file
        txt_file = i + '.txt'
        wav_file = i + '.wav'
        try:
            shutil.move(original_path + '\\' + txt_file, new_path + '\\' + txt_file)
        except: 
            print(txt_file)
            missing.append(txt_file)
        try:
            shutil.move(original_path + '\\' + wav_file, new_path + '\\' + wav_file)
        except: 
            print(wav_file)
            missing.append(wav_file)
    return missing
 
    
# def main():
#     path = 'C:\\Users\\Crystal\\OneDrive\\Desktop\\trial'
#     transcription = segment_audio(path)
#     perform_FA(path)
#     get_duration(path,"ORT-MAU",transcription)
    
# if __name__ == "__main__":
#     main()


path = 'C:\\Users\\Crystal\\OneDrive\\Desktop\\trial'
transcription = segment_audio(path)
perform_FA(path +'\\segmented')
get_duration(path +'\\segmented',"ORT-MAU",transcription)


