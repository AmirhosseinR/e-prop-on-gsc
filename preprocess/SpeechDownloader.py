"""
File containing scripts to download audio from various datasets

Also has tools to convert audio into numpy
"""
from tqdm import tqdm
import requests
import math
import os
import tarfile
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
# import pyaudio
# import audioUtils


# ##################
# Google Speech Commands Dataset V2
# ##################

# GSCmdV2Categs = {'unknown' : 0, 'silence' : 1, '_unknown_' : 0,'_silence_' : 1, '_background_noise_' : 1, 'yes' : 2,
#                 'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11}
# numGSCmdV2Categs = 12

# "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero",
# "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", and "Nine"

GSCmdV2Categs = {
    'unknown': 0,
    'silence': 0,
    '_unknown_': 0,
    '_silence_': 0,
    '_background_noise_': 0,
    'yes': 2,
    'no': 3,
    'up': 4,
    'down': 5,
    'left': 6,
    'right': 7,
    'on': 8,
    'off': 9,
    'stop': 10,
    'go': 11,
    'zero': 12,
    'one': 13,
    'two': 14,
    'three': 15,
    'four': 16,
    'five': 17,
    'six': 18,
    'seven': 19,
    'eight': 20,
    'nine': 1}
numGSCmdV2Categs = 21

def get_random_noise(noise_files, size):
    
    noise_idx = np.random.choice(len(noise_files))
    fs, noise_wav = wav.read(noise_files[noise_idx])
    # noise_wav = np.load(noise_files[noise_idx])
    
    offset = np.random.randint(len(noise_wav)-size)
    noise_wav = noise_wav[offset:offset+size].astype(float)
    
    return noise_wav

def generate_random_silence_files(nb_files, noise_files, size, prefix, sr=16000):
    
    for i in range(nb_files):
        silence_wav = get_random_noise(noise_files, size)
        # np.save(prefix+"_"+str(i)+".wav", silence_wav)
        wav.write(prefix+"_"+str(i)+".wav", sr, silence_wav.astype(np.int16))
        
        
def PrepareGoogleSpeechCmd(version=2, forceDownload=False, task='20cmd', basePath = None):
    """
    Prepares Google Speech commands dataset version 2 for use

    tasks: 20cmd, 12cmd, leftright or 35word

    Returns full path to training, validation and test file list and file categories
    """
    allowedTasks = ['1newclass','2newclass','3newclass','4newclass','5newclass','12cmd', 'leftright', '35word', '20cmd']
    if task not in allowedTasks:
        raise Exception('Task must be one of: {}'.format(allowedTasks))

    # basePath = None
    if version == 2:
        _DownloadGoogleSpeechCmdV2(forceDownload, basePath)
        basePath += 'GSC_V2'
    elif version == 1:
        _DownloadGoogleSpeechCmdV1(forceDownload, basePath)
        basePath += 'GSC_V1'
    else:
        raise Exception('Version must be 1 or 2')

    if task == '1newclass':
        GSCmdV2Categs = {
            'follow'   : 12}
        numGSCmdV2Categs = 1
    elif task == '2newclass':
        GSCmdV2Categs = {
            'follow'   : 12,
            'forward'  : 13}
        numGSCmdV2Categs = 2
    elif task == '3newclass':
        GSCmdV2Categs = {
            'follow'   : 12,
            'forward'  : 13,
            'learn'    : 14}
        numGSCmdV2Categs = 3
    elif task == '4newclass':
        GSCmdV2Categs = {
            'follow'   : 12,
            'forward'  : 13,
            'learn'    : 14,
            'visual'   : 15}
        numGSCmdV2Categs = 4
    elif task == '5newclass':
        GSCmdV2Categs = {
            'follow'   : 12,
            'forward'  : 13,
            'learn'    : 14,
            'visual'   : 15,
            'backward' : 16}
        numGSCmdV2Categs = 5
    elif task == '12cmd':
        GSCmdV2Categs = {
            'unknown': 0,
            'silence': 1,
            '_unknown_': 0,
            '_silence_': 1,
            '_background_noise_': 1,
            'yes': 2,
            'no': 3,
            'up': 4,
            'down': 5,
            'left': 6,
            'right': 7,
            'on': 8,
            'off': 9,
            'stop': 10,
            'go': 11}
        numGSCmdV2Categs = 12
    elif task == 'leftright':
        GSCmdV2Categs = {
            'unknown': 0,
            'silence': 0,
            '_unknown_': 0,
            '_silence_': 0,
            '_background_noise_': 0,
            'left': 1,
            'right': 2}
        numGSCmdV2Categs = 3
    elif task == '35word':
        GSCmdV2Categs = {
            'unknown': 0,
            'silence': 0,
            '_unknown_': 0,
            '_silence_': 0,
            '_background_noise_': 0,
            'yes': 2,
            'no': 3,
            'up': 4,
            'down': 5,
            'left': 6,
            'right': 7,
            'on': 8,
            'off': 9,
            'stop': 10,
            'go': 11,
            'zero': 12,
            'one': 13,
            'two': 14,
            'three': 15,
            'four': 16,
            'five': 17,
            'six': 18,
            'seven': 19,
            'eight': 20,
            'nine': 1,
            'backward': 21,
            'bed': 22,
            'bird': 23,
            'cat': 24,
            'dog': 25,
            'follow': 26,
            'forward': 27,
            'happy': 28,
            'house': 29,
            'learn': 30,
            'marvin': 31,
            'sheila': 32,
            'tree': 33,
            'visual': 34,
            'wow': 35}
        numGSCmdV2Categs = 36
    elif task == '20cmd':
        GSCmdV2Categs = {
            'unknown': 0,
            'silence': 0,
            '_unknown_': 0,
            '_silence_': 0,
            '_background_noise_': 0,
            'yes': 2,
            'no': 3,
            'up': 4,
            'down': 5,
            'left': 6,
            'right': 7,
            'on': 8,
            'off': 9,
            'stop': 10,
            'go': 11,
            'zero': 12,
            'one': 13,
            'two': 14,
            'three': 15,
            'four': 16,
            'five': 17,
            'six': 18,
            'seven': 19,
            'eight': 20,
            'nine': 1}
        numGSCmdV2Categs = 21

    # print('Converting test set WAVs to numpy files')
    # audioUtils.WAV2Numpy(basePath + '/test/')
    # print('Converting training set WAVs to numpy files')
    # audioUtils.WAV2Numpy(basePath + '/train/')
    
    newTasks = ['1newclass','2newclass','3newclass','4newclass','5newclass']
    # if task not in newTasks:
    noise_path  = os.path.join(basePath  + '/train' +  '/_background_noise_')
    noise_files = [fname for fname in os.listdir(noise_path)]
    noiseWAVs   = [noise_path + '/' + fname for fname in os.listdir(noise_path) if fname.endswith('.wav')]


    # read split from files and all files in folders
    testWAVs = pd.read_csv(basePath + '/train/testing_list.txt',
                           sep=" ", header=None)[0].tolist()
    valWAVs = pd.read_csv(basePath + '/train/validation_list.txt',
                          sep=" ", header=None)[0].tolist()

    testWAVs = [os.path.join(basePath + '/train/', f)
                for f in testWAVs if f.endswith('.wav')]
    valWAVs = [os.path.join(basePath + '/train/', f)
               for f in valWAVs if f.endswith('.wav')]
    allWAVs = []
    for root, dirs, files in os.walk(basePath + '/train/'):
        allWAVs += [root + '/' + f for f in files if f.endswith('.wav')]
    
    trainWAVs = list(set(allWAVs) - set(valWAVs) - set(testWAVs) - set(noiseWAVs))


    if task in newTasks:
        for k in GSCmdV2Categs.keys():
            new_class_path  = os.path.join(basePath  + '/train' + '/' + k)
            allNewWAVs    = [new_class_path + '/' + fname for fname in os.listdir(new_class_path) if fname.endswith('.wav')]

        testNewWAVs  = list(set(allNewWAVs) - set(valWAVs)  - set(trainWAVs))
        valNewWAVs   = list(set(allNewWAVs) - set(testWAVs) - set(trainWAVs))
        trainNewWAVs = list(set(allNewWAVs) - set(valWAVs)  - set(testWAVs))


    testWAVsREAL = []
    for root, dirs, files in os.walk(basePath + '/test/'):
        testWAVsREAL += [root + '/' +
                         f for f in files if f.endswith('.wav')]

    # get categories
    if task in newTasks:
        testWAVlabels     = [_getFileCategory(f, GSCmdV2Categs) for f in testNewWAVs]
        valWAVlabels      = [_getFileCategory(f, GSCmdV2Categs) for f in valNewWAVs]
        trainWAVlabels    = [_getFileCategory(f, GSCmdV2Categs) for f in trainNewWAVs]
        testWAVREALlabels = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVsREAL] # Not updated
    else:
        testWAVlabels     = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVs]
        valWAVlabels      = [_getFileCategory(f, GSCmdV2Categs) for f in valWAVs]
        trainWAVlabels    = [_getFileCategory(f, GSCmdV2Categs) for f in trainWAVs]
        testWAVREALlabels = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVsREAL]

    
    if task not in newTasks:
        size = 16000
        silence_folder = os.path.join(basePath  + '/train' +  '/_silence_')

        if not os.path.exists(silence_folder):
            os.makedirs(silence_folder)
            # 260 validation / 2300 training
            print('generate random silence files...')
            generate_random_silence_files(2560, noiseWAVs, size, os.path.join(silence_folder + "/rd_silence"))    
            # audioUtils.WAV2Numpy(silence_folder)
        
        # save 260 files for validation
        silence_files = [fname for fname in os.listdir(silence_folder)]
        silence_files_val =  [None] * 260
        silence_files_trn =  [None] * (2560 - 260)
        
        for i in range(0, len(silence_files_val)):
            silence_files_val[i] = silence_folder + '/'  + silence_files[i]
        silence_files_vallabels = [GSCmdV2Categs['silence']] * 260
        
        for i in range(0, len(silence_files_trn)):
            silence_files_trn[i] = silence_folder + '/' + silence_files[i+260]
        silence_files_trnlabels = [GSCmdV2Categs['silence']] * (2560 - 260)
        
        if numGSCmdV2Categs == 12:
            valWAVs        += silence_files_val
            valWAVlabels   += silence_files_vallabels
            trainWAVs      += silence_files_trn
            trainWAVlabels += silence_files_trnlabels
        

    # build dictionaries
    if task in newTasks:
        testWAVlabelsDict = dict(zip(testNewWAVs, testWAVlabels))
        valWAVlabelsDict = dict(zip(valNewWAVs, valWAVlabels))
        trainWAVlabelsDict = dict(zip(trainNewWAVs, trainWAVlabels))
        testWAVREALlabelsDict = dict(zip(testWAVsREAL, testWAVREALlabels)) # Not updated
    else:
        testWAVlabelsDict = dict(zip(testWAVs, testWAVlabels))
        valWAVlabelsDict = dict(zip(valWAVs, valWAVlabels))
        trainWAVlabelsDict = dict(zip(trainWAVs, trainWAVlabels))
        testWAVREALlabelsDict = dict(zip(testWAVsREAL, testWAVREALlabels))

    # a tweak here: we will heavily underuse silence samples because there are few files.
    # we can add them to the training list to reuse them multiple times
    # note that since we already added the files to the label dicts we don't
    # need to do it again

    # for i in range(200):
    #     trainWAVs = trainWAVs + backNoiseFiles

    # info dictionary
    if task in newTasks:
        trainInfo = {'files': trainNewWAVs, 'labels': trainWAVlabelsDict}
        valInfo = {'files': valNewWAVs, 'labels': valWAVlabelsDict}
        testInfo = {'files': testNewWAVs, 'labels': testWAVlabelsDict}
        testREALInfo = {'files': testWAVsREAL, 'labels': testWAVREALlabelsDict} # Not updated
        gscInfo = {'train': trainInfo,
                'test': testInfo,
                'val': valInfo,
                'testREAL': testREALInfo}
    else:
        trainInfo = {'files': trainWAVs, 'labels': trainWAVlabelsDict}
        valInfo = {'files': valWAVs, 'labels': valWAVlabelsDict}
        testInfo = {'files': testWAVs, 'labels': testWAVlabelsDict}
        testREALInfo = {'files': testWAVsREAL, 'labels': testWAVREALlabelsDict}
        gscInfo = {'train': trainInfo,
                'test': testInfo,
                'val': valInfo,
                'testREAL': testREALInfo}

    print('Done preparing Google Speech commands dataset version {}'.format(version))

    return gscInfo, numGSCmdV2Categs


def _getFileCategory(file, catDict):
    """
    Receives a file with name GSC_V2/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ, 0)


def _DownloadGoogleSpeechCmdV2(forceDownload=False, basePath = ''):
    """
    Downloads Google Speech commands dataset version 2
    """
    if os.path.isdir(basePath + "GSC_V2/") and not forceDownload:
        print('Google Speech commands dataset version 2 already exists. Skipping download.')
    else:
        if not os.path.exists(basePath + "GSC_V2/"):
            os.makedirs(basePath + "GSC_V2/")
        trainFiles = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        testFiles  = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
        _downloadFile(testFiles,  basePath + 'GSC_V2/test.tar.gz')
        _downloadFile(trainFiles, basePath + 'GSC_V2/train.tar.gz')

    # extract files
    if not os.path.isdir(basePath + "GSC_V2/test/"):
        _extractTar(basePath + 'GSC_V2/test.tar.gz', basePath + 'GSC_V2/test/')

    if not os.path.isdir(basePath + "GSC_V2/train/"):
        _extractTar(basePath + 'GSC_V2/train.tar.gz', basePath + 'GSC_V2/train/')


def _DownloadGoogleSpeechCmdV1(forceDownload=False, basePath = ''):
    """
    Downloads Google Speech commands dataset version 1
    """
    if os.path.isdir(basePath + "GSC_V1/") and not forceDownload:
        print('Google Speech commands dataset version 1 already exists. Skipping download.')
    else:
        if not os.path.exists(basePath + "GSC_V1/"):
            os.makedirs(basePath + "GSC_V1/")
        trainFiles = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
        testFiles  = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz'
        _downloadFile(testFiles,  basePath + 'GSC_V1/' + 'test.tar.gz')
        _downloadFile(trainFiles, basePath + 'GSC_V1/' + 'train.tar.gz')

    # extract files
    if not os.path.isdir(basePath + "GSC_V1/test/"):
        _extractTar(basePath + 'GSC_V1/test.tar.gz', basePath + 'GSC_V1/test/')

    if not os.path.isdir(basePath + "GSC_V1/train/"):
        _extractTar(basePath + 'GSC_V1/train.tar.gz', basePath + 'GSC_V1/train/')

##############
# Utilities
##############


def _downloadFile(url, fName):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    print('Downloading {} into {}'.format(url, fName))
    with open(fName, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size // block_size),
                         unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")


def _extractTar(fname, folder):
    print('Extracting {} into {}'.format(fname, folder))
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=folder)
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall(path=folder)
        tar.close()
