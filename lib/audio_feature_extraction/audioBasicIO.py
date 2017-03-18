import os, glob, eyed3, ntpath, shutil, numpy
import scipy.io.wavfile as wavfile
from pydub import AudioSegment

log_tag = "AUDIO_BASIC_IO"

def convertDirMP3ToWav(dirName, Fs, nC, useMp3TagsAsName = False):
    '''
    This function converts the MP3 files stored in a folder to WAV. If required, the output names of the WAV files are based on MP3 tags, otherwise the same names are used.
    ARGUMENTS:
     - dirName:     the path of the folder where the MP3s are stored
     - Fs:          the sampling rate of the generated WAV files
     - nC:          the number of channesl of the generated WAV files
     - useMp3TagsAsName:    True if the WAV filename is generated on MP3 tags
    '''

    print "[AUDIO_BASIC_IO] converting dir: '" + dirName +"' mp3 to wav (using mp3 tags as name: " + "true" if useMp3TagsAsName else "false" + ", Fs:" + str(Fs) + ", number of channels: " + str(nC) + ")"
    types = (dirName + os.sep + '*.mp3',) # the tuple of file types
    filesToProcess = []

    for files in types:
        filesToProcess.extend(glob.glob(files))

    print "[AUDIO_BASIC_IO] files: " + ', '.join(filesToProcess)

    for f in filesToProcess:
        #tag.link(f)
        audioFile = eyed3.load(f)
        if useMp3TagsAsName and audioFile.tag != None:
            artist = audioFile.tag.artist
            title = audioFile.tag.title
            if artist!=None and title!=None:
                if len(title)>0 and len(artist)>0:
                    wavFileName = ntpath.split(f)[0] + os.sep + artist.replace(","," ") + " --- " + title.replace(","," ") + ".wav"
                else:
                    wavFileName = f.replace(".mp3",".wav")
            else:
                wavFileName = f.replace(".mp3",".wav")
        else:
            wavFileName = f.replace(".mp3",".wav")
        command = "avconv -i \"" + f + "\" -ar " +str(Fs) + " -ac " + str(nC) + " \"" + wavFileName + "\"";
        print "[AUDIO_BASIC_IO] command: " + command
        os.system(command.decode('unicode_escape').encode('ascii','ignore').replace("\0",""))

def convertFsDirWavToWav(dirName, Fs, nC):
    '''
    This function converts the WAV files stored in a folder to WAV using a different sampling freq and number of channels.
    ARGUMENTS:
     - dirName:     the path of the folder where the WAVs are stored
     - Fs:          the sampling rate of the generated WAV files
     - nC:          the number of channesl of the generated WAV files
    '''
    print "[AUDIO_BASIC_IO] converting dir: '" + dirName +"' wav to wav (using mp3 tags as name: " + "true" if useMp3TagsAsName else "false" + ", Fs:" + str(Fs) + ", number of channels: " + str(nC) + ")"

    types = (dirName+os.sep+'*.wav',) # the tuple of file types
    filesToProcess = []

    for files in types:
        filesToProcess.extend(glob.glob(files))     

    newDir = dirName + os.sep + "Fs" + str(Fs) + "_" + "NC"+str(nC)
    if os.path.exists(newDir) and newDir!=".":
        shutil.rmtree(newDir)   
    os.makedirs(newDir) 

    for f in filesToProcess:    
        _, wavFileName = ntpath.split(f)    
        command = "avconv -i \"" + f + "\" -ar " +str(Fs) + " -ac " + str(nC) + " \"" + newDir + os.sep + wavFileName + "\"";
        print "[AUDIO_BASIC_IO] command: " + command
        os.system(command)

def readAudioFile(path):
    '''
    This function returns a numpy array that stores the audio samples of a specified WAV of AIFF file
    '''
    print "[AUDIO_BASIC_IO] reading audio file, path: '" + path + "'"

    extension = os.path.splitext(path)[1]

    try:
        #if extension.lower() == '.wav':
            #[Fs, x] = wavfile.read(path)
        if extension.lower() == '.aif' or extension.lower() == '.aiff':
            s = aifc.open(path, 'r')
            nframes = s.getnframes()
            strsig = s.readframes(nframes)
            x = numpy.fromstring(strsig, numpy.short).byteswap()
            Fs = s.getframerate()
        elif extension.lower() == '.mp3' or extension.lower() == '.wav' or extension.lower() == '.au':
            audiofile = AudioSegment.from_file(path)
            data = numpy.fromstring(audiofile._data, numpy.int16)
            Fs = audiofile.frame_rate
            x = []
            for chn in xrange(audiofile.channels):
                x.append(data[chn::audiofile.channels])
            x = numpy.array(x).T
        else:
            print "[AUDIO_BASIC_IO] Error in readAudioFile(): Unknown file type!"
            raise Exception("Error in readAudioFile(): Unknown file type!")
            return (-1,-1)
    except IOError: 
        print "[AUDIO_BASIC_IO] Error: file not found or other I/O error."
        raise
        #return (-1,-1)

    if x.ndim==2:
        if x.shape[1]==1:
            x = x.flatten()

    return (Fs, x)

def stereo2mono(x):
    '''
    This function converts the input signal (stored in a numpy array) to MONO (if it is STEREO)
    '''
    if x.ndim==1:
        return x
    elif x.ndim==2:
        if x.shape[1]==1:
            return x.flatten()
        else:
            if x.shape[1]==2:
                return ( (x[:,1] / 2) + (x[:,0] / 2) )
            else:
                return -1

