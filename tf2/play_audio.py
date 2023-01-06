import pyaudio 
import wave

#define stream chunk   
chunk = 16000  

#open a wav format music  
f = wave.open(r"C:\\ANDANTE\eprop\\GSC_SNN\\Salaj\\tmp\\speech_dataset\\backward\\0a2b400e_nohash_3.wav","rb")  
#instantiate PyAudio  
p = pyaudio.PyAudio()  
#open stream  

print('format: ', p.get_format_from_width(f.getsampwidth()))
print('channels: ', f.getnchannels())
print('rate: ', f.getframerate())

stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
#read data  
data = f.readframes(chunk)  

#play stream  
while data:  
    stream.write(data)  
    data = f.readframes(chunk)  

#stop stream  
stream.stop_stream()  
stream.close()  

#close PyAudio  
p.terminate() 