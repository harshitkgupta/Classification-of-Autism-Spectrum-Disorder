import alsaaudio, struct
import aubio
from aubio.task import *
 
# constants
CHANNELS    = 1
INFORMAT    = alsaaudio.PCM_FORMAT_FLOAT_LE
RATE        = 44100
FRAMESIZE   = 1024
PITCHALG    = aubio_pitch_yin
PITCHOUT    = aubio_pitchm_freq
 
# set up audio input
recorder=alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE)
recorder.setchannels(CHANNELS)
recorder.setrate(RATE)
recorder.setformat(INFORMAT)
recorder.setperiodsize(FRAMESIZE)
 
# set up pitch detect
detect = new_aubio_pitchdetection(FRAMESIZE,FRAMESIZE/2,CHANNELS,
                                  RATE,PITCHALG,PITCHOUT)
buf = new_fvec(FRAMESIZE,CHANNELS)
 
# main loop
runflag = 1
while runflag:
 
  # read data from audio input
  [length, data]=recorder.read()
 
  # convert to an array of floats
  floats = struct.unpack('f'*FRAMESIZE,data)
 
  # copy floats into structure
  for i in range(len(floats)):
    fvec_write_sample(buf, floats[i], 0, i)
 
  # find pitch of audio frame
  freq = aubio_pitchdetection(detect,buf)
 
  # find energy of audio frame
  energy = vec_local_energy(buf)
 
  print "{:10.4f} {:10.4f}".format(freq,energy)
