# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML

class Video:  
    def save_video(folder, img_list, G_losses, D_losses):
        plt.rcParams['animation.ffmpeg_path'] = '/home/ramanlab/anaconda3/pkgs/ffmpeg-3.1.3-0/bin/ffmpeg'
         
         #%%capture
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        
        HTML(ani.to_jshtml())
        ani.save('animation.mp4')
        
        # define a list of places
        
        with open('training_vars.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump([img_list,G_losses,D_losses], filehandle)
        
        return ims, ani
