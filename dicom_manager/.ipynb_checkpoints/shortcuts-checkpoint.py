# standard imports 

import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import zipfile
import sys
from glob import glob
from tqdm import tqdm

from matplotlib.patches import Patch
from matplotlib.lines import Line2D



def dice_stats(dict_dice):
    
    dict_dice_sorted = {k: v for k, v in sorted(dict_dice.items(), key=lambda item: item[1])}
    
    print('AVERAGE DICE: ', str(sum(v for v in dict_dice.values() ) / len(dict_dice)), '\n')
    
    print('LOWEST DICE: ')
    for num, (key, value) in enumerate(dict_dice_sorted.items()): 
        print ('\t', key, ' : ', value)
        if num > 5: break
    
    return dict_dice_sorted


def unzip_check(dir_unzip):
    
    list_zip = glob(os.path.join(dir_unzip, '**/*.zip'), recursive = True)
    list_unzip = [file for file in list_zip if not os.path.isdir(os.path.split(file)[0])]
    
    return list_zip, list_unzip



def normalize_arr(arr, norm_range = [0, 1]): 
    
    norm = (norm_range[1] - norm_range[0]) * ((arr - np.amin(arr))/(np.amax(arr) - np.amin(arr))) + norm_range[0]
    
    return norm

# def normalize_arr(arr, norm_range = [0, 1]):  #fix so works with negative
    
#     norm = (norm_range[1]*(arr - np.amin(arr))/np.ptp(arr)) 
    
#     return norm



def print_range(arr):
    
    print('RANGE: ', np.amin(arr), ' : ', np.amax(arr))
    
    return



def plot_res(list_img, mag = 1, row_col = False, legend = False, legend_size = 12, axis = True, tight = False, fig_size = False, loc_save = False, close = False):
    
    '''
    list_img[list] = list of images to plot
    mag[int] = plot size
    row_col[list] = list of [row, col] for organization of plots
    legend[list] = nested list [[label1, color1], [label2, color2],...]
    '''
        
    if not row_col : row_col = [1, len(list_img)]
    if fig_size:
        fig = plt.figure(figsize = fig_size)
    else:
        fig = plt.figure(figsize = (15 * mag, 15 * mag), dpi = 64)
    
    for i in range(1, (1 + len(list_img))):
        ax = fig.add_subplot(row_col[0], row_col[1], i)
        if legend:
            legend_elements = []
            for j in range(len(legend[0])):
                legend_elements.append(Patch(facecolor = legend[1][j], edgecolor = legend[1][j], label = legend[0][j]))
            ax.legend(handles=legend_elements, loc='upper left', prop={'size':  legend_size * mag})
        if loc_save: plt.imshow((list_img[i - 1]).astype(np.uint8), cmap = 'gray')
        if not loc_save: plt.imshow((255*list_img[i - 1]).astype(np.uint8), cmap = 'gray')
       
    if not axis: plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if tight: plt.tight_layout()
    
    if not loc_save: plt.show()
    if loc_save: plt.savefig(loc_save)
    if close: plt.close() 
        
    

def remove_mac(dir_mac):
    
    list_ds = glob(os.path.join(dir_mac, '.DS_Store'))
    for file in list_ds: os.remove(file) 
    list_ds = glob(os.path.join(dir_mac, '**/.DS_Store'), recursive = True)
    for file in list_ds: os.remove(file) 
       
    list_os = glob(os.path.join(dir_mac, '__MACOSX'))
    for file in list_os: shutil.rmtree(file) 
    list_os = glob(os.path.join(dir_mac, '**/__MACOSX'), recursive = True)
    for file in list_os: shutil.rmtree(file) 
    
    return



# clean up a little so it does recursive stuff smarter -- descend into dir after unzipping
def unzip_rec(file_zip, remove_zips = False, _remove_mac = True): #add remove zip files option, add removeMac option
    
    if file_zip[-4:] == '.zip':
        with zipfile.ZipFile(file_zip, 'r') as zip_ref:
            zipto = file_zip.split('.zip')[0]
            while ' .' in zipto:
                zipto = '.'.join(zipto.split(' .')) 
            zipto = "_".join(zipto).split(" ")
            try:
                zip_ref.extractall(zipto)
            except:
                print('ERROR: unzip fail -- ', file_zip)
            for tail in os.listdir(zipto):
                dst = os.path.join(zipto, tail)
                while dst[-1] == ' ':
                    dst = dst[:-1]
                if not dst == os.path.join(zipto, tail):
                    print('ERROR: empty space on dir name -- ', zipto)
                    shutil.move(os.path.join(zipto, tail), src)
                    
        dir_zip = zipto
    elif os.path.isdir(file_zip):
        dir_zip = file_zip
    else:
        print('file_zip arg needs to be either .zip or a directory')
    
    list_error = []
    zips_remain = True
    while zips_remain:
        
        list_glob = glob(os.path.join(dir_zip, '**/*.zip'), recursive = True)
        list_unzipped = [file for file in list_glob if not os.path.exists(file.split('.zip')[0]) and not os.path.exists(''.join(file.split('.zip')[0].split(' ')))]
        
        for file_zip in list_unzipped:
            with zipfile.ZipFile(file_zip, 'r') as zip_ref:
                zipto = file_zip.split('.zip')[0]
                while ' .' in zipto:
                    zipto = '.'.join(zipto.split(' .')) 
                try:
                    zip_ref.extractall(zipto)
                except:
                    print('ERROR: unzip fail -- ', file_zip)
                    list_error.append(file_zip)
                    continue
                for tail in os.listdir(zipto):
                    dst = os.path.join(zipto, tail)
                    while dst[-1] == ' ':
                        dst = dst[:-1]
                    if not dst == os.path.join(zipto, tail):
                        print('ERROR: empty space on dir name -- ', zipto)
                        shutil.move(os.path.join(zipto, tail), dst)
        list_glob = glob(os.path.join(dir_zip, '**/*.zip'), recursive = True)
        list_unzipped = [file for file in list_glob if not os.path.exists(file.split('.zip')[0]) and not os.path.exists(''.join(file.split('.zip')[0].split(' ')))]
        if len(list_unzipped) == 0 + len(list_error) : zips_remain = False 
            
    if _remove_mac: remove_mac(dir_zip)
    
    return



def build_dir(dir_check, overwrite = False):
    
    if overwrite and os.path.exists(dir_check):
        shutil.rmtree(dir_check)
    
    if not os.path.exists(dir_check):
        os.makedirs(dir_check, exist_ok = True)
#     else:
#         print(dir_check, 'ERROR: directory exists, set overwrite arg to True')
        
    return




def histo(arr, bins = 10, threshold = False, title = False):
    #arr
    #bins
    #threshold = [min, max]
    #title
    # -- add min, max for X and Y axis
    
    vect = np.ndarray.flatten(arr)
    
    if threshold:
        vect = vect[vect>threshold[0]]; vect = vect[vect<threshold[1]]
    if not title:
        title = "histogram"
    plt.hist(vect, bins = bins)
    plt.title(title) 
    plt.show()
    
    return

def unq(list_full):
    
    list_unq = set(list_full)
    list_unq = list(list_unq)
    
    return list_unq

def head(list_inp, len_head = 5):
    
    print(*list_inp[:len_head], sep = '\n')
    
    return
