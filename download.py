import glob
import os
import argparse

import imghdr
import requests
import shutil

from multiprocessing import Pool


def download(input_dir):
    bad = 0
    good = 0

    files = glob.glob(input_dir+'/**/*.urls', recursive=True)
    
    for file in files:
        dirname, subdirname = os.path.split(file)
        dirname = os.path.join(dirname.split('/')[-1]+'_images',subdirname.split('.')[0])
        os.makedirs(dirname, exist_ok=True)
        with open(file, 'r') as f:
            for line in f.readlines():
                
                filename = line.split('/')[-1][:-1]
                filename = filename.split('.')[0]

                if len(filename) > 254:
                    filename = filename[-254:]
                print(line[:-1])

                if ',' in filename:
                    filename.replace(',','')
                
                try:
                    r = requests.get(line[:-1], stream=True, timeout=15)
                except:
                    continue
                # print(r.status_code)
                

                if r.status_code == 200:
                    r.raw.decode_content = True
                    out_file_path = os.path.join(dirname, filename)
                    
                    with open(out_file_path,'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    
                    file_type = imghdr.what(out_file_path)

                    if file_type == None or file_type == 'gif':
                        os.remove(out_file_path)
                        print('Image Couldn\'t be retreived')
                    else:
                        os.rename(out_file_path, out_file_path+'.'+file_type)

                    print('Image sucessfully Downloaded: ',filename)
                    good+=1
                else:
                    print('Image Couldn\'t be retreived')
                    bad+=1
    print('___________')
    print(good, bad)
    print('___________')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = None, type = str)
    args = parser.parse_args()
    
    download(args.input_dir)