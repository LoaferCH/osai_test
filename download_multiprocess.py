import glob
import os
import argparse
import requests
import imghdr
import shutil

from multiprocessing import Pool

def download_from_file(input_file):
    # create _images directory and subfolder with sport
    dirname, subdirname = os.path.split(input_file)
    subdirname = subdirname.split('.')[0]
    dirname = os.path.join(dirname.split('/')[-1]+'_images',subdirname)
    os.makedirs(dirname, exist_ok=True)

    downloaded_files = []

    with open(input_file, 'r') as f:
        for line in f.readlines():
            # getting filename
            filename = line.split('/')[-1][:-1]
            filename = filename.split('.')[0]+'.jpg'
            
            filename = filename.replace(',','')

            if len(filename) > 254:
                filename = filename[-254:]

            # request
            try:
                r = requests.get(line[:-1], stream=True, timeout=10)
            except:
                continue
            
            if r.status_code == 200:
                r.raw.decode_content = True
                out_file_path = os.path.join(dirname, filename)

                # saving bytes to file
                with open(out_file_path,'wb') as f:
                    shutil.copyfileobj(r.raw, f)

                # checking if the file is valid
                file_type = imghdr.what(out_file_path)

                # if not valid or gif, delete it and continue
                if file_type == None or file_type == 'gif':
                    os.remove(out_file_path)
                    print('Image Couldn\'t be retreived')
                    continue
                # if valid, change extension
                else:
                    ext_file_path = out_file_path[:-3]+file_type
                    os.rename(out_file_path, ext_file_path)

                downloaded_files.append(os.path.abspath(ext_file_path))
                print('Image sucessfully Downloaded: ',filename)
            else:
                print('Image Couldn\'t be retreived: ', filename)

    # write names of valid files to .txt
    with open(os.path.join(dirname, subdirname + '.txt'), 'w') as f:
        for file in downloaded_files:
            f.write(file + '\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = None, type = str)
    args = parser.parse_args()
    
    # get all .urls files
    files = glob.glob(args.input_dir+'/**/*.urls', recursive=True)

    with Pool(len(files)) as p:
        print(p.map(download_from_file, files))

    # find all .txt files and collect to final csv
    new_dir= args.input_dir.split('/')[-1]+'_images'
    txts = glob.glob(new_dir+'/**/*.txt', recursive=True)

    with open(os.path.join(new_dir, 'downloaded.csv'), 'w') as f:
        f.write('img_path,label,class_name\n')
        for i, txt in enumerate(txts):
            with open(txt, 'r') as t:
                for line in t:
                    f.write('%s,%d,%s\n'%(line[:-1], i, os.path.split(txt)[1].split('.')[0]))

        
