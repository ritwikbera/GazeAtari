import os
import glob
import zipfile
import tarfile

def unzip(dir_path = 'dataset/alien'):
    dir_path = os.getcwd()+'/'+dir_path
    for path, dir_list, file_list in os.walk(dir_path):
        print(path)
        print(file_list)
        for file_name in file_list:
            if file_name.endswith('.tar.bz2'):
                abs_file_path = os.path.join(path, file_name)

                parent_path = os.path.split(abs_file_path)[0]
                output_path = os.path.join(parent_path, 'extracted')

                print('Unzipping ', file_name)
                zip_obj = tarfile.open(abs_file_path, 'r:bz2')
                zip_obj.extractall(output_path)
                zip_obj.close()

if __name__=='__main__':
    unzip()