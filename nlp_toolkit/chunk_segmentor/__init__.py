import os
import sys
import glob
import pickle
import socket
from pathlib import Path
from datetime import datetime

INIT_PATH = os.path.realpath(__file__)
STATIC_ROOT = os.path.dirname(INIT_PATH)
DATA_PATH = Path(STATIC_ROOT) / 'data'
MD5_FILE_PATH = DATA_PATH / 'model_data.md5'
UPDATE_TAG_PATH = DATA_PATH / 'last_update.pkl'
UPDATE_INIT_PATH = DATA_PATH / 'init_update.txt'
MD5_HDFS_PATH = '/user/xxxx/chunk_segmentor/model_data.md5'
MODEL_HDFS_PATH = '/user/xxxx/chunk_segmentor/model_data.zip'
USER_NAME = 'xxxx'
PASSWORD = 'xxxxx'
FTP_PATH_1 = 'ftp://xxx.xxx.xx.xx:xx/chunk_segmentor'
FTP_PATH_2 = 'ftp://xxx.xxx.xx.xx:xx/chunk_segmentor'
IP = socket.gethostbyname(socket.gethostname())


def check_version():
    if MD5_FILE_PATH.exists():
        src = get_data_md5()
        if src:
            flag = update(src)
            if not flag:
                print('模型和数据更新失败！')
            else:
                for fname in glob.glob('model_data.md5*'):
                    os.remove(fname)
        else:
            print('拉取md5文件失败！')
    else:
        print("这是第一次启动Chunk分词器。 请耐心等待片刻至数据和模型下载完成。")
        flag = download()
        if flag:
            current_time = datetime.now()
            init_update_time = str(os.path.getctime(INIT_PATH))
            pickle.dump(current_time, open(UPDATE_TAG_PATH, 'wb'))
            with open(UPDATE_INIT_PATH, 'w') as fout:
                fout.write(init_update_time)
        else:
            print('请寻找一台有hadoop或者能访问ftp://xxx.xxx.xx.xx:xx或者ftp://xxx.xxx.xx.xx:xx的机器')


def write_config(config_path, new_root_path):
    content = []
    with open(config_path, encoding='utf8') as f:
        for line in f:
            if line.startswith('root'):
                line = 'root={}{}'.format(new_root_path, os.linesep)
            content.append(line)
    with open(config_path, 'w', encoding='utf8') as f:
        f.writelines(content)


def download():
    # 下载数据文件
    ret1 = -1
    ret2 = -1
    for fname in glob.glob('model_data.md5*'):
        os.remove(fname)
    for fname in glob.glob('model_data.zip*'):
        os.remove(fname)

    if not IP.startswith('127'):
        print('尝试从ftp://xxx.xxx.xx.xx:xx获取数据')
        ret2 = os.system('wget -q --timeout=2 --tries=1 --ftp-user=%s --ftp-password=%s %s/model_data.md5' %
                        (USER_NAME, PASSWORD, FTP_PATH_1))
        if ret2 == 0:
            ret1 = os.system('wget --ftp-user=%s --ftp-password=%s %s/model_data.zip' %
                            (USER_NAME, PASSWORD, FTP_PATH_1))
        if ret1 != 0:
            print('尝试从hdfs上拉取数据，大约20-30s')
            ret1 = os.system('hadoop fs -get %s' % MODEL_HDFS_PATH)
            ret2 = os.system('hadoop fs -get %s' % MD5_HDFS_PATH)
    else:
        print('尝试从ftp://xxx.xxx.xx.xx:xx获取数据')
        ret2 = os.system('wget -q --timeout=2 --tries=1 --ftp-user=%s --ftp-password=%s %s/model_data.md5' %
                        (USER_NAME, PASSWORD, FTP_PATH_2))
        if ret2 == 0:
            ret1 = os.system('wget --ftp-user=%s --ftp-password=%s %s/model_data.zip' %
                            (USER_NAME, PASSWORD, FTP_PATH_2))
    if ret1 != 0 or ret2 != 0:
        return False
    if ret1 == 0 and ret2 == 0:
        os.system('unzip -q model_data.zip')
        os.system('cp -r model_data/data %s' % STATIC_ROOT)
        os.system('cp -f model_data/best_model.txt %s' % DATA_PATH)
        os.system('cp -f model_data.md5 %s' % DATA_PATH)
        os.system('rm -r model_data')
        os.system('rm model_data.md5*')
        os.system('rm model_data.zip*')
        print('数据和模型下载成功')
        return True


def get_data_md5():
    for fname in glob.glob('model_data.md5*'):
        os.remove(fname)
    ret = -1

    if not IP.startswith('127'):
        ret = os.system('wget -q --timeout=2 --tries=1 --ftp-user=%s --ftp-password=%s %s/model_data.md5' %
                        (USER_NAME, PASSWORD, FTP_PATH_1))
        if ret == 0:
            src = 'ftp1'
        else:
            ret = os.system('hadoop fs -get /user/kdd_wangyilei/chunk_segmentor/model_data.md5')
            if ret == 0:
                src = 'hdfs'
    else:
        ret = os.system('wget -q --timeout=2 --tries=1 --ftp-user=%s --ftp-password=%s %s/model_data.md5' % 
                        (USER_NAME, PASSWORD, FTP_PATH_2))
        if ret == 0:
            src = 'ftp2'
    if ret != 0:
        print('请寻找一台有hadoop或者能访问ftp://xxx.xxx.xx.xx:xx或者ftp://xxx.xxx.xx.xx:xx的机器')
        return None
    else:
        return src


def update(src):
    with open(MD5_FILE_PATH, 'rb') as f:
        current_data_md5 = f.readlines()[0].strip()
    with open('model_data.md5', 'rb') as f:
        latest_data_md5 = f.readlines()[0].strip()
    try:
        if current_data_md5 != latest_data_md5:
            x = input('发现新的数据和模型？是否决定下载更新？ Yes/No?')
            if x in ['Yes', 'Y', 'y', 'YES', '1', 1, 'yes'] or x == '':
                flag = update_data(src)
                if flag:
                    print('模型和字典数据已更新到最新版本')
                    return True
                else:
                    return False
            else:
                print('希望您下次来更新数据！')
                return True
        else:
            return True
    except:
        return False


def update_data(src):
    try:
        for fname in glob.glob('model_data.zip*'):
            os.remove(fname)
        if src == 'hdfs':
            print('尝试从hdfs上拉取数据，大约20-30s')
            os.system('hadoop fs -get /user/xxxxx/chunk_segmentor/model_data.zip')
        elif src == 'ftp1':
            print('尝试从ftp://xxx.xxx.xx.xx:xx获取数据')
            os.system('wget --ftp-user=%s --ftp-password=%s %s/model_data.zip' % (USER_NAME, PASSWORD, FTP_PATH_1))
        elif src == 'ftp2':
            print('尝试从ftp://xxx.xxx.xx.xx:xx获取数据')
            os.system('wget --ftp-user=%s --ftp-password=%s %s/model_data.zip' % (USER_NAME, PASSWORD, FTP_PATH_2))

        os.system('unzip -q model_data.zip')
        os.system('rm -r %s' % DATA_PATH)
        os.system('cp -r model_data/data %s' % STATIC_ROOT)
        os.system('cp -f model_data/best_model.txt %s' % DATA_PATH)
        os.system('cp -f model_data.md5 %s' % DATA_PATH)
        os.system('rm -r model_data')
        os.system('rm model_data.md5*')
        os.system('rm model_data.zip*')
        return True
    except:
        return False


check_version()
from .segment import Chunk_Segmentor
