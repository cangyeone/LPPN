import pickle 
import matplotlib.pyplot as plt 
import numpy as np 
import obspy 
import os 
import datetime 
import math 
import scipy.signal as signal 
import time 
import multiprocessing 
plt.switch_backend('agg')
import h5py 
def _csv_file(file_dir):
    if os.path.exists("ckpt/phasedata.pkl")==True:
        outfile = open("ckpt/phasedata.pkl", "rb")
        phase_data = pickle.load(outfile) 
        outfile.close()
        train_data, test_data = phase_data 
    else:
        phasefile = open(os.path.join(file_dir, "metadata_11_13_19.csv"), "r")
        line1 = phasefile.readline()
        phase_data = [] 
        for itr in phasefile.readlines():
            line = itr.strip().split(",")
            if len(line)!=35:continue
            #print([a for a in zip(range(len(line)), line)], len(line))
            try:
                if "noise" == line[33]:
                    name = line[-1].strip()
                    pt = -100
                    st = -100 
                    snr = -100
                else:
                    name = line[-1].strip()
                    lon = float(line[4])
                    lat = float(line[3]) 
                    if lon < -126 or lon > -118:continue 
                    if lat < 34 or lat > 41:continue  
                    lon = float(line[17])
                    lat = float(line[16]) 
                    if lon < -126 or lon > -118:continue 
                    if lat < 34 or lat > 41:continue
                    pt = float(line[6])
                    st = float(line[10]) 
                    snr = [float(a) for a in line[30][1:-1].split(" ") if len(a)>3]
            except:
                print([a for a in zip(range(len(line)), line)], len(line))
            phase_data.append([name, pt, st, snr]) 
        np.random.shuffle(phase_data) 
        print("Length", len(phase_data))
        length = len(phase_data) 
        train_idx = int(length * 0.8) 
        train_data = phase_data[:train_idx] 
        test_data = phase_data[train_idx:]
        datas = [train_data, test_data]
        outfile = open("ckpt/phasedata.pkl", "wb")
        pickle.dump(datas, outfile)
        phasefile.close() 
    return train_data, test_data 
import time 
class Data():
    def __init__(self, file_dir="data", batch_size=32, n_thread=1, strides=8, n_length=3000):
        self.batch_size = batch_size 
        self.freq = 100
        self.time_length = 4.0
        self.rage = 0.05
        self.file_name = file_dir
        self.n_length = n_length
        self.n_stride = strides
        self.train_label, self.test_label = _csv_file(file_dir)
        self.queue = multiprocessing.Queue(maxsize=10) 
        self.in_queue = multiprocessing.Queue(maxsize=10) 
        self.out_queue = multiprocessing.Queue(maxsize=10)
        self.batch_queue = multiprocessing.Queue(maxsize=10) 
        self.data_thread = []
        multiprocessing.Process(target=self.batch_data_input, args=(self.in_queue, self.train_label)).start() 
        for i in range(6):
            multiprocessing.Process(target=self.process_multithread, args=(self.in_queue, self.out_queue)).start()
        multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)).start() 
    def batch_data_input(self, in_queue, train_label):
        h5files = h5py.File(os.path.join(self.file_name, "waveforms_11_13_19.hdf5"), "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            for key, pt, st, snr in train_label:
                if pt == -100 and st == -100:
                    waves = noise[key][:, :]
                    snr = [1, 1, 1]
                else:
                    waves = earthquake[key][:, :]
                in_queue.put([waves, pt, st, snr])
    def process_multithread(self, in_queue, out_queue):
        """CCC"""
        while True:
            temp = in_queue.get()
            wave, pt, st, snr = temp 
            if pt == -100 and st == -100:
                n_legnth, n_channel = wave.shape
                n_stride = self.n_stride 
                n_legnth_s = self.n_length // n_stride
                n_range = 50 
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                #
                p_idx = 500
                s_idx = 500
                logit = -np.ones([1, n_legnth_s, 2])
                wave = np.reshape(wave, [1, -1, 3])[:, :self.n_length, :]
                n_stride = self.n_stride
                
            else:
                d_legnth, n_channel = wave.shape
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                
                p_idx = int(pt)
                s_idx = int(st)
                begin = np.random.randint(0, max(p_idx, 2)) 
                
                begin = max(0, begin)
                p_idx = p_idx - begin 
                s_idx = s_idx - begin 

                n_stride = self.n_stride
                n_legnth_s = self.n_length // n_stride
                p_idx_s = p_idx // n_stride 
                s_idx_s = s_idx // n_stride
                logit = -np.ones([1, n_legnth_s, 2]) 
                if p_idx_s > 0:
                    logit[0, p_idx_s:p_idx_s+1, 0] = 0 
                    logit[0, p_idx_s:p_idx_s+1, 1] = p_idx % n_stride  
                if s_idx_s > 0:
                    logit[0, s_idx_s:s_idx_s+1, 0] = 1  
                    logit[0, s_idx_s:s_idx_s+1, 1] = s_idx % n_stride  
                wave = np.reshape(wave, [1, -1, 3])[:, begin:begin+self.n_length, :] 

            out_queue.put([wave, logit, snr, [p_idx, s_idx]]) 
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            a1, a2, a3, a4 = [], [], [], []
            for itr in range(self.batch_size):
                wave, labels, snr, tm = out_queue.get() 
                a1.append(wave)
                a2.append(labels) 
                a3.append(snr) 
                a4.append(tm) 

            
            a1 = np.concatenate(a1, axis=0)
            a2 = np.concatenate(a2, axis=0)
            batch_queue.put([a1, a2, a3, a4])
    def batch_data(self):
        a1, a2, a3, a4 = self.batch_queue.get() 
        return a1, a2, a3, a4

import time 
class DataTest():
    def __init__(self, file_dir="data", batch_size=32, n_thread=1, strides=8, n_length=3000):
        self.batch_size = batch_size 
        self.freq = 100
        self.time_length = 4.0
        self.rage = 0.05
        self.n_length = n_length
        self.n_stride = strides
        self.file_name = file_dir 
        self.train_label, self.test_label = _csv_file(file_dir)
        self.queue = multiprocessing.Queue(maxsize=10) 
        self.in_queue = multiprocessing.Queue(maxsize=10) 
        self.out_queue = multiprocessing.Queue(maxsize=10)
        self.batch_queue = multiprocessing.Queue(maxsize=10) 
        self.data_thread = []
        multiprocessing.Process(target=self.batch_data_input, args=(self.in_queue, self.test_label)).start() 
        for i in range(6):
            multiprocessing.Process(target=self.process_multithread, args=(self.in_queue, self.out_queue)).start()
        multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)).start() 
    def batch_data_input(self, in_queue, train_label):
        h5files = h5py.File(os.path.join(self.file_name, "waveforms_11_13_19.hdf5"), "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            for key, pt, st, snr in train_label:
                if pt == -100 and st == -100:
                    waves = noise[key][:, :]
                    snr = [1, 1, 1]
                else:
                    waves = earthquake[key][:, :]
                in_queue.put([waves, pt, st, snr])
    def process_multithread(self, in_queue, out_queue):
        """CCC"""
        while True:
            temp = in_queue.get()
            wave, pt, st, snr = temp 
            if pt == -100 and st == -100:
                continue 
                n_legnth, n_channel = wave.shape
                n_stride = self.n_stride 
                n_legnth_s = self.n_length // n_stride
                n_range = 50 
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                #
                p_idx = -500
                s_idx = -500
                logit = -np.ones([1, n_legnth_s, 2])
                wave = np.reshape(wave, [1, -1, 3])[:, :self.n_length, :]
                n_stride = self.n_stride
                
            else:
                d_legnth, n_channel = wave.shape
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                
                p_idx = int(pt)
                s_idx = int(st)
                begin = np.random.randint(0, s_idx)
                begin = 0 
                p_idx = p_idx - begin 
                s_idx = s_idx - begin 

                wave = np.reshape(wave, [1, -1, 3])[:, begin:begin+self.n_length, :] 
            if p_idx < 0:
                isnoise = True 
            else:
                isnoise = False 
            out_queue.put([wave, isnoise, snr, [p_idx, s_idx]]) 
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            a1, a2, a3, a4 = [], [], [], []
            for itr in range(self.batch_size):
                wave, labels, snr, tm = out_queue.get() 
                a1.append(wave)
                a2.append(labels) 
                a3.append(snr) 
                a4.append(tm) 
            
            a1 = np.concatenate(a1, axis=0)
            batch_queue.put([a1, a2, a3, a4])
    def batch_data(self):
        a1, a2, a3, a4 = self.batch_queue.get() 
        return a1, a2, a3, a4


import time 
class DataTestForShow():
    def __init__(self, batch_size=32, n_thread=1, strides=8, n_length=3000):
        self.batch_size = batch_size 
        self.freq = 100
        self.time_length = 4.0
        self.rage = 0.05
        self.n_length = n_length
        self.n_stride = strides
        self.train_label, self.test_label = _csv_file()
        self.queue = multiprocessing.Queue(maxsize=10) 
        self.in_queue = multiprocessing.Queue(maxsize=10) 
        self.out_queue = multiprocessing.Queue(maxsize=10)
        self.batch_queue = multiprocessing.Queue(maxsize=10) 
        self.data_thread = []
        multiprocessing.Process(target=self.batch_data_input, args=(self.in_queue, self.test_label)).start() 
        multiprocessing.Process(target=self.process_multithread, args=(self.in_queue, self.out_queue)).start()
        multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)).start() 
    def batch_data_input(self, in_queue, train_label):
        h5files = h5py.File("data/waveforms_11_13_19.hdf5", "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            for key, pt, st, snr in train_label:
                if pt == -100 and st == -100:
                    waves = noise[key][:, :]
                    snr = [1, 1, 1]
                else:
                    waves = earthquake[key][:, :]
                in_queue.put([waves, pt, st, snr])
    def process_multithread(self, in_queue, out_queue):
        """CCC"""
        while True:
            temp = in_queue.get()
            wave, pt, st, snr = temp 
            if pt == -100 and st == -100:
                continue 
                n_legnth, n_channel = wave.shape
                n_stride = self.n_stride 
                n_legnth_s = self.n_length // n_stride
                n_range = 50 
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                #
                p_idx = -500
                s_idx = -500
                logit = -np.ones([1, n_legnth_s, 2])
                wave = np.reshape(wave, [1, -1, 3])[:, :self.n_length, :]
                n_stride = self.n_stride
                
            else:
                d_legnth, n_channel = wave.shape
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                if np.abs(pt-st)/100>10:continue 
                for i in range(self.batch_size):
                    begin = i 
                    p_idx = int(pt)
                    s_idx = int(st)
                    p_idx = p_idx - begin 
                    s_idx = s_idx - begin 

                    w = np.reshape(wave, [1, -1, 3])[:, begin:begin+self.n_length, :] 
                    out_queue.put([w, False, snr, [p_idx, s_idx]]) 
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            a1, a2, a3, a4 = [], [], [], []
            for itr in range(self.batch_size):
                wave, labels, snr, tm = out_queue.get() 
                print(wave.shape)
                a1.append(wave)
                a2.append(labels) 
                a3.append(snr) 
                a4.append(tm) 
            
            a1 = np.concatenate(a1, axis=0)
            batch_queue.put([a1, a2, a3, a4])
    def batch_data(self):
        a1, a2, a3, a4 = self.batch_queue.get() 
        return a1, a2, a3, a4

class DataTestForShow2():
    def __init__(self, batch_size=32, n_thread=1, strides=8, n_length=3000):
        self.batch_size = batch_size 
        self.freq = 100
        self.time_length = 4.0
        self.rage = 0.05
        self.n_length = n_length
        self.n_stride = strides
        self.train_label, self.test_label = _csv_file()
        self.queue = multiprocessing.Queue(maxsize=10) 
        self.in_queue = multiprocessing.Queue(maxsize=10) 
        self.out_queue = multiprocessing.Queue(maxsize=10)
        self.batch_queue = multiprocessing.Queue(maxsize=10) 
        self.data_thread = []
        multiprocessing.Process(target=self.batch_data_input, args=(self.in_queue, self.test_label)).start() 
        multiprocessing.Process(target=self.process_multithread, args=(self.in_queue, self.out_queue)).start()
        multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)).start() 
    def batch_data_input(self, in_queue, train_label):
        h5files = h5py.File("data/waveforms_11_13_19.hdf5", "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            for key, pt, st, snr in train_label:
                if pt == -100 and st == -100:
                    waves = noise[key][:, :]
                    snr = [1, 1, 1]
                else:
                    waves = earthquake[key][:, :]
                in_queue.put([waves, pt, st, snr])
    def process_multithread(self, in_queue, out_queue):
        """CCC"""
        while True:
            temp = in_queue.get()
            wave, pt, st, snr = temp 
            if pt == -100 and st == -100:
                continue 
                n_legnth, n_channel = wave.shape
                n_stride = self.n_stride 
                n_legnth_s = self.n_length // n_stride
                n_range = 50 
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                #
                p_idx = -500
                s_idx = -500
                logit = -np.ones([1, n_legnth_s, 2])
                wave = np.reshape(wave, [1, -1, 3])[:, :self.n_length, :]
                n_stride = self.n_stride
                
            else:
                d_legnth, n_channel = wave.shape
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                #if np.abs(pt-st)/100>10:continue 
                begin = 0 
                p_idx = int(pt)
                s_idx = int(st)
                p_idx = p_idx - begin 
                s_idx = s_idx - begin 

                w = np.reshape(wave, [1, -1, 3])[:, begin:begin+self.n_length, :] 
                out_queue.put([w, False, snr, [p_idx, s_idx]]) 
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            a1, a2, a3, a4 = [], [], [], []
            for itr in range(self.batch_size):
                wave, labels, snr, tm = out_queue.get() 
                #print(wave.shape)
                a1.append(wave)
                a2.append(labels) 
                a3.append(snr) 
                a4.append(tm) 
            
            a1 = np.concatenate(a1, axis=0)
            batch_queue.put([a1, a2, a3, a4])
    def batch_data(self):
        a1, a2, a3, a4 = self.batch_queue.get() 
        return a1, a2, a3, a4
class DataTestForShow3():
    def __init__(self, batch_size=32, n_thread=1, strides=8, n_length=3000):
        self.batch_size = batch_size 
        self.freq = 100
        self.time_length = 4.0
        self.rage = 0.05
        self.n_length = n_length
        self.n_stride = strides
        self.train_label, self.test_label = _csv_file()
        self.queue = multiprocessing.Queue(maxsize=10) 
        self.in_queue = multiprocessing.Queue(maxsize=10) 
        self.out_queue = multiprocessing.Queue(maxsize=10)
        self.batch_queue = multiprocessing.Queue(maxsize=10) 
        self.data_thread = []
        multiprocessing.Process(target=self.batch_data_input, args=(self.in_queue, self.test_label)).start() 
        multiprocessing.Process(target=self.process_multithread, args=(self.in_queue, self.out_queue)).start()
        multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)).start() 
    def batch_data_input(self, in_queue, train_label):
        h5files = h5py.File("data/waveforms_11_13_19.hdf5", "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            for key, pt, st, snr in train_label:
                if pt == -100 and st == -100:
                    waves = noise[key][:, :]
                    snr = [1, 1, 1]
                else:
                    waves = earthquake[key][:, :]
                in_queue.put([waves, pt, st, snr])
    def process_multithread(self, in_queue, out_queue):
        """CCC"""
        count = 0 
        while True:
            temp = in_queue.get()
            wave, pt, st, snr = temp 
            if pt == -100 and st == -100:
                continue 
                n_legnth, n_channel = wave.shape
                n_stride = self.n_stride 
                n_legnth_s = self.n_length // n_stride
                n_range = 50 
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                #
                p_idx = -500
                s_idx = -500
                logit = -np.ones([1, n_legnth_s, 2])
                wave = np.reshape(wave, [1, -1, 3])[:, :self.n_length, :]
                n_stride = self.n_stride
            else:
                d_legnth, n_channel = wave.shape
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                if np.abs(pt-st)/100>10:continue 
                begin = 0 
                p_idx = int(pt)
                s_idx = int(st)
                p_idx = p_idx - begin 
                s_idx = s_idx - begin 
                snr = np.mean(snr)
                w = np.reshape(wave, [1, -1, 3])[:, begin:begin+self.n_length, :] 
                #print(snr)
                if snr < 10  and count % 2 == 1:
                    count += 1
                    out_queue.put([w, False, snr, [p_idx, s_idx]])
                if snr > 50 and count % 2 == 0:
                    count += 1 
                    out_queue.put([w, False, snr, [p_idx, s_idx]]) 
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            a1, a2, a3, a4 = [], [], [], []
            for itr in range(self.batch_size):
                wave, labels, snr, tm = out_queue.get() 
                print(wave.shape)
                a1.append(wave)
                a2.append(labels) 
                a3.append(snr) 
                a4.append(tm) 
            
            a1 = np.concatenate(a1, axis=0)
            batch_queue.put([a1, a2, a3, a4])
    def batch_data(self):
        a1, a2, a3, a4 = self.batch_queue.get() 
        return a1, a2, a3, a4

if __name__=="__main__":
    tool = PhaseData() 
    for step in range(20):
        a1, a2, a3, a4, a5, a6 = tool.next_batch() 
        plt.cla()
        plt.clf()
        plt.plot(a1[0, :, 0]) 
        plt.plot(a2[0, :])
        plt.plot(a3[0, :, 0])
        plt.savefig(f"datafig/a-{step}.png")
        print(f"datafig/a-{step}.png")


