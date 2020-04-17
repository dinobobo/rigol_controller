##################################################################
#
# Interface for the AWG controlling 2D-AOD stirring beam
#
##################################################################
import os
import numpy as np
from matplotlib import pyplot as plt
import pyvisa
import ast

#==============================================================================
class AODController():
    """
    This class implements an interface to a Rigol 2 channel AWG, using
    pyVISA. It is primarily intended to allow upolading of arbitrary waveforms
    used to drive a 2D acousto-optic deflector
    """
    MAX_SAMPLES = 2**14
       
    def __init__(self, instr_ID):
        self.instr = pyvisa.ResourceManager().open_resource(instr_ID)
        self.instr.read_termination = '\n'
        self.write('*IDN?')
        print("Connected to %s" % self.read())     
      
    def write(self, cmd_str):
        try:
            msg = self.instr.write(cmd_str)
            return msg
        except Exception as exc:
            print(exc)
            
    def query(self, cmd_str):
        try:
            msg = self.instr.query(cmd_str)
            return msg
        except Exception as exc:
            print(exc)
        
    def read(self):
        try:
            response = self.instr.read()
        except Exception as exc:
            response = exc
            print(exc)
        return response
        
    def chdir(self, path):
        self.write(':MMEMory:CDIRectory "%s"' % path)
    
    def load(self, fname_raf):
        msg = self.write(':MMEMory:LOAD "%s"' % fname_raf)
        if msg[0] != 29:
            print("Warning: can't load file!")
    def copy_data(self, fromchan, tochan):
        self.write(':SYSTem:CWCopy %s,%s' % (fromchan, tochan))
    
    def period(self, period):
        self.write(':SOURce1:PERiod %s' % period)
        self.write(':SOURce2:PERiod %s' % period)
        
    def amplitude(self, x_amp, y_amp):
        self._x_amplitude(x_amp)
        self._y_amplitude(y_amp)
    
    def _x_amplitude(self, amp):
        self.write('SOURce1:VOLT %.6f' % amp)
    
    def _y_amplitude(self, amp):
        self.write('SOURce2:VOLT %.6f' % amp)    

    def offset(self, x_offset, y_offset):
        self._x_offset(x_offset)
        self._y_offset(y_offset)
    
    def _x_offset(self, offset):
        self.write('SOURce1:VOLTage:OFFSet %.6f' % offset)
    
    def _y_offset(self, offset):
        self.write('SOURce2:VOLTage:OFFSet %.6f' % offset)
        
    def apply_values(self, freq, xamp, yamp, xoffset, yoffset):
        self.write('SOURce1:APPLy:BURSt %f,%f,%f,0' % (freq, xamp, xoffset))
        self.write('SOURce2:APPLy:BURSt %f,%f,%f,0' % (freq, yamp, yoffset))
        
    def configure_burst(self):
        for chan in ['SOURce1:BURSt:', 'SOURce2:BURSt:']:
            self.write(chan + 'MODE TRIGgered')
            self.write(chan + 'TRIGger:SLOPe POSitive')
            self.write(chan + 'TRIGger:SOURce EXTernal')
            
    def configure_output(self):
        for chan in [':OUTPut1:', ':OUTPut2:']:
            self.write(chan + 'IMPedance 50')
            self.write(chan + 'LOAD 50')
            self.write(chan + 'STATe ON')
                
    def load_to_CHs(self, fname1, fname2):
        """Loading files to both channels, make sure CH1 is active while loading"""
        self.write(':SOURce1:PHASe:INITiate')  # write (not query) in to a specific channel activates it
        self.load(fname1)
        self.write(':SOURce2:PHASe:INITiate')
        self.load(fname2)
        
    def load_and_config(self, dir_name, file1 = "xvalues.RAF", file2 = "yvalues.RAF"):
        dir_name = os.path.join('D:\\', dir_name)
        self.write(':MMEMory:CDIRectory "%s"' % dir_name)
        self.load_to_CHs(file1, file2)
        self.write(':SOURce1:PHASe:INITiate')
        self.write(':SOURce2:PHASe:INITiate')
        self.write(':SOURce1:BURSt ON')
        self.write(':SOURce2:BURSt ON')
        
    def get_disc(self):
        """Find the name of the newwork drive disc name"""
        return os.getcwd().split('\\')[0]
        
        
     
class waveform():
    def __init__(self,form, xdata = None, ydata = None, total_time = 1):
        self.form = form
        self.form['start_phase'] *= np.pi*2
        self.form['stir_frequency'] *= np.pi*2
        self.accel_time = self.form['stir_frequency']/self.form['acceleration']
        self.total_time = self.form['initial_hold'] + self.accel_time + self.form['end_hold']
              
        
    def to_str(self,form = None):
        if form == None:
            form = self.form
        form_str = 'ini_phase=%0.1ftwopi_ini_hold=%0.2fs_acc=%0.2frad_freq=%0.1fHz_end_hold=%0.2fs_scan_freq=%0.0fHz_scan_amp=%0.3f' % (form['start_phase']/np.pi/2,
        form['initial_hold'], form['acceleration'], form['stir_frequency']/np.pi/2, form['end_hold'], form['scan_frequency'], form['scan_amp'])
        return form_str
        
    
    def thetafunc(self, t):
        if t < self.form['initial_hold']:
            return self.form['start_phase']
        elif self.form['initial_hold'] < t < self.form['initial_hold'] + self.accel_time:
            return self.form['start_phase'] + 0.5 * self.form['acceleration'] * (t - self.form['initial_hold'])**2
        else: # t > t_hold_0 + accel_time
            return self.form['start_phase'] + 0.5 * self.form['acceleration'] * self.accel_time**2 + self.form['stir_frequency'] * (t - self.form['initial_hold'] - self.accel_time)

    def barrier_scan(self, t, r_avg = 0.3):
        return r_avg + self.form['scan_amp'] * np.sin(2 * np.pi * t * self.form['scan_frequency'])
    
    def polar_to_rect_waveforms(self, rvalues, thetavalues):
        self.xvalues = rvalues * np.cos(thetavalues)
        self.yvalues = rvalues * np.sin(thetavalues)
        
    def to_raf_obselete(self, values):
        """Normalize the the values for saving to RAF, obselete already"""
        # Shift and convert the values
        values = values - values.min()
        values = ((values/values.max()) * int("3fff", 16)).astype('int16')
        # Write the signal as binary.
        return values

    def to_raf(self, values, bit_depth = 14):
        """Convert the signal (already the correct length) to a RAF file."""
        # Shift and convert the values
        values = values + 0.5
        values = (values * int(hex(2**bit_depth - 1), 16)).astype('int16')
        return values
    
    def get_disc(self):
        """Find the name of the newwork drive disc name"""
        return os.getcwd().split('\\')[0]
    
    def check_sample_rate(self, max_sample):
        if self.form['scan_frequency'] != 0 and max_sample/self.total_time/self.form['scan_frequency'] < 6:       
           print("Warning, radial scan frequency too high, not enough samples!")
        
    
    def save_waveform(self):
        self.x_rafdata = self.to_raf(self.xvalues)
        self.y_rafdata = self.to_raf(self.yvalues) 
        self.data_path = os.path.join(self.get_disc(), '\\Wright Lab Code\\Rigol\\RAF_files', self.to_str())       
        if os.path.exists(self.data_path) != True:
            os.mkdir(self.data_path)
        xfile = os.path.join(self.data_path, 'xvalues.RAF')
        fp = open(xfile, "w")
        self.x_rafdata.tofile(fp)
        fp.close()
        
        yfile = os.path.join(self.data_path, 'yvalues.RAF')
        fp = open(yfile, "w")
        self.y_rafdata.tofile(fp)
        fp.close()
        
        
        
      
        
        
def awg_initialize(awg):
    awg.configure_output()
    awg.configure_burst()
    awg.offset(-0.02,0)
    awg.amplitude(0.67, 0.61)

def gen_wf(form):
    wf = waveform(form)
    wf.times = np.linspace(0, wf.total_time, AODController.MAX_SAMPLES)
    wf.thetavalues = [wf.thetafunc(t) for t in wf.times] 
    wf.rvalues = [wf.barrier_scan(t = t) for t in wf.times] 
    wf.polar_to_rect_waveforms(wf.rvalues, wf.thetavalues)
    wf.check_sample_rate(AODController.MAX_SAMPLES)
    wf.save_waveform()
    return wf

def debug_plot(times, thetavalues, rvalues, xvalues, yvalues, x_rafdata, y_rafdata):
    plt.figure(1, clear=True)
    plt.subplot(2,1,1)
    plt.plot(times, thetavalues)
       
    plt.subplot(2,1,2)
    plt.plot(times, rvalues, 'r-')
    plt.plot(times, rvalues, 'k.')
       
    plt.figure(2, clear=True)
    plt.subplot(2,1,1)
    plt.plot(times, xvalues)
    plt.subplot(2,1,2)
    plt.plot(times, yvalues)
    
    plt.figure(3, clear=True)
    plt.scatter(xvalues, yvalues)
    
    
    plt.figure(4, clear=True)
    plt.scatter(x_rafdata, y_rafdata)

def display_files(path):
    os.listdir(path)
    file_list = []
    with open(os.path.join(path,'file_list.txt'),'w') as f:
        for item in os.listdir(path):
            file_list.append(item)
            f.write("%s\n" % item)
    return file_list

def to_vals(file_str):
    vals = []
    str_arr = file_str.split('=')[1:]
    for i in str_arr:
        item = ''
        for j in i:
            if j.isdigit() or j == '.':
                item += j
        vals.append(ast.literal_eval(item))
    return vals
    
    
        
#==============================================================================       

        
if __name__== "__main__":
    gen_wave = False
    file_num = 5
    ###################Generate Wave form###########################
    if gen_wave == True:
        keys = ['start_phase','initial_hold','acceleration','stir_frequency','end_hold','scan_frequency','scan_amp']
        vals = [0.25,             0.1,           50,            0,              0.15,      1000,             0.05]
        form = {}  
        for i, j in zip(keys, vals):
            form[i] = j
            
            
        for stir in [4]:
            form['stir_frequency'] = stir
            form['start_phase'] = vals[0]
            wf = gen_wf(form)
        #debug_plot(wf.times, wf.thetavalues, wf.rvalues, wf.xvalues, wf.yvalues, wf.x_rafdata, wf.y_rafdata)
        file_list_path = os.path.join(wf.get_disc(), '\\Wright Lab Code\\Rigol\\RAF_files')
        file_list = display_files(file_list_path)
    
    ###################Instantiate AWG and load RAF files###########################
    else:
        instr_ID = 'USB0::0x1AB1::0x0641::DG4E175003380::INSTR'
        awg = AODController(instr_ID)
        
        #Generate a list that contains all the folders
        file_list_path = os.path.join(awg.get_disc(), '\\Wright Lab Code\\Rigol\\RAF_files')
        file_list = display_files(file_list_path)
                
        
        vals = to_vals(file_list[file_num])
        keys = ['start_phase','initial_hold','acceleration','stir_frequency','end_hold','scan_frequency','scan_amp']
        form = {}
        for i, j in zip(keys, vals):
            form[i] = j
        
        
        awg_initialize(awg)
        wf = waveform(form)
        awg.period(wf.total_time)
        print('Total time of the scan is %.3fs, recommended stir_time is %.3f' % (wf.total_time, wf.total_time - 0.15))
        awg.load_and_config(wf.to_str())
        
        
        

    
'''    
Work flow:
    -- To load an already generated file, first make sure it's in the memory stick, change gen_wave to False, then find the file list and its number in the array, change file_num to that number and run the code
    -- To generate more waveforms, change gen_wave to True and change parameters in the loop to generat files. Transfer them to the memory stick
'''    

    
    
 

    
    
 
