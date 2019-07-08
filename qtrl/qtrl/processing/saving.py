# Copyright (c) 2018-2019, UC Regents

from .base import ADCProcess
import pickle
import os
import datetime


class SaveMeasurement(ADCProcess):
    """ADC process that saves the measurement results"""
    def __init__(self, save_directory='C://Users/QNL/qtrl/projects/quantum_simulation/data', subdir=None, filename=None, save_seq=False, timestamp=True):

        # self._save_directory = save_directory
        self._base_directory = save_directory
        subdir = '' if subdir is None else subdir
        self._subdir = subdir
        self.save_directory = os.path.join(self._base_directory, self._subdir)
        self._filename = filename
        self._last_file = None
        self._last_filename = None
        self._save_seq = save_seq
        self.timestamp = timestamp

    def post(self, measurement, seq=None):
        if seq is not None and seq._name != self._subdir:
            subdir_old = self._subdir
            self._subdir = os.path.join(self._subdir, seq._name)
            self._make_data_dir()
            if hasattr(seq,'fname'):
                self._filename_timestamp(filename=seq.fname)
            else:
                self._filename_timestamp()
            self._subdir = subdir_old  # return to old subdir to avoid recursive directory creation

        else:
            self._make_data_dir()
            self._filename_timestamp()

        measurement['save_path'] = {'file': self._last_file,
                                    'directory': self.save_directory,
                                    'filename': self._last_filename}

        if seq is not None and self._save_seq:

            measurement = {'measurement': measurement,
                           'sequence': seq}
        else:
            measurement = {'measurement': measurement,
                           'sequence': None}

        with open(self._last_file, 'wb') as f:
            print('saving...')
            pickle.dump(measurement, f)

    def _save(self, measurement, subdir=None, filename=None):

        self._make_data_dir(subdir=subdir)
        self._filename_timestamp(filename)

        with open(self._last_file, 'wb') as f:
            print('saving...')
            pickle.dump(measurement, f)

    def _make_data_dir(self, subdir=None, date=None):  # , data_dir, subdir):
        """Make a directory in the data saving directory according to the day,
        can add topical subdirectory if desired"""
        if subdir is not None:
            self._subdir = subdir
        if date is None:
            date = datetime.datetime.now().strftime('%Y_%m_%d')
        elif date is False:  # if you want to make a directory that doesn't specify date, for saving more general stuff
            date = ''

        data_dir = os.path.join(self._base_directory,
                                date,
                                self._subdir)

        if not os.path.exists(data_dir):
            print("Data storage dir does not exist,"
                  " creating it at {}".format(data_dir))
            os.makedirs(data_dir)
        # return data_dir
        self.save_directory = data_dir


    def _filename_timestamp(self, filename=None, extension='pickle'):
        if self.timestamp is True:
            ts = datetime.datetime.now().strftime('%H-%M-%S-%f')[:-3]
        else:
            ts = ''
        if filename is not None:
            self._filename = filename
        if self._filename is None:
            filename = ts + '_No_Name'
        else:
            filename = ts + '_' + self._filename

        self._last_filename = os.path.join(self.save_directory, filename)
        self._last_file = self._last_filename + f'.{extension}'

class Saver():
    def __init__(self, base_directory, subdir=None, filename=None, timestamp=True):
        self.base_directory = base_directory
        subdir = '' if subdir is None else subdir
        self.subdir = subdir
        self.save_directory = os.path.join(self.base_directory, self.subdir)
        self.filename = filename
        self.timestamp = timestamp

    def save(self, object, subdir=None, filename=None):
        # if seq is not None:
        #     self._subdir = seq._name

        self.make_data_dir(subdir=subdir)
        self._filename_timestamp(filename)

        with open(self.last_file, 'wb') as f:
            print('saving...')
            pickle.dump(object, f)

    def load(self,filename, subdir=None, date=None):
        """
        Loads data from filename.
        :param filename: string, either full path or just a filename.
        :param subdir:   string or False. If string, cwd is changed to this subdir
        :param date:     either date string, None, or False. If None, today's date is made.
                         If False, no date is used.
        :return:         data in filepath
        """
        if os.path.isabs(filename):
            with open(filename,'rb') as f:
                return pickle.load(f)
        else:
            self.make_data_dir(subdir,date)
            with open(os.path.join(self.save_directory,filename),'rb') as f:
                return pickle.load(f)

    def make_data_dir(self, subdir = None, date = None):#, data_dir, subdir):
        """Make a directory in the data saving directory according to the day,
        can add topical subdirectory if desired"""
        if subdir is not None:
            self.subdir = subdir
        if date is None:
            date = datetime.datetime.now().strftime('%Y_%m_%d')
        elif date is False: #if you want to make a directory that doesn't specify date, for saving more general stuff
            date = ''
        data_dir = os.path.join(self.base_directory,
                                date,
                                self.subdir)

        if not os.path.exists(data_dir):
            print("Data storage dir does not exist,"
                  " creating it at {}".format(data_dir))
            os.makedirs(data_dir)
        # return data_dir
        self.save_directory = data_dir

    def _filename_timestamp(self, filename=None, extension='pickle'):

        if filename is not None:
            self.filename = filename
        if self.filename is None:
            filename = datetime.datetime.now().strftime('%H-%M-%S-%f')[:-3]+'_No_Name'
        else:
            filename = datetime.datetime.now().strftime('%H-%M-%S-%f_')[:-3]+'_'+self.filename

        self.last_filename = os.path.join(self.save_directory, filename)
        self.last_file = self.last_filename + f'.{extension}'
