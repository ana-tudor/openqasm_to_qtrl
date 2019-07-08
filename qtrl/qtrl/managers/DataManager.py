import datetime
import os
import pandas as pd
from qtrl.utils import Config


class DataManager(Config):

    def __init__(self, config_file=None, variables=None, session=None):
        """


        :param config_file: Must contain keys:
                    - (string) base_directory
                    - (string) metadata_directory
                    - (string) data_format
                    - (string) metadata_format
        :param variables: normal config variables argument, no clear use here yet
        :param session: (string) name of experimental session (default today's date YYYY-MM-DD).
                        This will tag the metadata .csv with this session tag
        """
        # standard config initialization
        super().__init__(config_file, variables)

        # set the directory where data is stored
        self.base_directory = self._config_dict['base_directory']  # where data is saved
        self.metadata_directory = self._config_dict['metadata_directory']  # location of meta-data database
        self.save_directory = None
        # session is enumerated by the day, but can be customized to be any stringable thing
        if session is None:
            print(f"No session set...setting to {datetime.datetime.now().strftime('%Y_%m_%d')}")
            self.session = datetime.datetime.now().strftime('%Y_%m_%d')
        else:
            self.session = session

        # initialize the database for the session
        self.data_format = self._config_dict['data_format']
        self.metadata_format = self._config_dict['metadata_format']
        self._metadata_file = rf'{self.metadata_directory}/metadata_{self.session}.{self.metadata_format}'



        self.set_exp_id()
        self.update_ID = self._config_dict['update_ID']  # this will update for every call to make_save_dir()
        self.prepend_ID = self._config_dict['prepend_ID']
        # if self.update_ID:
        #     self.set_exp_id()


        self.saved_files = []

    # def _init_session_database(self):
    #     """
    #     If there is no database file, this function will make it. If the file exists, it does nothing
    #     """
    #     self._metadata_file = rf'{self.metadata_directory}/metadata_{self.session}.{self.metadata_format}'
    #     if not os.path.exists(self._metadata_file):
    #         print(f'Metadatabase does not exist. Creating \n {self._metadata_file}')
    #         with open(self._metadata_file, 'w') as f:
    #             df = pd.DataFrame([])
    #             if self.metadata_format == 'csv':
    #                 df.to_csv(f, index=False)

    def get_session_metadata(self):
        """
        Wrapper to get the meta-data database as a Pandas dataframe
        """
        assert os.path.exists(self._metadata_file), f"No database exists at {self._metadata_file}"
        with open(self._metadata_file, 'r') as f:
            df = pd.read_csv(f)
        return df

    def make_save_dir(self, dir_string='', date='auto', date_first=True):
        """
        Makes a save_directory for the data given directory string. By default
        the order is
        <base_dir> / <date> / <dir_string>,

        but, if date is False:
        <base_dir> / <dir_string>

        if date_first=True:
        <base_dir> /<dir_string> / <date>,

        depending on preference

        Returns save_dir
        """
        # Make a new unique identifier when making a new directory for data to be stored
        if self.update_ID:
            self.set_exp_id()

        # add the ID to the subdirectory, if prepend_ID is true
        if dir_string is not '' and self.prepend_ID:
            dir_string = rf'{self.exp_ID}_{dir_string}'

        # add the date automatically
        if date is 'auto':
            date = datetime.datetime.now().strftime('%Y_%m_%d')

        # decide ordering which to put the date
        if date_first:
            save_dir = rf'{self.base_directory}/{date}/{dir_string}'
        else:
            save_dir = rf'{self.base_directory}/{dir_string}/{date}'

        # or don't add the date
        if date is None:  # if you want to make a directory that doesn't specify date, for saving more general stuff
            save_dir = rf'{self.base_directory}/{dir_string}'

        if not os.path.exists(save_dir):
            print("Data storage dir does not exist,"
                  " creating it at {}".format(save_dir))
            os.makedirs(save_dir)

        return save_dir

    def _filename_timestamp(self, filename=None, extension='json'):

        if filename is not None:
            self.filename = filename
        if self.filename is None:
            filename = datetime.datetime.now().strftime('%H-%M-%S-%f')[:-3] + '_No_Name'
        else:
            filename = datetime.datetime.now().strftime('%H-%M-%S-%f_')[:-3] + '_' + self.filename

        self.last_filename = os.path.join(self.save_directory, filename)
        self.last_file = self.last_filename + f'.{extension}'

        return self.last_file

    def _append_metadata_to_database(self, metadata):
        """
        Appends (dict-like) metadata to the session metadata database.

        """
        df = pd.DataFrame([metadata])
        if os.path.exists(self._metadata_file):
            session_df = self.get_session_metadata()
            df = pd.concat([session_df, df], axis=0, ignore_index=True)

        with open(self._metadata_file, 'w') as f:
            df.to_csv(f, index=False)

    def set_exp_id(self):
        self.exp_ID = int(datetime.datetime.timestamp(datetime.datetime.now()))

    def save_data(self, dir_string, filename, data_dict, config_dict, extension='json'):

        self.save_directory = self.make_save_dir(dir_string=dir_string)
        filepath = self._filename_timestamp(filename)
        file_metadata = {'path': filepath,
                         'id': self.exp_ID,
                         'timestamp': f'{datetime.datetime.now()}'}
        metadata = {**file_metadata, **config_dict}
        # print(f'metadata keys: {metadata.keys()}')
        self._append_metadata_to_database(metadata)
        if 'config' not in data_dict:
            data_dict['config'] = metadata

        data_df = pd.DataFrame([data_dict])
        if extension == 'json':
            if os.path.exists(filepath):
                print('File exists! not saving...')
            else:
                with open(filepath, 'w') as f:
                    data_df.to_json(f)
                # add to quick-access list of saved
                self.saved_files.append(file_metadata)
        else:
            print('Need to implement other formats besides .json!')
