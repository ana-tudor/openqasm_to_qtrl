# Copyright (c) 2018-2019, UC Regents

from qtrl.settings import Settings
from qtrl.utils.config import NoSetting


if Settings.setup == Settings.ONLINE or Settings.setup == Settings.OFFLINE:

    from ..utils.config import YAMLConfig

  # TODO: configuration is deeply hardwired and should not be a
  # base class but a data member. In the interest of time, the
  # ManagerBase is just a stub to switch out configuration types.

  # "old-style"
    class ManagerBase(YAMLConfig):
        pass

else:

    from ..utils.config import load_json_config

  # "new-style"
    class ManagerBase(object):
        def __init__(self, config_file, variables={}):
            self.config = load_json_config(config_file, variables)

        def load(self):
            self.config.load()

        def get(self, key, default=NoSetting, reload_config=False):
            return self.config.get(key)
