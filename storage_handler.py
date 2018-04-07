from functional import seq
from functools import partial
import os
import re
import time
import yaml


class StorageHandler(object):
    def __init__(self, dataset="iris"):
        self.dataset = dataset
        # Get run id
        self.run_id = self._get_run_id()
        # Make directory for this run
        os.mkdir(os.path.join(self._get_log_dir_path(), self.run_id))

    def _get_log_dir_path(self):
        # Get path of source code root
        root_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(root_path, 'logs', self.dataset)

        return dir_path

    def _get_run_id(self):
        # Get logdir path
        dir_path = self._get_log_dir_path()

        # If logs directory is not present
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            return '1'

        # Get list of all subdirectires
        all_dirs = [x for x in os.listdir(dir_path) if not os.path.isfile(x)]

        # Filter out directories which are not numbers
        match_int = partial(re.match, "^\d+?$")
        run_dirs = seq(all_dirs).filter(match_int).map(int).list()

        if len(run_dirs) == 0:
            return '1'

        # Find the latest run
        last_run = sorted(run_dirs)[-1]

        return str(last_run + 1)

    def store_args(self, args):
        stream = open(os.path.join(self._get_log_dir_path(),
                                   self.run_id, 'args.yml'), "w+")
        yaml.dump(args, stream)

    def get_log_dir(self):
        return os.path.join(self._get_log_dir_path(),
                     self.run_id)
