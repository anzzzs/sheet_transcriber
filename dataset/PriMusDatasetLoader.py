import numpy as np
import logging
import glob
from sklearn.model_selection import train_test_split
from dataset.PriMusDataset import PriMusDataset

class PriMusDatasetLoader:
    def __init__(self, root, logger: logging = None, agnostic: bool = True):
        label = 'agnostic' if agnostic else 'semantic'
        imgPattern = f"{root}/**/**/**.png"
        labelPattern = f"{root}/**/**/**.{label}"
        images = glob.glob(imgPattern)
        labels = glob.glob(labelPattern)

        self.root = root
        self.paths = np.array(list(zip(images, labels)))
        self.alphabet_dict = None
        if logger is None:
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s %(name)s.%(funcName)s %(levelname)s: %(message)s")
            self.logger = logging.getLogger(__name__)

    def set_alphabet_dict(self):
        self.logger.info("set alphabet dict")
        alphabet = set()
        for i in range(len(self.paths)):
            target = open(self.paths[i][1], "r").read().split("\t")
            target.pop()
            alphabet = alphabet | set(target)
            if (i % 10000 == 0) or i == len(self.paths):
                self.logger.info(f"i = {i}, len alphabet = {len(alphabet)}")

        alphabet_dict = {}
        alphabet_dict[''] = 0
        for ind, tg in enumerate(alphabet):
            alphabet_dict[tg] = ind+1
        self.alphabet_dict = alphabet_dict
        self.logger.info(f"alphabet len {len(alphabet_dict)}")

    def get_train_test(self, **kwargs):
        if self.alphabet_dict is None:
            self.set_alphabet_dict()

        train_idxs, test_idxs = train_test_split(list(range(len(self.paths))), **kwargs)
        return (PriMusDataset(np.array(self.paths)[train_idxs], self.alphabet_dict),
                PriMusDataset(np.array(self.paths)[test_idxs], self.alphabet_dict))