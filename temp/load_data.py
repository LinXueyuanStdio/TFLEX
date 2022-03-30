import numpy as np


class Data:
    def __init__(self, data_dir='data/icews14/'):
        # load data
        self.train_data = self._load_data(data_dir, "train")
        self.valid_data = self._load_data(data_dir, "valid")
        self.test_data = self._load_data(data_dir, "test")

        # Put it in 1 big matrix
        self.data = np.vstack((self.train_data, self.test_data))
        self.data = np.vstack((self.data, self.valid_data))

        # List of individual entities/relations/time
        self.entities = sorted(list(set(np.concatenate((self.data[:, 0], self.data[:, 2])))))
        self.relations = sorted(list(set(self.data[:, 1])))
        self.time = sorted(list(set(self.data[:, 3])))

        # Dict converting from name to idx
        self.entities_dict = dict()
        self.relations_dict = dict()
        self.temporal_dict = dict()

        for i in range(len(self.relations)):
            self.relations_dict[self.relations[i]] = i

        for i in range(len(self.entities)):
            self.entities_dict[self.entities[i]] = i

        for i in range(len(self.time)):
            self.temporal_dict[self.time[i]] = i

        # Data matrix with idxs
        self.train_data_idxs = self._create_idxs_matrix(self.train_data)
        self.valid_data_idxs = self._create_idxs_matrix(self.valid_data)
        self.test_data_idxs = self._create_idxs_matrix(self.test_data)

    def _create_idxs_matrix(self, data_matrix):
        idxs_matrix = np.zeros(data_matrix.shape, dtype=int)

        for i in range(data_matrix.shape[0]):
            idxs_matrix[i, 0] = self.entities_dict[data_matrix[i, 0]]
            idxs_matrix[i, 1] = self.relations_dict[data_matrix[i, 1]]
            idxs_matrix[i, 3] = self.entities_dict[data_matrix[i, 2]]
            idxs_matrix[i, 2] = self.temporal_dict[data_matrix[i, 3]]

        return idxs_matrix

    def _load_data(self, data_dir, filename='test.txt'):
        return np.loadtxt(data_dir + filename, dtype='str', delimiter='\t')
