# 数据集路径，下载数据集
# outline
# 1. utils function
#    - extract_tar(tar_path, extract_path='.')
#    - extract_zip(zip_path, extract_path='.')
# 2. remote dataset
#    - RemoteDataset
#    - fetch_from_remote(name: str, url: str, root_path: Path)
# 3. RelationalTriplet class
#    - RelationalTriplet
#    - RelationalTripletDatasetMeta
#    - RelationalTripletDatasetCachePath
#    - RelationalTripletDatasetSchema
#      1. FreebaseFB15k
#      2. DeepLearning50a
#      3. WordNet18
#      4. WordNet18_RR
#      5. YAGO3_10
#      6. FreebaseFB15k_237
#      7. Kinship
#      8. Nations
#      9. UMLS
#     10. NELL_995
#    - get_dataset(dataset_name: str, custom_dataset_path=None)
# 3. custom dataset

import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

from toolbox.utils.Log import Log


# region 1. utils function
def extract_tar(tar_path, extract_path='.'):
    """This function extracts the tar file.

        Most of the knowledge graph datasets are downloaded in a compressed
        tar format. This function is used to extract them

        Args:
            tar_path (str): Location of the tar folder.
            extract_path (str): Path where the files will be decompressed.
    """
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract_tar(item.name, "./" + item.name[:item.name.rfind('/')])


def extract_zip(zip_path, extract_path='.'):
    """This function extracts the zip file.

        Most of the knowledge graph datasets are downloaded in a compressed
        zip format. This function is used to extract them

        Args:
            zip_path (str): Location of the zip folder.
            extract_path (str): Path where the files will be decompressed.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


# endregion


# region 2. remote dataset
class RemoteDataset:
    def __init__(self, name: str, url: str, root_path: Path):
        root_path.mkdir(parents=True, exist_ok=True)
        self._logger = Log(str(root_path / "fetch.log"))
        self.name = name
        self.url = url
        self.root_path: Path = root_path

        self.tar: Path = self.root_path / ('%s.tgz' % self.name)
        self.zip: Path = self.root_path / ('%s.zip' % self.name)

    def download(self):
        """ Downloads the given dataset from url"""
        self._logger.info("Downloading the dataset %s" % self.name)

        if self.url.endswith('.tar.gz') or self.url.endswith('.tgz'):
            if self.tar.exists():
                return
            with urllib.request.urlopen(self.url) as response, open(str(self.tar), 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        elif self.url.endswith('.zip'):
            if self.zip.exists():
                return
            with urllib.request.urlopen(self.url) as response, open(str(self.zip), 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            raise NotImplementedError("Unknown compression format")

    def extract(self):
        """ Extract the downloaded file under the folder with the given dataset name"""

        try:
            if os.path.exists(self.tar):
                self._logger.info("Extracting the downloaded dataset from %s to %s" % (self.tar, self.root_path))
                extract_tar(str(self.tar), str(self.root_path))
                return
            if os.path.exists(self.zip):
                self._logger.info("Extracting the downloaded dataset from %s to %s" % (self.zip, self.root_path))
                extract_zip(str(self.zip), str(self.root_path))
                return
        except Exception as e:
            self._logger.error("Could not extract the target file!")
            self._logger.exception(e)
            raise


def fetch_from_remote(name: str, url: str, root_path: Path):
    remote_data = RemoteDataset(name, url, root_path)
    remote_data.download()
    remote_data.extract()


# endregion


# region 3. Relational Triplet
class RelationalTriplet:
    """ The class defines the datastructure of the knowledge graph triples.

        Triple class is used to store the head, tail and relation triple in both its numerical id and
        string form. It also stores the dictonary of (head, relation)=[tail1, tail2,..] and
        (tail, relation)=[head1, head2, ...]

        Args:
          h (str or int): String or integer head entity.
          r (str or int): String or integer relation entity.
          t (str or int): String or integer tail entity.

        Examples:
           >>> from toolbox.data.DatasetSchema import RelationalTriplet
           >>> trip1 = RelationalTriplet(2,3,5)
           >>> trip2 = RelationalTriplet('Tokyo','isCapitalof','Japan')
    """

    def __init__(self, h, r, t):
        self.h = h
        self.r = r
        self.t = t

    def set_ids(self, h, r, t):
        """ This function assigns the head, relation and tail.

            Args:
                h (int): Integer head entity.
                r (int): Integer relation entity.
                t (int): Integer tail entity.
        """
        self.h = h
        self.r = r
        self.t = t


class BaseDatasetSchema:
    def __init__(self, name: str, home: str = "data"):
        self.name = name
        self.root_path = self.get_dataset_home_path(home)  # ./data/${name}

    def get_dataset_home_path(self, home="data") -> Path:
        data_home_path: Path = Path('.') / home
        data_home_path.mkdir(parents=True, exist_ok=True)
        data_home_path = data_home_path.resolve()
        return data_home_path / self.name

    def force_fetch_remote(self, url):
        fetch_from_remote(self.name, url, self.root_path)

    def try_to_fetch_remote(self, url):
        if not (self.root_path / "fetch.log").exists():
            self.force_fetch_remote(url)

    def dump(self):
        """ Displays all the metadata of the knowledge graph"""
        log_path = self.root_path / "DatasetSchema.log"
        _logger = Log(str(log_path), name_scope="DatasetSchema")
        for key, value in self.__dict__.items():
            _logger.info("%s %s" % (key, value))


class RelationalTripletDatasetSchema(BaseDatasetSchema):
    """./data
        - dataset name
          - name.zip
          - name (extracted from zip)
            - cache
              - cache_xxx.pkl
              - cache_xxx.pkl
            - ${prefix}train.txt
            - ${prefix}test.txt
            - ${prefix}valid.txt
        if dataset can be downloaded from url, call self.try_to_fetch_remote(url: str) after __init__

        Args:
            name (str): Name of the datasets

        Examples:
            >>> from toolbox.data.DatasetSchema import RelationalTripletDatasetSchema
            >>> kgdata = RelationalTripletDatasetSchema("dL50a")
            >>> kgdata.dump()

    """

    def __init__(self, name: str, home: str = "data"):
        BaseDatasetSchema.__init__(self, name, home)
        self.dataset_path = self.get_dataset_path()
        self.cache_path = self.get_dataset_path_child("cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.data_paths = self.get_data_paths()

    def get_dataset_path(self) -> Path:
        return self.root_path / self.name

    def get_dataset_path_child(self, name) -> Path:
        return self.dataset_path / name

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths()

    def default_data_paths(self, prefix="") -> Dict[str, Path]:
        """default data paths, using prefix

        :param prefix: for example, "${self.dataset_path}/${prefix}train.txt"

        """
        return {
            'train': self.get_dataset_path_child('%strain.txt' % prefix),
            'test': self.get_dataset_path_child('%stest.txt' % prefix),
            'valid': self.get_dataset_path_child('%svalid.txt' % prefix)
        }


class FreebaseFB15k(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading Freebase dataset.

        FreebaseFB15k module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(FreebaseFB15k, self).__init__("FB15k", home)
        url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths("freebase_mtr100_mte100-")


class DeepLearning50a(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading DeepLearning50a dataset.

        DeepLearning50a module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(DeepLearning50a, self).__init__("dL50a", home)
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/dL50a.tgz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths('deeplearning_dataset_50arch-')


class WordNet18(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading WordNet18 dataset.

        WordNet18 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(WordNet18, self).__init__("WN18", home)
        url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths('wordnet-mlj12-')

    def get_dataset_path(self):
        return self.root_path / 'wordnet-mlj12'


class WordNet18_RR(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading WordNet18_RR dataset.

        WordNet18_RR module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(WordNet18_RR, self).__init__("WN18RR", home)
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/WN18RR.tar.gz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths()

    def get_dataset_path(self):
        return self.root_path


class YAGO3_10(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading YAGO3_10 dataset.

        YAGO3_10 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(YAGO3_10, self).__init__("YAGO3_10", home)
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/YAGO3-10.tar.gz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths()

    def get_dataset_path(self):
        return self.root_path


class FreebaseFB15k_237(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading FB15k-237 dataset.

        FB15k-237 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(FreebaseFB15k_237, self).__init__("FB15K_237", home)
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/fb15k-237.tgz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths()

    def get_dataset_path(self):
        return self.root_path


class Kinship(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading Kinship dataset.

        Kinship module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(Kinship, self).__init__("Kinship", home)
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/kinship.tar.gz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths()

    def get_dataset_path(self):
        return self.root_path


class Nations(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading Nations dataset.

        Nations module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(Nations, self).__init__("Nations", home)
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/nations.tar.gz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths()

    def get_dataset_path(self):
        return self.root_path


class UMLS(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading UMLS dataset.

        UMLS module inherits the KnownDataset class for processing
        the knowledge graph dataset.
    """

    def __init__(self, home: str = "data"):
        super(UMLS, self).__init__("UMLS", home)
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/umls.tar.gz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths()

    def get_dataset_path(self):
        return self.root_path


class NELL_995(RelationalTripletDatasetSchema):
    """This data structure defines the necessary information for downloading NELL-995 dataset.

        NELL-995 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

    """

    def __init__(self, home: str = "data"):
        super(NELL_995, self).__init__("NELL_995", home)
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/NELL_995.zip"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths()

    def get_dataset_path(self):
        return self.root_path


def get_dataset(dataset_name: str):
    if dataset_name.lower() == 'freebase15k' or dataset_name.lower() == 'fb15k':
        return FreebaseFB15k()
    elif dataset_name.lower() == 'deeplearning50a' or dataset_name.lower() == 'dl50a':
        return DeepLearning50a()
    elif dataset_name.lower() == 'wordnet18' or dataset_name.lower() == 'wn18':
        return WordNet18()
    elif dataset_name.lower() == 'wordnet18_rr' or dataset_name.lower() == 'wn18_rr':
        return WordNet18_RR()
    elif dataset_name.lower() == 'yago3_10' or dataset_name.lower() == 'yago':
        return YAGO3_10()
    elif dataset_name.lower() == 'freebase15k_237' or dataset_name.lower() == 'fb15k_237':
        return FreebaseFB15k_237()
    elif dataset_name.lower() == 'kinship' or dataset_name.lower() == 'ks':
        return Kinship()
    elif dataset_name.lower() == 'nations':
        return Nations()
    elif dataset_name.lower() == 'umls':
        return UMLS()
    elif dataset_name.lower() == 'nell_995':
        return NELL_995()
    elif dataset_name.lower() == 'dbp15k':
        return DBP15k()
    elif dataset_name.lower() == 'dbp100k':
        return DBP100k()
    else:
        raise ValueError("Unknown dataset: %s" % dataset_name)


class DBP15k(RelationalTripletDatasetSchema):

    def __init__(self, name="fr_en", home: str = "data"):
        """
        :param name: choice "fr_en", "ja_en", "zh_en"
        """
        self.dataset_name = name
        super(DBP15k, self).__init__("DBP15k", home)
        url = "http://ws.nju.edu.cn/jape/data/DBP15k.tar.gz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        kg1, kg2 = self.dataset_name.split("_")
        return {
            'train': self.get_dataset_path_child('train.txt'),
            'test': self.get_dataset_path_child('test.txt'),
            'valid': self.get_dataset_path_child('valid.txt'),
            'seeds': self.get_dataset_path_child('ent_ILLs'),
            'kg1_attribute_triples': self.get_dataset_path_child('%s_att_triples' % kg1),
            'kg1_relational_triples': self.get_dataset_path_child('%s_rel_triples' % kg1),
            'kg2_attribute_triples': self.get_dataset_path_child('%s_att_triples' % kg2),
            'kg2_relational_triples': self.get_dataset_path_child('%s_rel_triples' % kg2),
        }

    def get_dataset_path(self):
        return self.root_path / self.name / self.dataset_name


class DBP100k(RelationalTripletDatasetSchema):
    def __init__(self, name="fr_en", home: str = "data"):
        """
        :param name: choice "fr_en", "ja_en", "zh_en"
        """
        self.dataset_name = name
        super(DBP100k, self).__init__("DBP100k", home)
        url = "http://ws.nju.edu.cn/jape/data/DBP100k.tar.gz"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        kg1, kg2 = self.dataset_name.split("_")
        return {
            'train': self.get_dataset_path_child('train.txt'),
            'test': self.get_dataset_path_child('test.txt'),
            'valid': self.get_dataset_path_child('valid.txt'),
            'seeds': self.get_dataset_path_child('ent_ILLs'),
            'kg1_attribute_triples': self.get_dataset_path_child('%s_att_triples' % kg1),
            'kg1_relational_triples': self.get_dataset_path_child('%s_rel_triples' % kg1),
            'kg2_attribute_triples': self.get_dataset_path_child('%s_att_triples' % kg2),
            'kg2_relational_triples': self.get_dataset_path_child('%s_rel_triples' % kg2),
        }

    def get_dataset_path(self):
        return self.root_path / self.name / self.dataset_name


class SimplifiedDBP15k(RelationalTripletDatasetSchema):
    def __init__(self, name="fr_en", home: str = "data"):
        """
        :param name: choice "fr_en", "ja_en", "zh_en"
        """
        self.dataset_name = name
        super(SimplifiedDBP15k, self).__init__("SimplifiedDBP15k", home)
        url = "https://github.com/LinXueyuanStdio/KG_datasets/raw/master/datasets/SimplifiedDBP15k.zip"
        self.try_to_fetch_remote(url)

    def get_data_paths(self) -> Dict[str, Path]:
        kg1, kg2 = self.dataset_name.split("_")
        return {
            'train': self.get_dataset_path_child('train.txt'),
            'test': self.get_dataset_path_child('test.txt'),
            'valid': self.get_dataset_path_child('valid.txt'),
            'seeds': self.get_dataset_path_child('ent_ILLs'),
            'kg1_attribute_triples': self.get_dataset_path_child('%s_att_triples' % kg1),
            'kg1_relational_triples': self.get_dataset_path_child('%s_rel_triples' % kg1),
            'kg2_attribute_triples': self.get_dataset_path_child('%s_att_triples' % kg2),
            'kg2_relational_triples': self.get_dataset_path_child('%s_rel_triples' % kg2),
        }

    def get_dataset_path(self):
        return self.root_path / self.name / self.dataset_name
# endregion
