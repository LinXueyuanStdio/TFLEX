from pathlib import Path
from typing import *

from toolbox.data.DatasetSchema import RelationalTripletDatasetSchema


class InductiveDataset(RelationalTripletDatasetSchema):
    def __init__(self, home: Union[Path, str] = "data/inductive", dataset_name: str = "fb237", version: int = 1, _ind: bool = False):
        """
        dataset_name: fb237, WN18RR, nell
        version: 1, 2, 3, 4
        _ind: True if this dataset is for inductive validation, otherwise False for transductive training
        """
        self.folder_name = f"{dataset_name}_v{version}{'' if not _ind else '_ind'}"
        super(InductiveDataset, self).__init__(self.folder_name, home)
        self._ind = _ind

    def get_data_paths(self) -> Dict[str, Path]:
        return self.default_data_paths('')

    def get_dataset_path(self):
        return self.root_path


class InductiveFB237(InductiveDataset):
    def __init__(self, version: int = 1, _ind: bool = False):
        super(InductiveFB237, self).__init__(dataset_name="fb237", version=version, _ind=_ind)


class InductiveWN18RR(InductiveDataset):
    def __init__(self, version: int = 1, _ind: bool = False):
        super(InductiveWN18RR, self).__init__(dataset_name="WN18RR", version=version, _ind=_ind)


class InductiveNELL(InductiveDataset):
    def __init__(self, version: int = 1, _ind: bool = False):
        super(InductiveNELL, self).__init__(dataset_name="nell", version=version, _ind=_ind)
