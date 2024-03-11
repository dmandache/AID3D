from pathlib import Path
from typing import Union, Literal
from dataclasses import dataclass, asdict, field

@dataclass
class ConfigIRCAD:
    root: Union[str, Path]
    split_strategy: Union[str, None] = None
    base_url: Union[str, Path] = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"
    csv_path: Union[str, Path] = "data.csv"
    download: bool = False
    n_patients: int = field(init=False)
    patient_ids: list = field(init=False)
    train_ids: list = field(init=False)
    test_ids: list = field(init=False)
        
    seed: int = 22

    def __post_init__(self):
        # Additional customization after the regular initialization
        self.compute_additional_arguments()

    def compute_additional_arguments(self):
        # CSV path
        self.csv_path = Path(self.root) / Path(self.csv_path)

        # Get general info from dirs
        self.n_patients = self._get_number_of_patients_in_dir(self.root)

        # split IDs for train / test
        self.patient_ids = list(range(self.n_patients))
        self.train_ids, self.test_ids = self._split_train_test(self.patient_ids, self.split_strategy)

        # dowload
        if self.download:
            self._download(self.root)
        if not self._check_exists(self.root):
            message = "Dataset not found. You can use download=True to download it"
            raise RuntimeError(message)

    def to_dict(self) -> dict:
        """Transforms object into a Python dictionnary

        Returns:
            (dict): The dictionnary containing all the parameters"""
        return asdict(self)
    
    @staticmethod
    def _split_train_test(ids, split_strategy=None):
        if split_strategy is None:
            train_ids = [0, 1 ,2 ,3 ,4 ,5 , 6]
            test_ids = [7, 8, 9]
        else:
            pass
        return train_ids, test_ids
    
    @staticmethod   
    def _get_number_of_patients_in_dir(root):
        return len(list(root.glob("**/PATIENT_DICOM.zip")))
    
    @staticmethod
    def _check_exists(root):
        return root.is_dir()
    
    def _download(self, root):
        """Download the IRCAD data if it does not exist already."""
        raise NotImplementedError


# class ConfigIRCAD():
#     def __init__(self, root, split_strategy=None, download=False):
#         # Paths
#         self.base_url = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"
#         self.root = Path(root)
#         self.csv_path = root / "data.csv"
        
#         # Get general info from dirs
#         self.n_patients = self._get_number_of_patients_in_dir(self.root)

#         # split IDs for train / test
#         self.patient_ids = list(range(self.n_patients))
#         self.train_ids, self.test_ids = self._split_train_test(self.patient_ids, split_strategy)

#         # global random seed
#         self.seed = 22

#         # dowload
#         if download:
#             self._download(root)
#         if not self._check_exists(root):
#             message = "Dataset not found. You can use download=True to download it"
#             raise RuntimeError(message)
    
#     def asdict(self):
#         return {'name': 'IRCAD', 'n_patients': self.n_patients}
    
#     @staticmethod
#     def _split_train_test(ids, split_strategy=None):
#         if split_strategy is None:
#             train_ids = [0, 1 ,2 ,3 ,4 ,5 , 6]
#             test_ids = [7, 8, 9]
#         else:
#             pass
#         return train_ids, test_ids
    
#     @staticmethod   
#     def _get_number_of_patients_in_dir(root):
#         return len(list(root.glob("**/PATIENT_DICOM.zip")))
    
#     @staticmethod
#     def _check_exists(root):
#         return root.is_dir()
    
#     def _download(self, root):
#         """Download the IRCAD data if it does not exist already."""
#         raise NotImplementedError
