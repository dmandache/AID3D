from pathlib import Path
import glob
from zipfile import ZipFile
from natsort import natsorted
import numpy as np
import pandas as pd
import pydicom
import torch
import torchio as tio
from torchio.data.subject import Subject

from .util import *
from .config import ConfigIRCAD

from pythae.data.datasets import DatasetOutput


class IRCAD(tio.SubjectsDataset):
    """
    IRCAD dataset.
    - realized during the arterial phase in inhaled position
    """
    base_url = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"

    def __init__(
        self,
        config,
        subset=None,  # train / test / None (all) / list of ids
        transform=None,
        **kwargs,
    ):
        root = config.root
        
        if subset is None:
            self.ids = config.patient_ids
        else:
            if type(subset) is list:
                self.ids = subset
            elif type(subset) is int:
                self.ids = list(range(subset))
            elif subset == 'train':
                self.ids = config.train_ids
            elif subset == 'test':
                self.ids = config.test_ids
            else:
                message = "Type {} not supported for subset parameter, must be int or list.".format(type(subset))
                raise RuntimeError(message)

        subjects_list = self._get_subjects_list(root, self.ids)
        # self.label_names = self._get_all_labels(root)
        super().__init__(subjects_list, transform=transform, **kwargs)

    
    @staticmethod
    def _get_all_labels(root):
        zip_paths_masks_dirs = natsorted(root.glob("**/MASKS_DICOM.zip"))
        print(zip_paths_masks_dirs)
        labels = set()
        for zip_dir in zip_paths_masks_dirs:
            with ZipFile(zip_dir, "r") as zip_ref:
                organs = [IRCAD._get_mask_name(fp) for fp in zip_ref.namelist()]
                labels.update(set(organs))
        return labels
    

    @staticmethod
    def _get_targets(root, csv_target_key):
        csv = pd.read_csv(root / glob.glob("*.csv", root_dir=root, recursive=True)[0])
        targets = csv[csv_target_key]
        targets = (np.array(targets) > 0).astype(int)  # binarize
        return targets

    @staticmethod
    def _get_mask_name(path):
        pattern = re.compile(r"/(\w+)*/")
        match = pattern.search(str(path))
        if match:
            return match.group(1)
        else:
            return None

    @staticmethod
    def _get_subject_id(path):
        pattern = re.compile(r"\.(\d{1,2})/")
        match = pattern.search(str(path))
        if match:
            return int(match.group(1)) - 1
        else:
            return None
    
    @staticmethod
    def _get_subjects_list(root, subjects_ids):

        targets = IRCAD._get_targets(root, csv_target_key="no_of_tumors")

        zip_paths_patient = natsorted(root.glob("**/PATIENT_DICOM.zip"))
        zip_paths_labels = natsorted(root.glob("**/LABELLED_DICOM.zip"))
        zip_paths_masks = natsorted(root.glob("**/MASKS_DICOM.zip"))

        subjects = []

        for i in subjects_ids:

            assert (
                i
                == IRCAD._get_subject_id(zip_paths_patient[i])
                == IRCAD._get_subject_id(zip_paths_labels[i])
                == IRCAD._get_subject_id(zip_paths_masks[i])
            )

            subjects.append(
                tio.Subject(
                    ct=tio.ScalarImage(
                        root / zip_paths_patient[i],
                        type=tio.INTENSITY,
                        reader=IRCAD._get_scan_from_archive,
                    ),
                    labels=tio.LabelMap(
                        root / zip_paths_labels[i],
                        type=tio.LABEL,
                        reader=IRCAD._get_scan_from_archive,
                    ),
                    liver=tio.LabelMap(
                        root / zip_paths_masks[i],
                        type=tio.LABEL,
                        reader=lambda x: IRCAD._get_mask_from_archive(x, "liver"),
                    ),
                    cancer=targets[i],
                    id=i,
                )
            )

        return subjects

    @staticmethod
    def _get_scan_from_archive(zip_path):
        with ZipFile(zip_path, "r") as zip_ref:
            file_names = zip_ref.namelist()
            file_names = file_names[1:]  # Remove first position wich is the rootdir name
            file_names = sorted_alphanum(file_names)  # Ensure slices are in order

            # Safely read an image in archive
            def read_dicom_from_archive(filename):
                with zip_ref.open(filename) as file:
                    return pydicom.dcmread(file)

            # Read all dicom slices
            dicom_slices = [read_dicom_from_archive(fn) for fn in file_names]

        # Extract pixel arrays from DICOM slices [1, width, height, depth]
        ct_scan = get_numpy_volume_from_dicoms(dicom_slices)

        # Compute affine transform from dicom metadata
        affine = get_affine_transform_from_dicom(dicom_slices[0])

        return torch.from_numpy(ct_scan).float(), affine

    @staticmethod
    def _get_mask_from_archive(zip_path, organ="liver"):
        with ZipFile(zip_path, "r") as zip_ref:
            mask_dirs = zip_ref.namelist()
            organ_filenames = [
                dir for dir in mask_dirs if "MASKS_DICOM/" + organ + "/" in dir
            ]
            organ_filenames = organ_filenames[1:]  # on first position DIRNAME
            organ_filenames = sorted_alphanum(organ_filenames)

            # Safely read an image in archive
            def read_dicom_from_archive(filename):
                with zip_ref.open(filename) as file:
                    return pydicom.dcmread(file)

            # Read all dicom slices
            dicom_slices = [read_dicom_from_archive(fn) for fn in organ_filenames]

        # Extract pixel arrays from DICOM slices [1, width, height, depth]
        mask = get_numpy_volume_from_dicoms(dicom_slices)

        # Compute affine transform
        affine = get_affine_transform_from_dicom(dicom_slices[0])

        return torch.from_numpy(mask).int(), affine


class IRCAD_pythae(IRCAD):
    def __init__(self, image_types=['ct'], which_labels=['cancer'], **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, index: int) :
        subject = super().__getitem__(index) 
        #return subject['ct']['data'].float()
        return DatasetOutput(
            data=subject['ct']['data'].float(),
            #labels=subject['cancer'].float()
        )
    

if __name__ == "__main__":
    import torchio as tio

    mount_dir = (
        Path("/mnt/Shared/")
        if Path.exists(Path("/mnt/Shared/"))
        else Path.home() / "data"
    )
    root_data_dir = mount_dir / "3Dircadb1"

    transforms = [
        tio.ToCanonical(),  # to RAS
        tio.Clamp(-150, 250),
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        tio.Resample((1, 1, 2)),  # to 1 mm iso
    ]

    config = ConfigIRCAD(root_data_dir)

    train_dataset = IRCAD(
        config=config,
        subset='train',
        transform=tio.Compose(transforms),
    )
    test_dataset = IRCAD(
        config=config,
        subset='train',
        transform=tio.Compose(transforms),
    )
    print("Number of subjects in dataset:", len(train_dataset))
    print("Subject IDs:", train_dataset.ids)
    sample_subject = train_dataset[np.random.randint(0, len(train_dataset))]
    print("Keys in subject:", tuple(sample_subject.keys()))
    print("Shape of CT data:", sample_subject["ct"].shape)
    print("Orientation of CT data:", sample_subject["ct"].orientation)
    sample_subject.plot(radiological=True)

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('qtagg')

    # def show_slices(slices):
    #     """ Function to display row of image slices """
    #     fig, axes = plt.subplots(1, len(slices))
    #     for i, slice in enumerate(slices):
    #         axes[i].imshow(slice.T, cmap="gray", origin="lower")

    # _, W, H, D = sample_subject['ct'].shape
    # volume = sample_subject['ct'].data
    # slice_0 = volume[0, W//2, :, :]     # saggital (left to right) side
    # slice_1 = volume[0, :, H//2, :]     # coronal (anterior to posterior) frontal
    # slice_2 = volume[0,:, :, D//2]      # axial (superior to inferior) transversal

    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("Center slices for CT volume")
    # plt.show()
