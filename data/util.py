import re
import numpy as np


def get_numpy_volume_from_dicoms(dicom_slices):

    slice_projections = []
    for slice in dicom_slices:
        #int(dicom_slices[i]["InstanceNumber"].value)
        IOP = np.array(slice.ImageOrientationPatient)
        IPP = np.array(slice.ImagePositionPatient)
        normal = np.cross(IOP[0:3],IOP[3:])
        projection = np.dot(IPP, normal)
        slice_projections += [{"d": projection, "slice": slice}]

    sorted_slices = sorted(slice_projections, key=lambda i: i["d"])

    volume = np.dstack([slice["slice"].pixel_array for slice in sorted_slices])

    volume = np.swapaxes(volume, 0, 1) 

    return volume



# def get_numpy_volume_from_dicoms(dicom_slices):
#     # Extract pixel arrays from DICOM slices [1, height, width, depth]
#     ct_slices = []
#     for i in range(len(dicom_slices)):
#         assert i == int(dicom_slices[i]["InstanceNumber"].value)
#         ct_slices.append(dicom_slices[i].pixel_array)
#     ct_scan = np.stack(ct_slices, axis=-1)       
#     # ct_scan = np.rot90(ct_scan, k=1, axes=(0, 1)).copy()
#     # ct_scan = np.swapaxes(ct_scan, 0, 1)           ######
#     ct_scan = np.expand_dims(ct_scan, axis=0)
#     return ct_scan



def get_affine_transform_from_dicom(dicom_data):

    # Get the pixel spacing and slice thickness from DICOM metadata
    pixel_spacing = dicom_data.PixelSpacing
    slice_thickness = dicom_data.SliceThickness

    # Extract the direction cosines for the image orientation
    direction_cosines = dicom_data.ImageOrientationPatient

    # Extract the image position
    image_position = dicom_data.ImagePositionPatient

    # Construct the affine transform matrix
    affine_matrix = np.array(
        [
            [
                direction_cosines[0] * pixel_spacing[0],
                direction_cosines[3] * pixel_spacing[1],
                0,
                image_position[0],
            ],
            [
                direction_cosines[1] * pixel_spacing[0],
                direction_cosines[4] * pixel_spacing[1],
                0,
                image_position[1],
            ],
            [0, 0, slice_thickness, image_position[2]],
            [0, 0, 0, 1],
        ]
    )

    # LAS to RAS (dirty fix)
    #return np.diagflat([-1, 1, 1, 1]).dot(affine_matrix)
    return affine_matrix


def sorted_alphanum(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def about_mat(arr):
    print(
        "I am an array of shape {}, type {}, with values ranging [{} - {}], mean = {}".format(
            arr.shape, arr.type, np.min(arr), np.max(arr), np.mean(arr)
        )
    )


def list_torchio_transforms_to_dict(transform_list):
    d = dict()
    for transform_instance in transform_list:
        name = type(transform_instance).__name__
        args = {arg: getattr(transform_instance, arg) for arg in transform_instance.args_names}
        d[name] = args
    return d