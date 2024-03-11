import torch

def custom_collate(batch):
    """
    Custom collate function to handle varying channels in input images.
    """
    tensors, targets = zip(*batch)

    # Find the maximum number of channels in the batch
    max_channels = max(sample.shape[0] for sample in tensors)

    # Pad images with fewer channels to match the maximum number of channels
    padded_tensors_list = [
        torch.cat(
            [sample, torch.zeros(max_channels - sample.shape[0], *sample.shape[1:])],
            dim=0,
        )
        for sample in tensors
    ]

    # Stack the padded images into a batch
    padded_tensors = torch.stack(padded_tensors_list, dim=0)

    return padded_tensors, targets

if __name__ == "__main__":

    from config import ConfigIRCAD
    from dataset import IRCAD

    from pathlib import Path
    import torchio as tio

    mount_dir = (
        Path("/mnt/Shared/")
        if Path.exists(Path("/mnt/Shared/"))
        else Path.home() / "data"
    )
    root_data_dir = mount_dir / "3Dircadb1"

    test_transforms = [
        #tio.ToCanonical(),  # to RAS
        tio.Clamp(-150, 250),
        tio.ZNormalization(masking_method='liver'), # masking_method = tio.ZNormalization.mean
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        tio.Resample((1, 1, 2)),  # to 1 mm iso
        tio.CropOrPad((256,256,128), mask_name='liver'),
    ]

    train_transforms = test_transforms + [
        tio.OneOf({
          tio.RandomAffine(degrees = 15, translation = 15): 0.5,
          tio.RandomFlip(): 0.3,
          tio.RandomNoise(mean = 0.6): 0.2,
        })        
    ]

    config = ConfigIRCAD(root_data_dir)

    train_dataset = IRCAD(
        config=config,
        subset='train',
        transform=tio.Compose(train_transforms),
    )
    test_dataset = IRCAD(
        config=config,
        subset='test',
        transform=tio.Compose(test_transforms),
    )

    sample_subject = test_dataset[0]
    sample_subject.plot(radiological=False)

    print("Number of subjects in dataset:", len(train_dataset))
    print("Subject IDs:", train_dataset.ids)
    print("Keys in subject:", tuple(sample_subject.keys()))
    print("Shape of CT data:", sample_subject["ct"].shape)
    print("Orientation of CT data:", sample_subject["ct"].orientation)
    print("Size of CT data: {:.2f} MB".format(sample_subject['ct']['data'].nbytes / 1e6))
    print("Image path:", sample_subject.ct.path)
    
    #transformed_subject = tio.Compose(transforms)(sample_subject)
    #transformed_subject.plot()

    # Use DataLoader to create batches
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=4, shuffle=True, num_workers=4,       #collate_fn=custom_collate
    # )

    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, collate_fn=custom_collate, batch_size=4, shuffle=False
    # )


    # for batch in train_dataloader:
    #     print(batch['ct']['data'].shape)
    #     print(batch['cancer'])
    #     print('\n')