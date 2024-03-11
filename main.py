from data.config import ConfigIRCAD
from data.dataset import IRCAD
from models.toy import ToyNet

from pathlib import Path
import torchio as tio
import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


if __name__ == "__main__":

    # Path to Data Directory
    mount_dir = (
        Path("/mnt/Shared/")
        if Path.exists(Path("/mnt/Shared/"))
        else Path.home() / "data"
    )
    root_data_dir = mount_dir / "3Dircadb1"

    # Data Transforms
    test_transforms = [
        #tio.ToCanonical(),  # to RAS64
        tio.Clamp(-150, 250),
        tio.ZNormalization(masking_method = tio.ZNormalization.mean),  #(masking_method='liver'),
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        tio.Resample((1, 1, 2)),  # to 1 mm iso
        tio.CropOrPad((128,128,64), mask_name='liver'),
    ]

    train_transforms = test_transforms + [
        tio.OneOf({
          tio.RandomAffine(degrees = 15, translation = 15): 0.5,
          tio.RandomFlip(): 0.3,
          tio.RandomNoise(mean = 0.6): 0.2,
        })        
    ]

    # Instantiate Dataset
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

    ## Display Example
    # sample_subject = test_dataset[0]
    # sample_subject.plot(radiological=False)

    # print("Number of subjects in dataset:", len(tNetrain_dataset))
    # print("Subject IDs:", train_dataset.ids)
    # print("Keys in subject:", tuple(sample_subject.keys()))
    # print("Shape of CT data:", sample_subject["ct"].shape)
    # print("Orientation of CT data:", sample_subject["ct"].orientation)
    # print("Size of CT data: {:.2f} MB".format(sample_subject['ct']['data'].nbytes / 1e6))
    # print("Image path:", sample_subject.ct.path)
    
    #transformed_subject = tio.Compose(transforms)(sample_subject)
    #transformed_subject.plot()

    # Define DataLoader to create batches
    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,       #collate_fn=custom_collate
    )

    validation_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False
    )

    # Model architecture
    model = ToyNet()
    print(summary(model, (1,128,128,64)))

    # Loss
    loss_fn = torch.nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Define Training epoch
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, batch in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs = batch['ct']['data'].float()
            labels = batch['cancer'].unsqueeze(-1).float()
  
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            #if i % 1000 == 999:
            #last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, running_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        return last_loss
    

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('tensorboard/toynet_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 3

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vbatch in enumerate(validation_loader):
            vinputs = vbatch['ct']['data'].float()
            vlabels = vbatch['cancer'].unsqueeze(-1).float()
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1