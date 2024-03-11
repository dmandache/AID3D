import torch.nn as nn
import torch.optim as optim

# Define the model architecture
class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))        
        # Use adaptive pooling to dynamically reshape the tensor
        x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))        
        # Flatten the tensor before passing it to the fully connected layers
        x = x.view(x.size(0), -1)        
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# # Define the training function
# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = nn.functional.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()