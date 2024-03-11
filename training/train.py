import torch
import torch.nn as nn
from torch.utils.data import  Dataset
# from utils.DatasetProcess import GetData
# from utils.LossFunctions import Scores
# from utils.InputDataProcess import GlobalConfig
import torchio as tio
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output


class Training:
    def __init__(self,
                 model: nn.Module,
                 dataset : Dataset,
                 criterion: nn.Module,
                 lr: float,
                 num_epochs: int,
                 batch_size: int,
                 path_to_csv: str,
                 fold: int,
                 accumulation_steps: int,
                 save_model_history: bool=True,
                 display_plot: bool= True
                ):
        # 1.Setup criterion and optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Training on', self.device)
        if torch.cuda.device_count() >= 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          self.model = nn.DataParallel(model).to(self.device)
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.75,
                                                                    patience= 5,
                                                                    verbose=True,
                                                                   )
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.getdata = GetData()
        # 2. Setup dataset
        self.phases = ['train', 'val']
        self.dataloaders = {phase: self.getdata.get_dataloader(self.dataset,
                                                  path_to_csv=path_to_csv,
                                                  phase=phase,
                                                  fold=fold,
                                                  batch_size=self.batch_size,
                                                  num_workers=4
                                                 )for phase in self.phases}
        # 3. Setup loss and plot
        self.best_loss = float('inf')
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.save_model_history = save_model_history
        self.display_plot = display_plot
        self.config = GlobalConfig()
        self.best_dice_scores = float('inf')
        self.best_iou_scores = float('inf')
    def run(self):
        for epoch in range(self.num_epochs):
            self.one_epoch(epoch, 'train')
            with torch.no_grad():
                val_loss = self.one_epoch(epoch, 'val')
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print('Saved new checkpoint')
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.config.root_system, 'best_checkpoint.pth'))#self.config.check_point +
            if self.dice_scores['val'][-1] >= 0.75 or self.iou_scores['val'][-1] >= 0.75:
                print(f"Save highest_checkpoint_{self.dice_scores['val'][-1]:0.2f}.pth")
                save_name = f"highest_checkpoint_{self.dice_scores['val'][-1]:0.2f}.pth"
                self.best_iou_scores = self.iou_scores['val'][-1]
                self.best_dice_scores = self.dice_scores['val'][-1] 
                torch.save(self.model.state_dict(), os.path.join(self.config.root_system,  save_name))#self.config.check_point +
            print('-------------------------------')
        if self.display_plot:
            self.plot_history()
        if self.save_model_history:
            self.save_history()
        print(f'Best Dice Scores: {self.best_dice_scores}')
        print(f'Best IoU Scores: {self.best_iou_scores}')
                
    def one_epoch(self, epoch: int, phase: str):
        print(f'{phase} epoch: {epoch}')
        self.model.train() if phase == 'train' else self.model.eval()
        scores = Scores()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for i, data in enumerate(dataloader):
            images = data['image'][tio.DATA].to(self.device)
            targets = data['label'][tio.DATA].to(self.device)
            predict = self.model(images)
            loss = self.criterion(predict, targets.float()) #.float()
            loss = loss/self.accumulation_steps
            if phase == 'train':
                loss.backward()
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            scores.update(predict.detach().cpu(),
                          targets.detach().cpu()
                         )
        epoch_loss = (running_loss * self.accumulation_steps)/total_batches
        epoch_dice, epoch_iou = scores.get_metrics()
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.iou_scores[phase].append(epoch_iou)
        print(f'loss: {epoch_loss:.3f} dice: {epoch_dice:.3f} iou: {epoch_iou:.3f}')
        return epoch_loss
    
    def save_history(self):
        torch.save(self.model.state_dict(),os.path.join(self.config.root_system, 'last_checkpoint.pth'))#self.config.check_point +
#         torch.save(self.model,os.path.join(self.config.root_system,  'model.pth'))#self.config.check_point +
        print('Saved check point')
        logs_ = [self.losses, self.dice_scores, self.iou_scores]
        log_name_ = ['_loss', '_dice', '_iou']
        logs = [logs_[i][key] for i in range(len(logs_)) for key in logs_[i]]
        log_names = [key + log_name_[i] for i in range(len(logs_)) for key in logs_[i]]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv(os.path.join(self.config.root_system, 'train_log.csv'), index=False)#self.config.check_point +
    
    def load_predtrain_model(self, pred_path:str):
        self.model.load_state_dict(torch.load(pred_path))
        print('Predtrain model loaded')
    
    def plot_history(self):
        data = [self.losses, self.dice_scores, self.iou_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,
            
            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]} 
            """, 
                  
            f"""
            train jaccard score {self.iou_scores['train'][-1]}
            val jaccard score {self.iou_scores['val'][-1]}
            """,
        ]
        clear_output(True)
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")
            plt.tight_layout()
            plt.show()
    