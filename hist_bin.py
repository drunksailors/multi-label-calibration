import pytorch_lightning as pl
from voc_dataset import PascalVOCDataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as ET
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import timm
from pytorch_lightning.loggers import TensorBoardLogger
from model import MLCNNet
from binning_fn import prediction_and_label, bin_index, predictions_by_bin, new_predictions_test_data, positive_and_total_instances, fraction_of_positive_instances, ECE, avg_bin_conf, uncal_ECE, predictions
from sklearn.metrics import accuracy_score, hamming_loss


root_dir= '/home/arka/Pascal_voc/VOCdevkit/VOC2012'


transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

#preparing the dataset 
val_test_dataset= PascalVOCDataset(root_dir= root_dir, image_set='val', transform=transforms)

val_size= int(0.5*len(val_test_dataset))
test_size= len(val_test_dataset)- val_size

torch.manual_seed(20)
gen= torch.Generator().manual_seed(20)
val_dataset, test_dataset= torch.utils.data.random_split( val_test_dataset, [val_size, test_size], generator= torch.Generator().manual_seed(20))

val_loader= DataLoader(dataset= val_dataset, batch_size=32, shuffle=False)
test_loader= DataLoader(dataset= test_dataset, batch_size=32, shuffle=False)

backbone = timm.create_model("resnetv2_50",pretrained=False,num_classes = 0)
model= MLCNNet(backbone, 20)

#TVocnet is for test time
class TVOCNet(pl.LightningModule):
     def __init__(self, model):
          super().__init__()
          self.model= model
    

     def training_step(self, batch, batch_idx):
          x,y = batch
          outputs = self.model(x)
          loss= F.binary_cross_entropy_with_logits(outputs, y)
          self.log("train/loss", loss.item()/len(y))
          return loss
     
     def validation_step(self,batch,batch_idx):
            x,y = batch
            outputs = self.model(x)
            loss = F.binary_cross_entropy_with_logits(outputs,y)
            self.log("val/loss",loss.item() / len(y))

     def predict_step(self,batch,batch_idx,dataloader_idx=0):
            x,y = batch
            preds  = self.model(x)
            return preds,y
     
     def configure_optimizers(self):
            optim = torch.optim.AdamW(self.parameters(),lr = 8e-6)
            return optim
     
""""change path_to_saved_model to the path where you have saved your trained model """
test_model= TVOCNet.load_from_checkpoint('path_to_saved_model', model= model)
tester= pl.Trainer()

val_pred_label= tester.predict(test_model, val_loader)
test_pred_label= tester.predict(test_model, test_loader)

val_preds, val_labels = prediction_and_label(val_pred_label, val_size)
test_preds, test_labels = prediction_and_label(test_pred_label, test_size) 

calibrated_ECE= ECE(val_preds, 20, val_labels, test_preds, test_labels)
uncalibrated_ECE= uncal_ECE(test_labels, test_preds, 20)
print('Calibrated ECE')
print(calibrated_ECE)

print('Uncalibrated ECE')
print(uncalibrated_ECE)

calibrated_test_preds= new_predictions_test_data(val_preds, val_labels, test_preds, 20)

print('The accuracy scores are: \n')
print("%.10f" % accuracy_score(predictions(data= test_preds), test_labels))
print('For calibrated-',accuracy_score(predictions(data= calibrated_test_preds), test_labels))
print('.............................................................................................')


print('The hamming losses are\n')
print('For uncalibrated-' ,hamming_loss(predictions(data= test_preds), test_labels))
print('For Calibrated-',hamming_loss(predictions(data= calibrated_test_preds), test_labels))






