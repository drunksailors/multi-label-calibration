# multi-label-calibration 
To run the code follow the steps below. Follow them according to their order.  

Download the PASCAL-VOC dataset first. 

In the train.py file change the 'root_dir' to the path of the VOC2012 folder in the PASCAL-VOC dataset. Do this at line 21.
In train.py file change in line 86 change 'dirpath' to the path where the best model will be stored. 

In hist_bin.py in line 18 change the 'root_dir' to the path of the VOC2012 folder in the PASCAL-VOC dataset. 
In hist_bin.py in line 73 replace 'path_to_saved_model' to the path where the best model is saved.  

Run train.py to train the model.
Run hist_bin.py to get the calibrated results. 
