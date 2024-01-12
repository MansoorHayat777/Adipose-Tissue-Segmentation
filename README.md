## Abdominal Adipose Tissue Segmentation

Binhan Xi, Zhenghan Wang

The Digital Image Processing Course Project (Fall 2018)

#### Project Structure

The files include the following parts:

- **src:** includes two main code files,  `ot.py` and `train.py`
- **models:** includes the network models among which we use U-Net
- **train and val:** includes the dataset for training and validating
- **utils:** includes some tools for simple tranforms
- **results:** includes the result output of our experiment



#### Environment Settings

To run the code correctly, you need to 

1. modify the directory in `ot.py` and `train.py`. `ot.py` is a data loader to read in the image data and `train.py` is the main program to run the network.
2. install the necessary libraries mentioned and imported in both of these two main files, including **Pillow** and **torch**.
3. put the data you want into the dataset path and modify their names as the format in `ot.py`. For example, `IM10_label.png`.
4. in `train.py` you can change the epoch number, loss function, batch size and some other parameters to meet your need.
5. add correctly the **ckpt** directory to put the output as the format and structure tree mentioned in `ot.py` and `train.py`.
6. run `train.py` and wait for the result.



#### Experiment Results

The results are included in our submitted report and we also put it in the repository.