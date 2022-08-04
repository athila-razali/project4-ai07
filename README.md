# Cell Nuclei Segmentation using U-Net
## Summary
**Aim** : to develop a model for semantic segmentation of cell neuclei-containing images.       
**Data Source** : [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)
This dataset contains a large number of segmented nuclei images. There are two subfolders containing: 1. inputs contains image file; 2. masks contains segmented masks of each nucleus.

## Methodology
### 1. Data Pipeline
The dataset files come in the form of image inputs and image mask labels, with a train folder for training data and a test folder for testing data. 
The input images go through feature scaling during preprocessing. The labels are preprocessed to have binary values of 0 and 1. 
The ratio of the train-validation sets to the train data is 80:20.

### 2. Model Pipeline
U-Net is the model architecture utilised for this project. The model is designed to tackle challenges in biomedical semantic segmentation and 
it requires less training samples compared to other architectures, due to excessive image augmentation.          
In overview, the model consists of a contracting path (left side) and an expanding path (right side).
The contracting path aligns to the standard convolutional network architecture.
The decoder, also known as the expanding path, allows for exact localisation utilising transposed convolutions.
The graphic of the model architecture is in the figure below:         
<img width="491" alt="image" src="https://user-images.githubusercontent.com/91872382/182801138-ee673c28-75a2-4d4e-a6d1-1121a066f78e.png">
          
The model is trained with a batch size of 16 and 100 epochs. The callback also applied to help display results during model training. 
Figure below shows the sample prediction after epoch 96:          
<img width="516" alt="image" src="https://user-images.githubusercontent.com/91872382/182803072-12a43e94-2d95-4e24-89cd-96542de619bd.png">
          
## Results
The model evaluation with test data is       
<img width="496" alt="image" src="https://user-images.githubusercontent.com/91872382/182803491-72514f5c-134e-40b6-8f1b-3d63b89494ee.png">
        
The segmentation of the cell neuclei by the model is generally very accurate.               
Using some of the test data, the model also makes some predictions. The graphics below display the actual output masks and predicted masks.
<img width="433" alt="image" src="https://user-images.githubusercontent.com/91872382/182803848-58f9589e-66aa-4bce-b075-ce0a2b22c01f.png">          
<img width="427" alt="image" src="https://user-images.githubusercontent.com/91872382/182804014-a4b3e2b6-bf8f-497c-a184-6470d0331f2a.png">


### Credits
**Instructor : Kong Kah Chun         
Selangor Human Resource Development Center**
