# big_data_analytics


# FashionAI


Data Preparation
----------------
Our data are from Alibaba Tianchi Competition. Anyone who registered for the competition could download the data [here](https://tianchi.aliyun.com/getStart/information.htm?spm=5176.100067.5678.2.77b655052XICWe&raceId=231670).

Training Process
----------------
After downloading the data, we preprocessed the data and trained 8 models seperately for 8 tasks (4 tasks related to length: coat_length, skirt_lenth, sleeve_length, pant_length; 4 tasks related to design: neckline_design, collar_design, neck_design, lapel_design).

You could get access to the data preprocess, image augmentation process, training models, training and test accuracy in 8 notebooks:

inceptionV3_imgaug_lapel_design.ipynb

inceptionV3_imgaug_neckline_design.ipynb

inceptionV3_imgaug_pant_length.ipynb

inceptionV3_imgaug_sleeve_length.ipynb

resnet50_imgaug_coat_length.ipynb

resnet50_imgaug_collar_design.ipynb

resnet50_imgaug_neck_design.ipynb

resnet50_imgaug_skirt_length.ipynb

Prediction Results
----------------

We saved our method of combining 8 models and giving out final resualts of prediction in prediction.py

And we gave out prediction results for several images in Combined_Prediction.ipynb

Model Saving
----------------

We saved our 8 trained models [here](https://drive.google.com/open?id=1ym7w3cqBFTlIRnS_37CdWtgPLgVP3ykG) and you could get access to them with lionmail account.

More Detailed about Our Project
----------------

We have our report [here](https://drive.google.com/open?id=1jbXtrMNsQfmIPntXiXSqGXcDegh85W6lDKHE-RjWZYU), and you could get access to it with lionmail account. We give more detailed description of data, training models and results in it.
