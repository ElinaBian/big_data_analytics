# My Way on Big Data Analytics

# 1. Big Data Tools

We all know that if we want to build our environment on google cloud to deal with big data, we need to install Hadoop, Hive, Hbase, and Spark. Here, I posted a tutorial for beginners and my first try on using these tools in the path ./begin_with_bigdata_tools

# 2. MLlib

We all know that MLlib fits into Spark's APIs and interoperates with NumPy in Python (as of Spark 0.9) and R libraries (as of Spark 1.5). It's a significantly important machine learning tools dealing with large datasets. Here, I provide some basic knowledge and examples using MLlib to do recommendations, regressions, classifications, and clustering.

# 3. Advanced Big Data Analysis

Here, I use Spark Streaming and Twitter API to implement your Sentiment Analysis system. And I also carried out basic graph analysis on real wiki data.

Then, the topic comes into an important area of big data, that is visualization. Here, I give out simple examples to use d3 to visualize clustering results, which would give out intuitional interpretation to clustering results. 

Also, I use d3 to make bar plot to show the real-time stream of Tweets, which would help us see real-time change of big data.

Finally, I visualize the graph analysis using a simple example with real wiki data.

# 4. FashionAI

I carried out a project on fashionAI with two of my friends, Wenshan Wang and Xiuqi Shao. The details about the problem we worked on could be found [here](https://www.alibabacloud.com/zh/campaign/fashionai). 

Detecting detailed apparel attributes is a topic receiving increasing attentions, which also has wide applications. Recent year, the demands of online shopping for fashion items grow a lot, which raises problems such as the sellers provide information not consistent with the real stuff, different sellers have inconsistent understandings of apparel styles. An automatic fashion attributes detection system can help overcome these problems by providing precise and consistent taggings or descriptions of apparel from their pictures. This technique can be applied to various areas such as apparel image searching, navigating tagging, and mix-and-match recommendation, etc.

[Here](https://youtu.be/0_sKvq5NxpY) is a short video about our project.


4.1 Data Preparation
----------------
Our data are from Alibaba Tianchi Competition. Anyone who registered for the competition could download the data [here](https://tianchi.aliyun.com/getStart/information.htm?spm=5176.100067.5678.2.77b655052XICWe&raceId=231670).

4.2 Training Process
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

4.3 Prediction Results
----------------

We saved our method of combining 8 models and giving out final resualts of prediction in prediction.py

And we gave out prediction results for several images in Combined_Prediction.ipynb

4.4 Model Saving
----------------

We saved our 8 trained models [here](https://drive.google.com/open?id=1ym7w3cqBFTlIRnS_37CdWtgPLgVP3ykG) and you could get access to them with lionmail account.

4.5 More Detailed about Our Project
----------------

We have our report [here](https://drive.google.com/open?id=1jbXtrMNsQfmIPntXiXSqGXcDegh85W6lDKHE-RjWZYU), and you could get access to it with lionmail account. We give more detailed description of data, training models and results in it.
