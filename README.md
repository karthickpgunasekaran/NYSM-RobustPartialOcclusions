# Now You See Me : Robust Approach to Partial Occlusions
Occlusions of objects is one of the indispensable problems in Computer vision. While Convolutional Neural Networks (CNNs) provide various state of the art approaches for regular image classification, they however, prove to be not as effective for the classification of images with partial occlusions. Partial occlusion is scenario where an object is occluded partially by some other object/space. This problem when solved, holds tremendous potential to facilitate various scenarios. We in particular are interested in autonomous driving scenario and its implications in the same. Autonomous vehicle research is one of the hot topics of this decade, there are ample situations of partial occlusions of a driving sign or a person or other objects at different angles.Considering its prime importance in situations which can be further extended to video analytics of traffic data to handle crimes, anticipate income levels of various groups etc., this holds the potential to be exploited in many ways. In this paper, we introduce our own synthetically created dataset by utilising Stanford Car Dataset and adding occlusions of various sizes and nature to it. On this created dataset, we conducted a comprehensive analysis using various state of the art CNN models such as VGG-19, ResNet 50/101, GoogleNet, DenseNet 121. We further in depth study the effect of varying occlusion proportions and nature on the performance of these models by fine tuning and training these from scratch on dataset and how is it likely to perform when trained in different scenarios, i.e., performance when training with occluded images and unoccluded images, which model is more robust to partial occlusions and so on.

# Data preprocessing
For detailed understanding about the dataset and preprocessing kindly checkout the Project report [here](Report/now_you_see_me_robust_approach_to_partial_occlusions.pdf)

a) Figure Below shows the different occlusion propotions used in the dataset
![Different Occ sizes](Images/diff_occ_size_nocap.png)

b) Figure belows demonstrate the different types of artifacts used in the dataset
![Different artifacts](Images/diffartifacts.png)

c) Following figure shows the Dataset statistics

![Data stats](Images/dataset_stat.png)

# Results 
For detailed results and analysis please checkout the project report [here](Report/now_you_see_me_robust_approach_to_partial_occlusions.pdf)

a) Table belows shows the effect of training on Lower occlusion data and testing on higher occulusion data.


b) Table below Effect of training on Higher occlusion Datasets and Testing on Lower occlusion Datasets


c) Table below shows Model performance across different Occlusions Sizes


d) Table below shows the Effect of network depth on model performance for different artifacts with 13 % occ


e) Table below Pretrained (PT) vs Not pre-trained(NPT) models for 13% occlusion and artifact type 1



# Contributors 
1. [Karthick Gunasekaran](https://www.linkedin.com/in/karthickpgunasekaran/)
2. [Nikita Jaiman](https://www.linkedin.com/in/nikitajaiman/)
