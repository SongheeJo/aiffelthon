
## I. **RESPIRENET: A DEEP NEURAL NETWORK FOR ACCURATELY DETECTING ABNORMAL LUNG SOUNDS IN LIMITED DATA SETTING**

## Abstract

 - Auscultation of respiratory sounds is the primary tool for screening and diagnosing lung diseases. Automated analysis, coupled with digital stethoscopes, can play a crucial role in enabling tele-screening of fatal lung diseases. Deep neural networks (DNNs) have shown a lot of promise for such problems, and are an obvious choice. However, DNNs are extremely data hungry, and the largest respiratory dataset ICBHI has only 6898 breathing cycles, which is still small for training a satisfactory DNN model. In this work, RespireNet, we propose a simple CNN-based model, along with a suite of novel techniques—device specific fine-tuning, concatenation-based augmentation, blank region clipping, and smart padding—enabling us to efficiently use the small-sized dataset. We perform extensive evaluation on the ICBHI dataset, and improve upon the state-of-the-art results for 4-class classification by 2.2%.

- Index Terms : Abnormality detection, lung sounds, crackle and wheeze, ICBHI dataset, deep learning

- Data Augmentation : device specific fine-tuning, concatenation-based augmentation, blank region clipping, smart padding

### Summary

    - 의료 데이터는 구하기 어렵다. 
    - DNN도 성능이 좋을 것 같지만, 훈련시키려면 데이터가 부족하다. (Deep Neural Netwark는 훈련시키는 데에 많은 데이터가 필요하기 때문)
    - 본 논문에서는 심플한 CNN을 베이스로 한 모델인 `RespireNet`를 제안함.
    - 데이터 증강법 : device specific fine-tuning, concatenation-based augmentation, blank region clipping, smart padding 을 이용함.


### RespireNet Framework

![image](https://user-images.githubusercontent.com/67695343/166189313-a54288ff-39d8-407c-8937-d7da21e7e4b8.png)

> Summary <br>
사운드 신호 전처리(bandpass filtering, downsampling, normalization, etc., ...) → concatenation-based 증강 → smart padding → mel-spectrogram 생성 → blank region clipping → 처리된 파형이미지를 모델에 넣음 → 모델 훈련 1단계: 훈련 세트 전체를 이용 → 2단계: 파인튜닝, 데이터중 각각의 장비에 맞는 부분만 사용하여 훈련!

### CONCLUSION AND FUTURE WORK

- The paper proposes RespireNet a simple CNN-based model, along with a set of novel techniques—device specific fine-tuning, concatenation-based augmentation, blank region clipping, and smart padding—enabling us to effectively utilize a small-sized dataset for accurate abnormality detection in lung sounds. Our proposed method achieved a new SOTA for the ICBHI dataset, on both the 2-class and 4-class classification tasks. Further, our proposed techniques are orthogonal to the choice of network architecture and should be easy to incorporate within other frameworks. The current performance limit of the 4-class classification task can be mainly attributed to the small size of the ICBHI dataset, and the variation among the recording devices. Furthermore, there is lack of standardization in the 80-20 split and we found variance in the results based on the particular split. In future, **we would recommend that the community should focus on capturing a larger dataset, while taking care of the issues raised in this paper.**


![image](https://user-images.githubusercontent.com/67695343/166189777-b866bdb8-fda4-43cb-a3dd-f88ee07eda27.png)

 - concatenation-based augmentation (CBA), blank region clipping (BRC) and device specific fine-tuning (FT)

### Summary

    1. 작은 양의 ICBHI 데이터셋, 그리고 녹음기들 사이의 다양화
    2. 특정한 split 비율에 근거하여 결과값에 variance(변화(량))를 찾음
    3. not 80-20 split! → 80-20은 표준화(standardization)하기엔 부족
    4.  **이 논문에서 제기된 이슈 주의, 데이터셋의 양을 늘리는 거 자체에 집중하기를 추천**


## 논문에서 소개된 Sound data processing methods

### **METHOD**

Dataset*: We perform all evaluations on the ICBHI scientific challenge respiratory sound dataset. It is one of the largest publicly available respiratory datasets. The dataset comprises of 920 recordings from 126 patients with a combined total duration of 5.5 hours. Each breathing cycle in a recording is annotated by an expert as one of the four classes: normal, crackle, wheeze, or both (crackle and wheeze). The dataset comprises of recordings from four different devices from hospitals in Portugal and Greece. For every patient, data was recorded at seven different body locations.

*Pre-processing*: The sampling rate of recordings in the dataset varies from 4 kHz to 44.1 kHz. To standardize, we down-sample the recordings to 4 kHz and apply a 5-th order Butterworth band-pass filter to remove noise (heartbeat, background speech, etc.). We also apply standard normalization on the input signal to map the values within the range (-1, 1). The audio signal is then converted into a Mel-spectrogram, which is fed into our DNN.

*Network architecture*: We use a CNN-based network, ResNet34, followed by two 128-d fully connected linear layers with ReLU activations. The last layer applies softmax activation to model classwise probabilities. Dropout is added to the fully connected layers to prevent overfitting. The network is trained via a standard categorical cross-entropy loss to minimize the loss for multi-class classification. The overall framework and architecture is illustrated in Figure 1.

### **Efficient Dataset Utilization**
Even though ICBHI is the largest publicly available dataset with 6898 samples, it is still relatively small for training DNNs effectively. Thus, a major focus of our work has been to develop techniques to efficiently use the available samples.

 We extensively analyzed the dataset to identify dataset characteristics that inhibit training DNNs effectively, and propose solutions to overcome the same.

The first commonly used technique we apply is transfer learning, where we initialize our network with weights of a pre-trained ResNet-34 network on ImageNet [23]. This is followed by our training where we train the entire network end to-end. Interestingly, even though ImageNet dataset is very different from the spectrograms which our network sees, we still found this initialization to help significantly. Most likely, low level features such as edge-detection are still similar and thus “transfer” well.

**Concatenation-based Augmentation**: Like most medical datasets, ICBHI dataset has a huge class imbalance, with the normal class accounting for 53% of the samples. To prevent the model from overfitting on abnormal classes, we experimented with several data augmentation techniques. We first apply standard audio augmentation techniques, such as noise addition, speed variation, random shifting, pitch shift, etc., and also use a weighted random sampler to sample mini-batches uniformly from each class. These standard techniques help a little, but to further improve generalization of the underrepresented classes (wheeze, crackle, both), we developed a concatenation based augmentation technique where we generate a new sample of a class by randomly sampling two samples of the same class and concatenating them (see Figure 2). This scheme led to a non-trivial improvement in the classification accuracy of abnormal classes.


![image](https://user-images.githubusercontent.com/67695343/166191431-7f9be84c-c8f3-4e42-a656-0a92a40e115f.png)
Fig. 2. Proposed concatenation-based augmentation.

**Smart Padding**: The breathing cycle length varies across patients as well as within a patient depending on various factors (e.g., breathing rate can increase moderately during fever). In the ICBHI dataset, the length of breathing cycles ranges from 0.2s to 16.2s with a mean cycle length of 2.7s. This poses a problem while training our network as it expects a fixed size input. The standard way to handle this is to pad the audio signal to a fixed size via zero-padding or reflection based padding. We propose a novel smart padding scheme, which uses a variant of the augmentation scheme described above. For each data sample, smart padding first looks at the breathing cycle sample for the same patient taken just before and after the current one. If this neighbouring cycle is of the same class or of the normal class, we concatenate the current sample with it. If not, we pad by copying the same cycle again. We continue this process until we reach our desired size. This smart padding scheme also augments the data and helps prevent overfitting. We experimented with different input lengths, and found a 7s window to perform best. A small window led to clipping of samples, thus loosing valuable information in an already scarce dataset, while a very large window caused repetition leading to degraded performance.

![image](https://user-images.githubusercontent.com/67695343/166191602-c1f5f9ee-d7fe-4c72-a44e-befe07a56170.png)
Fig. 3. Blank region clipping: The network attention starts focusing more on the bottom half of the spectrogram, instead of blank spaces after clipping.

**Blank Region Clipping**: On analyzing samples using GradCam++ which our base model mis-classified, we found notable black regions at higher frequency regions of their spectrograms (Figure 3). On further analysis, we found that many samples, and in particular 100% of the Litt3200 device samples, had blank region in the 1500-2000Hz frequency range. Since this was adversely affecting our network performance, we selectively clip off the blank rows from the high frequency regions of such spectrograms. This ensures that the network focuses on the region of interest leading to improved
    performance. Figure 3 shows this in action.

**Device Specific Fine-tuning**: The ICBHI dataset has samples from 4 different devices. We found that the distribution of samples across devices is heavily skewed, e.g. the AKGC417L Microphone alone contributes to 63% of the samples. Since each device has different audio characteristics, the DNN may fail to generalize across devices, especially for the underrepresented devices in the already small dataset. To verify this, we divided the test set into 4 subsets depending on their device type, and compute the accuracy of abnormal class samples in each subset. As expected, we found the classification accuracy to be strongly correlated with the training set size of the corresponding device. To address this, we first train a common model with the full training data (stage-1, Figure 1). We then make 4 copies of this model and fine-tune (stage-2) them for each device separately by using only the subset of training data for that device. We found this approach to significantly improve the performance, especially for the underrepresented devices.



**Chest location: Trachea (Tc),  Anterior left (Al), Anterior right (Ar), Posterior left (Pl), Posterior right (Pr), Lateral left (Ll), Lateral right (Lr)**

---

### 데이터셋의 몇 가지 특성

 In order to efficiently use the available data, we did extensive analysis of the ICBHI dataset. We found several characteristics of the data that might inhibit training DNNs effectively. For example, the dataset contains audio recordings from four different devices, with skewed distribution of samples across the devices, which makes it difficult for DNNs to generalize well across devices. Similarly, the dataset has a skewed distribution across normal and abnormal classes, and varying lengths of audio samples. We propose multiple novel techniques to address these problems—device specific fine-tuning, concatenation-based augmentation, blank region clipping, and smart padding. We perform extensive evaluation and ablation analysis of these techniques.

### Summary
    - 위 데이터셋의 특성은 DNN을 효과적으로 돌리기 어렵게 함 
    - 소리 샘플에 녹음기마다 서로 다른 왜곡된 분포 O  → 모델이 일반화된 학습을 하기에는 어렵다
    - normal / abnormal 클래스에 서로 다른 왜곡된 분포 + 서로 다른 샘플 길이
    - 데이터셋을 효율적으로 사용하고자 만든 간단한 호흡분류기 네트워크 구조와 기법들
    - 이 논문에서 소개되는 기법들은 여기서 사용된 네트워크 구조 뿐만아니라 다른 네트워크에도 쉽게 포함될 수 있도록 고안됨
    
    
    

# II. An improved adventitious lung sound classification using non-local block
: resnet neural network with mixup data augmentation으로서 98%를 달성함.

### [참고] I. RespireNet (PPT논문) 내용 중 4. RELATED WORK

Recently, there has been a lot of interest in using deep learning models for respiratory sounds classification [1, 9, 12]. It has outperformed statistical methods (HMM-GMM) [8] and traditional machine learning methods (boosted decision trees, SVM) [4, 24]. In these deep learning based approaches, a time-frequency representation of the audio signal is provided as input to the model. Kochetov et al. [9] propose a deep recurrent network with a noise masking intermediate step for the four class classification task, obtaining a score of 65.7% on the 80-20 split. However the paper omits the details regarding noise label generation [1], thus making it hard to reproduce. Deep residual networks and optimized S-transform based features are used by Chen et al. [6] for three-class classification of anomalies in lung sounds. The model is trained and tested on a smaller subset of the ICBHI dataset on a 70-30 split and achieve a score of 98%.

---

### (1) 호흡음 분류에 딥러닝을 사용한 최근 연구

- Jyotibdha Acharya and Arindam Basu. Deep neural network
for respiratory sound classification in wearable devices enabled by patient specific model tuning. IEEE Transactions on
Biomedical Circuits and Systems, PP:1–1, 03 2020.
- Kirill Kochetov, Evgeny Putin, Maksim Balashov, Andrey
Filchenkov, and Anatoly Shalyto. Noise Masking Recurrent
Neural Network for Respiratory Sound Classification: 27th International Conference on Artificial Neural Networks, Rhodes,
Greece, October 4–7, 2018, Proceedings, Part III, pages 208–
217. 10 2018. ISBN 978-3-030-01423-0.
- Yi Ma, Xinzi Xu, and Yongfu Li. Lungrn+nl: An improved
adventitious lung sound classification using non-local block
resnet neural network with mixup data augmentation. 08 2020.


### Abstract

Performing an automated adventitious lung sound detection is a challenging task since the sound is susceptible to noises (heart-beat, motion artifacts, and audio sound) and there is subtle discrimination among different categories. An adventitious lung sound classification model, LungRN+NL, is proposed in this work, which has demonstrated a drastic improvement compared to our previous work and the state-of-the-art models. This new model has incorporated the non-local block in the ResNet architecture. To address the imbalance problem and to improve the robustness of the model, we have also incorporated the mixup method to augment the training dataset. Our model has been implemented and compared with the state-of-the-art works using the official ICBHI 2017 challenge dataset and their evaluation method. As a result, `**LungRN+NL**` has achieved a performance score of 52.26%, which is improved by 2.1-12.7% compared to the state-of-the-art models.

### Summary
   - Deep-learning based model은 이전에 사용되던 통계적 기법이나 전통적인 머신 러닝 기법(Boosting dedcision tree, SVM)보다 나은 결과를 냄. 
   - 딥러닝 모델들은 소리 신호를 파형이미지로 바꾸어 모델에게 인풋데이터로 투입함. (중략) 4개 분류에 사용되는 deep recurrent nework는 65.7%(80-20split). 
   - Deep residual networks와 특성들을 베이스로 최적화된(optimized) S-transform은 클래스 3개 분류에 사용됨. (이 모델은 70-30 분할로 ICBHI 데이터셋 중 더 작은 부분만 사용했고, 98%의 점수를 성취하였음.)
   - LungRN+LN 이라는 모델에 RespireNet 논문과 같은 데이터셋, 평가 방법을 사용한 사례이며, 성능이 52.26%에서 2.1~12.7% 범위로 개선되어 시도해볼 법한 모델로 판단됨.

**Index Terms**: adventitious lung sounds classification, mixup, data augmentation, convolutional neural network, non-local block



# III. Triple-Classification of Respiratory Sounds Using Optimized S-Transform and Deep Residual Networks

- RESPIRENET related works에서 98%라는 더 좋은 성과를 낸 논문이며, optimized S-transform(OST) 와 deep residual networks(ResNets)을 사용하여 wheeze, crackle and normal 을 분류하였음.

### Introduction
> Deep residual networks and optimized S-transform
based features are used by Chen et al. [6] for three-class classification of anomalies in lung sounds. The model is trained
and tested on a smaller subset of the ICBHI dataset on a 70-30 split and achieve a score of 98%
> 

### Problems

> However, due to the contained artifacts and constrained feature extraction methods, the reliability and accuracy of the classification of wheeze, crackle, and normal sounds need significant improvement


### Procedures

> raw respiratory sound 가 OST를 사용하여 processed → spectogram of OST 가 Resnet을 위해 rescaled → ResNet을 통해 feature learning 과 classification 을 하고 respiratory sound 의 클래스를 recognize하게 함.
