

<h1 align="center"> <br>빵 RD(Respiratory Disease)?</h1>
<h3 align="center"> 🥨 Team bAIkery 🥨</h3>
<h5 align="center"> 조송희 | 오원진 | 장혜선 | 최정인 </h5>
<h5 align="center">
모두의 연구소 AIFFEL X 스마트사운드<br>
호흡음으로 질병 분류하기<br>
</h5>
<br>

<div align="center">
  

</div>

## Description
<div align="center">
  

</div>

- 진행기간: 22.04.25 ~ 06.10
- 개발 목적: 호흡음 데이터를 기반으로 정상음과 비정상음을 감별하여 질병을 분류하기 위함
- 개발 개요:
  
  호흡음은 공기의 흐름, 폐 내부의 조직 변화, 폐 내 분비물의 위치와 직접적인 관련이 있어 호흡기 건강 및 호흡기 질환의 중요 지표로 사용됩니다. 이러한 호흡음을 전문 기기 혹은 가정용 기기를 이용해 녹음합니다. 그리고 해당 호흡음을 기반으로 폐의 이상여부를 판단하고, 1차적으로 질병의 유무를 예측을 해봄으로써 질병 확산 방지에 도움을 줄 수 있습니다. 이와 같은 데이터를 이용해 호흡음 분류를 자동으로 진행하는 분석 모델을 구축합니다.
- 과정: 전처리 > 클래스 밸런싱 처리 > 특징 추출 > 모델링 > 결과 확인

## About Aiffelthon
**<평가 방식>** 

- [참조 소스](https://github.com/Shivam-316/Respiratory-Disease-Detection) 실행 후 결과 확인
- 다양한 모델 적용 후,  결과 비교
- 과업 : 질병분류 모델 정확도(accuracy)는 95% 이상 달성
- 민감도(sensitivity)와 특이도(specificity)를 계산할 것
- 추가 과업 : 소리에서 수포음(crackle)와 천명음(wheeze) 존재 유무 확인


## 1. Dataset 

- 데이터셋: [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) 
- 10초에서 90초의 920개의 녹음 파일(126명의 환자) 수록
- 각각의 녹음 파일에 대한 주석이 포함됨
- 6898개의 호흡 주기를 포함해 총 5.5시간 분량의 녹음 기록 
- 1864개에는 수포음(crackle), 886개에는 천명음(wheeze), 506개에는 수포음(crackle) 와 천명음(wheeze) 이 모두 포함되어 있음. 
- wheeze - High-pitched, **100 -2500Hz**의 주파수 대역과 **80msec 이상의 지속시간**
- crackle - High-pitched, crackle의 **지속시간**은 **20 ms보다 더 낮고** 주파수 대역은 **100와 200 Hz** 사이
- 데이터에는 깨끗한 호흡음만을 포함한 녹음 파일과 실제 생활 조건을 반영해 소음이 포함된 녹음 파일이 있음. 
- 환자는 모든 연령대를 포함

## 2. 전처리
- down sampling(16000Hz)
- zero padding
- 5th Butterworth filter

## 3. 클래스 밸런싱 처리
- smart padding + duplicated padding
- concatenation-based augmentation
- train/test split(stratify=normal/crackle/wheeze/both 클래스 비율대로)
- augmentation(shiftaug, speedaug)

## 4. 특징 추출
- mel feature extraction
- blank region clipping
- frequency/time masking

## 5. 모델링
- transfer learning(ResNet34)
- ResNet

## 6. 결과 확인 (test.py)
1) Task 1 : Pulmonary Disease classification

<p align="center"><img width="392" alt="pulmonary_disease_classification_report" src="https://user-images.githubusercontent.com/97088101/172894322-12074c59-ad19-4a22-8451-86bf03397f3c.png"></p>


2) Task 2 : Abnormal Lung sounds classification (additional)
 - First_trial without smart_padding
<img width="527" alt="first_trial_simple_CNN(2)" src="https://user-images.githubusercontent.com/97088101/172894713-acee990e-5e25-4ac6-842d-de0435423313.png">

 <img width="234" alt="first_trial_simple_CNN(1)" src="https://user-images.githubusercontent.com/97088101/172894752-4edb56bc-7a36-4432-91d7-a031224caf39.png">

 - Seocnd_trial with smart_padding
 <img width="518" alt="second_trial_with_smart_padding(1)" src="https://user-images.githubusercontent.com/97088101/172894805-72c5f781-13fd-4a1f-9db1-8cf2b27bae09.png">
<img width="230" alt="second_trial_with_smart_padding(2)" src="https://user-images.githubusercontent.com/97088101/172894831-60cf3552-4c20-4822-91d0-d12bfbff998b.png">

 
## 7. 결 론

- Medlcal data를 첫 번째 프로젝트로 진행하며, sound data에 대한 전반적인 이해와 다양한 pre-processing 방법을 시도해보는 것을 목표로 하였습니다.
- RespireNet 논문 내용을 토대로, concantenated augmentation을 포함한, 다양한 augmentation 기법을 시도해 보았습니다.
- 첫번째 과업인, 질병분류 모델에서 단순히, Resnet CNN model로 바꿈으로써 추가적 전처리 없이 96% 에 달성하는 성과를 보였습니다.
- 두번째 과업에서는 Resnet 모델에서 보다 simple 한 CNN model이 더 높은 성능을 보였습니다.
- Wheezing 및 Crackle sounds classification model에서 높은 성능을 보기 힘든 이유는, 본 비정상적인 호흡들은 fine -> moderate -> coarse한 단계를 가지며, 주파수 영역대 또한 비슷하게 포함하고 있어 높은 성능을 확인하기까지는 많은 어려움이 있음으로 판단됩니다.
- Smart_padding 을 구현하여 train셋으로 활용하여 훈련한 결과, wheezing & crackle sound가 둘 다 있는 both class에 대해 높은 recall & f1-score를 보였으나, wheezing class에 대해서 점수가 낮아지는 것을 확인하였습니다.
- 해커톤 발표를 통하여, 뒤늦게, data split을 데이터양 기준이 아닌, patient_id(PID)를 기준으로 나누어야 모델의 성능도 향상될 수 있음을 알게 되었으며, 향후 다양한 model building 과 함께 추가적인 test를 해볼 예정입니다.


## 8. 참고 논문
- A Window Width Optimized S-Transform(2008)
- Classification of lung sounds using convolutional neural networks(2017)
- respireNet(2020)
- DC-UNet: Rethinking the U-Net Architecture with Dual Channel Efficient CNN for Medical Images Segmentation(2020)
- LungRN+NL: An Improved Adventitious Lung Sound Classification Using Non-Local Block ResNet Neural Network with Mixup Data Augmentation(2020)
- FILTERAUGMENT: AN ACOUSTIC ENVIRONMENTAL DATA AUGMENTATION METHOD(2021)


<br>
<br>
<br>
