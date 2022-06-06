

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
- 정확도(accuracy)는 95% 이상이어야 함.
- 민감도(sensitivity)와 특이도(specificity)를 계산할 것
- 추가 과업 : 소리에서 수포음(crackle)와 천명음(wheeze) 존재 유무 확인


## 1. Dataset 

- 데이터셋: [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) 
- 10초에서 90초의 920개의 녹음 파일(126명의 환자) 수록
- 각각의 녹음 파일에 대한 주석이 포함됨
- 6898개의 호흡 주기를 포함해 총 5.5시간 분량의 녹음 기록 
- 1864개에는 수포음(crackle), 886개에는 천명음(wheeze), 506개에는 수포음(crackle) 와 천명음(wheeze) 이 모두 포함되어 있음. 
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
- f1 score, recall

<br>
<br>
<br>
