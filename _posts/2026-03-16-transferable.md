---
layout: post
title: "[ing][논문정리] Learning Transferable Visual Models From Natural Language Supervision"
date: 2026-03-16 12:50
categories: MyStudy
tags: AI language
math: true
---


>Learning Transferable Visual Models From Natural Language Supervision(2021).Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.arXiv
{: .prompt-tip }



## 1. Introduction and Motivating Work
NLP에서의 대규모, 작업-무관(pretraining) 성과를 비전 분야로 옮겨와, 인터넷의 방대한 이미지-텍스트 쌍을 이용한 natural language supervision으로 범용적이고 제로샷 전이 가능한 시각 모델을 만들자는 동기와 배경을 설명한다.


저자들은 NLP에서 보인 "웹 규모의 텍스트 기반 사전학습 → 제로샷 전이" 패러다임이 시각 분야에도 적용될 수 있으며, 이를 위해 인터넷에서 수집한 대규모 이미지-텍스트 쌍(400M)을 사용하고 이미지와 텍스트 임베딩을 contrastive 학습하는 CLIP을 제안했다.


전통적 비전 모델은 고정된 class 라벨로 학습되므로 새로운 개념에 쓰려면 추가 레이블이 필요하지만 자연어 감독은 풍부한 개념 표현(자연어)을 그대로 학습 신호로 쓸 수 있어 훨씬 범용적이고 제로샷 활용이 가능하다.



NLP에서의 성공 사례(예: autoregressive/masked language modeling, GPT 계열 등)가 "작업-무관" 사전학습과 제로샷 전이의 실용성을 입증했다.
이 성공이 시각 분야에도 적용될 수 있는지를 묻고, 자연어-이미지 짝(pair)에서 직접 학습하는 방향을 제시함.


관련 선행연구 정리:
오래전부터 이미지와 텍스트를 연결하려는 시도(예: Mori et al., 1999)와, 캡션(또는 태그)으로부터 시각 표현을 학습한 연구들(Joulin et al., 2016 등)이 존재함.
최근에는 transformer 기반 언어모델·대조학습 등을 이미지-텍스트에 적용한 VirTex, ICMLM, ConVIRT 같은 연구들이 등장했으나, 이들 대부분은 데이터/연산 규모가 작아 실제 성능에서 한계가 있었음.


문제 제기(동기):
기존 자연어 감독 기반 접근법들은 벤치마크 성능에서 부족했고(예: Zero-shot ImageNet에서 낮은 정확도), 데이터·모델·연산 규모의 부족이 주요 원인으로 보임.
반면 대규모 웹 이미지(예: Instagram 수십억 건)를 이용한 약지도(weak supervision) 연구들은 좋은 성능을 보였지만, 이들은 여전히 클래스 수를 제한하거나(static softmax) 제로샷 유연성이 떨어짐.


저자들의 제안(요약):
규모의 격차를 메우기 위해 WIT(WebImageText)라는 4억 쌍의 이미지-텍스트 데이터셋을 만들고, 이미지 인코더와 텍스트 인코더를 대조 학습(배치 내 N×N 쌍 중 올바른 쌍만 유사도를 높이는 방식)으로 공동 학습하는 CLIP을 제안.
이렇게 학습된 텍스트 인코더를 통해 자연어로 클래스(또는 설명)를 주면 제로샷으로 분류기를 구성할 수 있음(텍스트가 분류기 가중치를 생성하는 하이퍼네트워크 역할).



핵심 용어 정리 (간단)

자연어 감독 (natural language supervision): 이미지와 연관된 자연어(캡션, 제목, 설명)를 직접 학습 신호로 쓰는 것.
제로샷 전이 (zero-shot transfer): 특정 다운스트림 데이터/태스크의 레이블을 본 적 없어도, 자연어 설명만으로 바로 수행하는 능력.
대조 학습 (contrastive learning): 올바른 이미지-텍스트 쌍의 임베딩 유사도를 높이고, 잘못 짝지어진 쌍의 유사도는 낮추는 학습 목표.
WIT: CLIP 저자들이 구축한 400M 이미지-텍스트 쌍 데이터셋.

동기에서 연결되는 중요 인사이트

NLP에서의 성공 요인(대규모 데이터·모델·작업-무관적 목표)이 비전에도 적용될 수 있으며, 핵심은 "자연어가 가지는 광범위한 개념표현"을 직접 학습에 활용하는 것.
스케일(데이터·연산)이 작으면 자연어 감독의 잠재력이 실현되지 못하므로, 대규모 수집과 효율적 학습(여기서는 contrastive objective)이 필요하다는 점을 강조.

확장적 생각 (향후 연구·검토 포인트)

자연어 감독은 유연하지만, 텍스트 분포(웹 텍스트의 잡음·편향)에 의한 한계와 사회적 편향 문제를 동반함(논문 후반부에서 논의).
대조 학습은 효율적이나 생성적(캡션 생성 등) 모델의 유연성을 잃는다 — 두 접근을 결합하는 연구가 필요함.



## 2. Approach


### 2.1. Natural Language Supervision





### 2.2. Creating a Sufficiently Large Dataset


### 2.3. Selecting an Efficient Pre-Training Method


### 2.4. Choosing and Scaling a Model


### 2.5. Training


## 3. Experiments



### 3.1. Zero-Shot Transfer


### 3.2. Representation Learning



### 3.3. Robustness to Natural Distribution Shift


## 4. Comparison to Human Performance




## 5. Data Overlap Analysis



## 6. Limitations


## 7. Broader Impacts

### 7.1.Bias



### 7.2. Surveillance



## 8. Related Work



## 9. Conclusion





