---
layout: post
title: "[논문정리] InstantID: Zero-shotIdentity-Preserving Generation in Seconds"
date: 2026-04-26 12:50
categories: MyStudy
tags: AI video
math: true
---


>InstantID: Zero-shotIdentity-Preserving Generation in Seconds(2024).Qixun Wang, Xu Bai12, Haofan Wang, Zekui Qin12, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu.arXiv
{: .prompt-tip }

InstantID는 한 장의 얼굴 이미지만으로도 동일 인물을 다양한 스타일로 생성하는 기술이다.

![alt](/assets/img/id.fig1.png)

* 이미지 생성 기술 발전
* Diffusion 모델 등장
* 개인화 생성 수요 증가

최근 diffusion 기반 이미지 생성 모델이 급격히 발전했다.
특히 개인화된 이미지 생성, 즉 특정 인물을 유지하면서 이미지를 생성하는 기술이 중요해졌다.


하지만 얼굴 정체성을 유지하는 것이 매우 어렵다.
텍스트만으로는 얼굴 특징을 정확히 표현하기 어렵습니다.

기존에는 DreamBooth나 LoRA 같은 fine-tuning 기반 방법이 사용되었다.

이들은 높은 성능을 보이지만 학습이 필요하다.


시간도 오래 걸리고, 여러 장의 이미지가 필요하다.
실시간 사용에는 적합하지 않습니다.

두번째 방법으로 fine-tuning 없이 사용하는 방법도 등장했다.
대표적으로 IP-Adapter가 있다.


하지만 이 방법은 얼굴 유사도가 낮다는 문제가 있다.
같은 사람처럼 보이지 않는 경우가 많다.



이를 해결하기 위해 InstantID가 제안되었다.
핵심은 한 장의 이미지로 높은 정체성 유지하는 것이다.

Plug & Play, Zero-shot
으로 이 모델은 추가 학습 없이 바로 사용할 수 있다.

기존 diffusion 모델에 쉽게 붙일 수 있다.


![alt](/assets/img/id.fig2.png)


* Face Encoder
* Image Adapter
* IdentityNet

전체 구조는 세 가지 핵심 모듈로 구성된다.
각각 역할이 다르다.

**Face Encoder**

* 얼굴 특징 추출

먼저 얼굴 인코더는 사람의 얼굴 특징을 벡터로 변환한다.
이 정보가 정체성을 결정한다.



**Image Adapter**

* cross-attention

Image Adapter는 이 정보를 diffusion 모델에 전달한다.
cross-attention 구조를 사용한다.

**IdentityNet**

* 디테일 유지

IdentityNet은 얼굴 디테일을 유지하는 핵심 모듈이다.
실제 사람처럼 보이게 만든다.



Spatial Control에 대해서 보자. 

얼굴의 위치 정보는 5개의 keypoint로 제한한다.
너무 강한 제약을 피하기 위한 선택이다.

학습 전략은 
기존 모델 freeze이다.

기존 diffusion 모델은 학습하지 않는다.
새로운 모듈만 학습한다.

추론 과정으로 one-step inference이다.

추론은 매우 빠르다.
한 번의 forward로 결과를 생성한다.

![alt](/assets/img/id.fig3.png)



실험 결과, 얼굴 유사도가 매우 높게 유지된다.


![alt](/assets/img/id.fig5.png)

![alt](/assets/img/id.fig7.png)


다음 비교 (IP-Adapter)이다.


![alt](/assets/img/id.fig6.png)

기존 방법보다 더 자연스럽고 정확한 얼굴을 생성한다.

LoRA와 비교를 해보면 

학습 없이 성능 확보가 가능하다

LoRA처럼 학습이 필요한 방법과 비교해도 경쟁력 있는 결과를 보입니다.

속도와 정확도를 동시에 만족하는 장점이 있다.

다만 얼굴 속성 분리가 어렵고 bias 문제가 존재할 수 있다.

>ID 임베딩이 신원 뿐만 아니라 나이, 성별, 표정, 조명 등 여러 attribute가 함께 섞여 있기에 특정 속성만 골라 바꾸거나 제어하기 어렵고 사용된 얼굴 모델, 학습 데이터의 편향으로 특정 그룹에서 성능이 떨어지거나 편향된 결과가 나올 수 있다.

InstantID는 효율성과 성능을 동시에 달성한 모델이다.


