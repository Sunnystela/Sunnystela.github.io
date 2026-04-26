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

**발표 멘트**

* IdentityNet은 얼굴 디테일을 유지하는 핵심 모듈입니다.
* 실제 사람처럼 보이게 만듭니다.

---

## 14. Spatial Control

**슬라이드 내용**

* 5 keypoints

**발표 멘트**

* 얼굴의 위치 정보는 5개의 keypoint로 제한합니다.
* 너무 강한 제약을 피하기 위한 선택입니다.

---

## 15. 학습 전략

**슬라이드 내용**

* 기존 모델 freeze

**발표 멘트**

* 기존 diffusion 모델은 학습하지 않습니다.
* 새로운 모듈만 학습합니다.

---

## 16. 추론 과정

**슬라이드 내용**

* one-step inference

**발표 멘트**

* 추론은 매우 빠릅니다.
* 한 번의 forward로 결과를 생성합니다.

---

## 17. 결과

**슬라이드 내용**

* 높은 유사도

**발표 멘트**

* 실험 결과, 얼굴 유사도가 매우 높게 유지됩니다.

---

## 18. 비교 (IP-Adapter)

**슬라이드 내용**

* fidelity 향상

**발표 멘트**

* 기존 방법보다 더 자연스럽고 정확한 얼굴을 생성합니다.

---

## 19. 비교 (LoRA)

**슬라이드 내용**

* 학습 없이 성능 확보

**발표 멘트**

* LoRA처럼 학습이 필요한 방법과 비교해도 경쟁력 있는 결과를 보입니다.

---

## 20. 활용

**슬라이드 내용**

* 프로필
* 캐릭터 생성

**발표 멘트**

* 다양한 분야에 활용 가능합니다.
* 특히 개인화 콘텐츠 생성에 적합합니다.

---

## 21. 추가 활용

**슬라이드 내용**

* interpolation

**발표 멘트**

* 여러 사람의 얼굴을 섞거나 새로운 시점을 생성할 수도 있습니다.

---

## 22. 장점

**슬라이드 내용**

* 빠름
* 정확함

**발표 멘트**

* 속도와 정확도를 동시에 만족합니다.

---

## 23. 한계

**슬라이드 내용**

* bias

**발표 멘트**

* 다만 얼굴 속성 분리가 어렵고 bias 문제가 존재할 수 있습니다.

---

## 24. 결론

**슬라이드 내용**

* 효율 + 성능

**발표 멘트**

* InstantID는 효율성과 성능을 동시에 달성한 모델입니다.

---

## 25. 향후 연구

**슬라이드 내용**

* 개선 방향

**발표 멘트**

* 앞으로는 속성 분리와 윤리적 문제 해결이 필요합니다.
