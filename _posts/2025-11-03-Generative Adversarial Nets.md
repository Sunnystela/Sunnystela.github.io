---
layout: post
title:  "[논문정리] Generative Adversarial Nets" 
date:   2025-11-03 20:35
categories: MyStudy AI
tags: CV GAN
---

전에 읽긴 했지만 정리해 둔 것이 없길래 정리 겸 다시 읽었다. 

>  Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
 Sherjil Ozair, Aaron Courville, Yoshua Bengio(2014). Generative Adversarial Nets. arXiv.
{: .prompt-tip }

Generative Adversarial Nets(GANs)라는 새로운 프레임워크를 제안하며 이는 생성 모델 G와 판별 모델 D를 경쟁시키는 방식으로 작동한다. 위조범과 경찰처럼 두 모델이 서로를 상대로 훈련하며 데이터 분포를 정확하게 복제하는 방법을 제시한다. 기존의 복잡한 확률 계산 없이 역전파(backpropagation) 알고리즘만으로 훈련이 가능하다는 가치가 있다. 

# GANs 프레임워크 개요
목표: 생성 모델이 데이터 분포를 정확하게 복제하도록 하는 새로운 프레임워크를 제안한다. 