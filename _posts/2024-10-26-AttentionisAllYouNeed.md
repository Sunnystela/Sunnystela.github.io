---
layout: post
title:  "[CV] Week5. Attention is All You Need"
date:   2024-10-26 17:17
categories: KHUDA CV 
tag: CV
---

attention은 기존 인코더 디코더의 성능을 강화시키며 주목받고 있던 메커니즘이다. 이 논문에서 attention을 발표현 것이 아닌 RNN을 사용하지 않고 attention만으로도 입력 데이터에서 중요한 정보를 찾아 단어를 인코딩 할 수 있다는 것을 발표한 것이다. 

# Transformer 개요

RNN은 LSTM고 기존 자연어 처리 task에서 가장 많이 쓰이는 모델이다. 자연어 처리에서 문장을 처리할 때 문장에 사용된 단어의 위치가 중요하다. 단어의 위치오 순서 정보를 잘 홀용하는  RNN과 LSTM이 자주 활용되었다. 

그러나 RNN은 문장 길이가 길어지면 학습 능력이 떨어지게 된다. 

Transformer은 이러한 과정을 병렬적으로 처리하기 때문에 성능과 속도면에서 훨씬 뛰어나다. 

단어 위치 정보가 중요한데 RNN을 사용하지 않는다면 어떻게 단어 위치정보를 활용할까?

Positional Encoding을 활용할 수 있다. 


이 논문에서 순환없이 입력값과 출력값 간 의존성을 모델링할 수 있는 attention mechanism 만을 사용한 모델 구조인 transfer을 제안한다. 


# Background
 
연속적인 계산을 줄이기 위해 CNN을 기본 구성 요소로 사용한 연구가 이루어진다. 

이는 거리에 따라 계산량이 증가하고 입력값과 출력 값의 거리가 멀수록 의존성을 알기 어려워진다. 

반면 Trnasformer에서는 Multi-head attnetion을 통해 상수 시간의 계산만으로 가능하다. 

## self-attention

자신에게 수행하는 어텐션 기법으로 단일 시퀀스에서 서로 다른 위치에 있는 요소들의 의존성을 찾아낸다. 

## end-to-end memory network

간단한 언어 질문 답변 및 언어 모델링 작업에서 좋은 성능을 보인다.

대부분 자연어 처리는 RNN, CNN이 주로 사용된다. 문장의 순차적인 특성이 유지되지만 먼 거리에 위치하면 의존성을 알기 어려워진다. 

RNN은 시퀀스 길이가 길어질수록 정보 압축 문제가 존재한다. <br>
CNN은 합성곱 필터 크기를 넘어서는 문맥은 알기 어려워진다. 

반면 transfer은 순환없이 attention mechanism을 이용해 의존성을 찾을 수 있다. 
self-attention에만 의존한다.

# Model Architecture

## seq2seq구조

![alt](/assets/img/cv5.seq.png)

이전에 생성된 출력을 다음 단계에서 사용한다.
이전 단계가 완료도어야 다음 단계를 수행할 수 있는 것으로 병렬적으로 처리를 할 수 없다.

![alt](/assets/img/cv5.trans.png)

### Encoder and Decoder Stacks

인코더 : 6개의 동일한 레이어로 <br> 하나의 인코더는 Self-Attention layer와 Feed Forward Neural Network라는 두 개의 Sub layer로 이루어져 있다.<br>
디코더 : 6개의 동일한 레이어<br> 각 레이어는 인코더가 Sub layer로 가진 Self-Attention layer와 Feed Forward Neural Network 외에 하나의 레이어를 더 가진다

인코더의 stack 출력에 대해 Multi head Attention 수행한다


### Attention
![alt](/assets/img/cv5.atten.png)


Q: 영향을 받는 벡터<br>
K: 영향을 주는 벡터<br>
V: 주는 영향의 가중치 벡터



![alt](/assets/img/cv5.out.png)



output: value의 가중합으로 계산한다<br>
가중합에 이용되는 가중치: query와 연관된 key의 호환성 함수에 의해 계산된다

![alt](/assets/img/cv5.multi.png)

모델이 다양한 관점에서 문장을 해석할 수 있도록 해준다. 

모델이 Head 개수만큼의 Scaled dot product Attention 연산을 수행할 수 있게 하여 모델이 다양한 관점의 Attention Map을 만들게한다.




Encoder-decoder attention :
이전 decoder 레이어에서 오는 query들과 encoder의 출력으로 나오는 memory key, value들과의 attention<bn>
decoder의 모든 위치에서 input sequence의 모든 위치를 참조할 수 있게 해준다<br>
Self-attention in encoder :
Encoder의 각 위치들은 이전 레이어의 모든 위치들을 참조할 수 있게 해준다.<br>
Self-attention in decoder :
decoder의 각 위치들은 decoder 내의 다른 위치들을 참조할 수 있는데, 이전부터 자신 위치까지만을 참조할 수 있다.<br>
auto-regressive 성질을 살리면서도 정보가 잘못 흐르는 것을 막는다.



## Position-wise Feed-Forward Networks


![alt](/assets/img/cv5.forward.png)

ReU활성화가 있는 두 개의 선형 변환으로 구성된다.



## Embeddings and Softmax

input과 output 토큰을 embedding layer를 거쳐서 사용한다. 
<br>
생성된 embedded vector는 컴퓨터가 정보를 쉽게 수집할 수 있도록 만든 의미적인 태그들로 구성된 특성을 나타내게 된다.<br>



## Positional Encoding
토큰의 상대적인 위치에 대한 정보를 제공하기 위한 역할



Transformer는 Recurrent model을 사용하지 않고 오직 Attention mechanism만을 사용하여 만들기 때문에 Sequence 정보를 담아낼 수가 없다.


sequence 정보를 데이터에 추가해주는 것이 positional encoding이다.

# why self-attention

하나의 레이어 당 전체 연산 복잡도 감소
: RNN 계열에서는 하지 못했던 병렬 처리 연산의 양을 대폭 늘려 자연스레 학습 시간 감소한다.<br>
Long term dependency의 문제점 해결
: 트랜스포머 모델은 대응관계가 있는 토큰들 간의 물리적인 거리값들 중 최댓값이 다른 모델에 비해  짧아 장기간 의존성을 잘 한다. 시퀀스 변환 문제도 해결할 수 있다.<br>
해석 가능한 모델
: Attention 가중치를 시각화하여 토큰들 간의 대응관계를 눈으로 직접 확인 가능하다. 
다른 모델에 비해 트랜스포머는 모델의 결과를 해석할 수 있다


# Result
![alt](/assets/img/cv5.res.png)



WMT 2014 English-to-German translation task, WMT 2014 English-to-French translation task에서 각각 최고의 성능을 낸다. training cost 또한 가장 낮다. 다른 taks에서도 일반화가 잘 이루어지며 좋은 성능을 보여준다.















https://velog.io/@qtly_u/Attention-is-All-You-Need-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0

https://hyunsooworld.tistory.com/entry/%EC%B5%9C%EB%8C%80%ED%95%9C-%EC%89%BD%EA%B2%8C-%EC%84%A4%EB%AA%85%ED%95%9C-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Attention-Is-All-You-NeedTransformer-%EB%85%BC%EB%AC%B8
