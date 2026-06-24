---
layout: post
title:  "[다시보기] Attention is All You Need"
date:   2026-06-24 17:17
categories: AI CV 
tag: CV
math: true
---

기존 시퀀스 변환 모델들은 주로 복잡한 RNN, CNN 기반으로 인코더 디코더 구조를 사용했다. 이러한 모델들은 입력과 출력 시퀀스의 각 위치를 순차적으로 처리하기 때문에 병렬화가 어렵고 학습시간이 오래 걸리는 단점이 있었다. 




attention은 기존 인코더 디코더의 성능을 강화시키며 주목받고 있던 메커니즘이다. 이 논문에서 attention을 발표한 것이 아닌 RNN을 사용하지 않고 attention만으로도 입력 데이터에서 중요한 정보를 찾아 단어를 인코딩 할 수 있다는 것을 발표한 것이다. 



# Transformer 

RNN은 LSTM고 기존 자연어 처리 task에서 가장 많이 쓰이는 모델이다. 자연어 처리에서 문장을 처리할 때 문장에 사용된 단어의 위치가 중요하다. 단어의 위치오 순서 정보를 잘 홀용하는  RNN과 LSTM이 자주 활용되었다. 

그러나 RNN은 문장 길이가 길어지면 학습 능력이 떨어지게 된다. 


트랜스포머는 순환과 convolution 을 완전히 배제하고 오직 어텐션 메커니즘만을 사용하여 모델을 설계했다. 이러한 구조적 변화로 여러 장점을 가져왔다. 

Transformer은 이러한 과정을 병렬적으로 처리하기 때문에 성능과 속도면에서 훨씬 뛰어나다. 

번역과 같은 시퀀스 변환 작업에서 기존 최고 성능을 뛰어넘는 결과를 달성했다. 영어 구문 분석 같은 다른 작업에도 성공적으로 적용되었다. 


단어 위치 정보가 중요한데 RNN을 사용하지 않는다면 어떻게 단어 위치정보를 활용할까?

Positional Encoding을 활용할 수 있다. 


이 논문에서 순환없이 입력값과 출력값 간 의존성을 모델링할 수 있는 attention mechanism 만을 사용한 모델 구조인 transfer을 제안한다. 


# Background
 
연속적인 계산을 줄이기 위해 CNN을 기본 구성 요소로 사용한 연구가 이루어진다. 

이는 거리에 따라 계산량이 증가하고 입력값과 출력 값의 거리가 멀수록 의존성을 알기 어려워진다. 

반면 Trnasformer에서는 Multi-head attnetion을 통해 상수 시간의 계산만으로 가능하다. 

> self-attention
자신에게 수행하는 어텐션 기법으로 단일 시퀀스에서 서로 다른 위치에 있는 요소들의 의존성을 찾아낸다. 
{: .prompt-info }

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

인코더:
6개의 동일한 레이어로 구성된다.
각 레이어는
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network

로 이루어지며
각 서브 레이어 뒤에는 Residual Connection과 Layer Normalization이 적용된다.

디코더:
6개의 동일한 레이어로 구성된다.
각 레이어는

1. Masked Multi-Head Self-Attention
2. Encoder-Decoder Multi-Head Attention
3. Position-wise Feed-Forward Network

로 이루어지며
각 서브 레이어 뒤에는 Residual Connection과 Layer Normalization이 적용된다.



### Attention
#### Scaled Dot-Product Attention
![alt](/assets/img/cv5.atten.png)


Q: 영향을 받는 벡터<br>
K: 영향을 주는 벡터<br>
V: 주는 영향의 가중치 벡터


입력 X로부터

$Q=XW_Q$
	​

$K=XW_K$

$V=XW_V$


를 만든다.

즉 같은 입력으로부터 서로 다른 가중치 행렬을 통해 생성된다.

> Q와 K  
→ 누구를 얼마나 볼지 결정  
→ Softmax  
→ 중요도 생성
→ V  
→ 실제 정보 가져오기
{: .prompt-warning }


![alt](/assets/img/cv5.out.png)



output: value의 가중합으로 계산한다<br>
가중합에 이용되는 가중치: query와 연관된 key의 호환성 함수에 의해 계산된다


> 왜 √dk로 나누는가?
차원이 커질수록 $QK^T$ 값이 매우 커진다
Softmax에 큰 값이 들어가면 0 또는 1에 가까운 값만 나온다. Gradient가 작아져 학습이 어려워지기 때문에 나눠서 분산을 조절한다. 
{: .prompt-warning }

#### Multi-Head Attention

![alt](/assets/img/cv5.multi.png)

모델이 다양한 관점에서 문장을 해석할 수 있도록 해준다. 각각 head 가 보는 관계가 다르기에 여러 종류의 관계를 동시에 학습한다. 

> 모델이 Head 개수만큼의 Scaled dot product Attention 연산을 수행할 수 있게 하여 모델이 다양한 관점의 Attention Map을 만들게한다.

여러 개의 어텐션 함수를 병렬로 실행하여 다양한 관점에서 정보를 얻는다. 

쿼리, 키, 값을 여러 개의 저차원 공간으로 선형 투영한 후 각 투영된 벡터에 대해 어텐션 함수를 병렬로 수행한다.  
각 헤드에서 얻은 어텐션 결과를 concatenate 하고 다시 선형으로 투영하여 최종 출력을 얻는다.   
이를 통해 서로 다른 subspace의 정보에 동시에 집중 할 수 있다. 

$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$   
$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$ 


트랜스포머는 멀티-헤드 어텐션을 세 가지 방식으로 활용한다

Encoder-decoder attention :
이전 decoder 레이어에서 오는 query들과 encoder의 출력으로 나오는 memory key, value들과의 attention  
decoder의 모든 위치에서 input sequence의 모든 위치를 참조할 수 있게 해준다  

Self-attention in encoder :
Encoder의 각 위치들은 이전 레이어의 모든 위치들을 참조할 수 있게 해준다.  

Self-attention in decoder :
decoder의 각 위치들은 decoder 내의 다른 위치들을 참조할 수 있는데, 이전부터 자신 위치까지만을 참조할 수 있다.  
마스킹 기법을 사용하여 미래 정보를 참조하는 것을 방지하므로써 
auto-regressive 성질을 살리면서도 정보가 잘못 흐르는 것을 막는다.



## Position-wise Feed-Forward Networks


![alt](/assets/img/cv5.forward.png)

ReU활성화가 있는 두 개의 선형 변환으로 구성된다.

각 위치마다 독립적으로 적용되는 완전 연결 신경망이다. 

> 입력벡터 x를 더 큰 차원으로 변환한다. ReLU 활성화 함수를 적용한다. 두번째 선형 변환으로 다시 원래 차원으로 줄인다. Transformer의 Attention 뒤에서 각 토큰의 표현력을 높이는 역할을 한다.
{: .prompt-warning }

## Embeddings and Softmax

input과 output 토큰을 embedding layer를 거쳐서 사용한다. 
<br>
생성된 embedded vector는 컴퓨터가 정보를 쉽게 수집할 수 있도록 만든 의미적인 태그들로 구성된 특성을 나타내게 된다.<br>



## Positional Encoding
토큰의 상대적인 위치에 대한 정보를 제공하기 위한 역할이다. 



Transformer는 Recurrent model을 사용하지 않고 오직 Attention mechanism만을 사용하여 만들기 때문에 Sequence 정보를 담아낼 수가 없다.


sequence내 토큰의 순서 정보를 모델에 추가해주는 것이 positional encoding이다.

포지셔널 인코딩은 입력 임베딩에 더해져 모델 하단에 주입된다. 

sine, cosine 함수를 사용하여 각 위치와 차원에 따라 고유한 인코딩 값을 생성한다. 

$PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}})$   
$PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d_{model}})$ 

이 방식은 모델이 상대적인 위치를 학습하는데 유리하다. 학습 시 보지 못했던 더 긴 시퀀스 길이에 대해서도 추론이 가능할 것이다. 

> 왜 Sin/Cos를 사용했을까?
위치마다 고유 벡터 생성 가능하다.  
상대 위치 계산 가능하다.  
학습되지 않아도 일반화 가능하다. 
{: .prompt-warning }


# why self-attention

하나의 레이어 당 전체 연산 복잡도 감소
: RNN 계열에서는 하지 못했던 병렬 처리 연산의 양을 대폭 늘려 자연스레 학습 시간 감소한다.<br>
Long term dependency의 문제점 해결
: 트랜스포머 모델은 대응관계가 있는 토큰들 간의 물리적인 거리값들 중 최댓값이 다른 모델에 비해  짧아 장기간 의존성을 잘 한다. 시퀀스 변환 문제도 해결할 수 있다.<br>
해석 가능한 모델
: Attention 가중치를 시각화하여 토큰들 간의 대응관계를 눈으로 직접 확인 가능하다. 
다른 모델에 비해 트랜스포머는 모델의 결과를 해석할 수 있다

