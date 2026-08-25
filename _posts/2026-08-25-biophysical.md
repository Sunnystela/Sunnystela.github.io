---
layout: post
title:  "[논문정리] Learning biophysical determinants of cell fate with deep neural networks"
date:   2026-08-25 10:17
categories: AI Bio
tag: CV Bio
math: true
---



> 공식 링크 / 확인
> - Nature article: [Learning biophysical determinants of cell fate with deep neural networks](https://www.nature.com/articles/s42256-022-00503-6)
> - DOI: [10.1038/s42256-022-00503-6](https://doi.org/10.1038/s42256-022-00503-6)
> - bioRxiv preprint: [Learning the Rules of Cell Competition Without Prior Scientific Knowledge](https://www.biorxiv.org/content/10.1101/2021.11.24.469554v1)
> - Code: [cellx-predict](https://github.com/lowe-lab-ucl/cellx-predict)
> - Data: [cellX-predict datasets](https://doi.org/10.5522/04/16578959)
> - Software repository: [cellX-predict software](https://doi.org/10.5522/04/19207923)
> - title/author 확인: Nature 최종 논문의 title과 author가 현재 노트의 title, author와 일치한다.
> - venue 확인: Nature 공식 페이지 기준 `Nature Machine Intelligence`, volume 4, pages 636-644, accepted 2022-05-13, published 2022-06-30.
{: .prompt-tip }

> PDF
> [PDF](zotero://select/library/items/4YERMCF7)
{: .prompt-info }



> 서지정보
> Soelistyo, Christopher J., Giulia Vallardi, Guillaume Charras와/과Alan R. Lowe. “Learning Biophysical Determinants of Cell Fate with Deep Neural Networks”. _Nature Machine Intelligence_ 4, 호 7 (2022): 636–44. [https://doi.org/10.1038/s42256-022-00503-6](https://doi.org/10.1038/s42256-022-00503-6).
{: .prompt-tip }

  
  

## 한줄 요약

MDCK cell competition time-lapse에서 β-VAE가 local tissue image를 해석 가능한 latent/PCA 표현으로 만들고, TCN이 cell fate를 예측하며, 모델 내부 표현과 prediction mismatch를 통해 cell density/crowding이 fate 결정의 핵심 biophysical determinant임을 복원한 논문이다.

세포가 과거 현미경 영상만 보고 미래 fate룰 예측하는 데서 끝나지 않고 모델을 해석해서 어떤 물리적 요인이 fate를 결정하는지 찾아내는 논문. 이 논문에서는 그 결과 local cell density 가 mechanical cell competition에서 fate를 예측하는 가장 중요한 요인이라는 것을 데이터로부터 자동으로 발견한다

MDCK 세포들이 경쟁하는 time-laps 영상을 찍고 한 세포가 분열할지 사멸할지 예측한다. 초기 상태에서 미래 fate를 예측한다





## 하이라이트 & 내 메모


![alt](/assets/img/biophysic-fig5.png)

세포 경쟁 관찰 → 미래 fate 예측 문제 정의 → τ-VAE로 시간 정보를 예측 → β-VAE로 이미지 표현 학습 → PCA로 해석 가능한 feature 생성

### a. Cell competition에서 무엇을 예측

- 초록색: **MDCKWT**
- 보라라색: **MDCK scribkd**

여기서 특정 세포 하나를 중심으로 봅니다. 이 세포의 미래 fate는 크게

- **Mitosis**: 세포 분열
- **Apoptosis**: 세포 사멸


현재 세포 하나의 모습뿐 아니라, 주변 세포들과의 local tissue organization이 시간에 따라 어떻게 변했는지가 미래 fate를 결정할 수 있다.



### b. 미래를 보기 전에 예측한다

세포 하나를 추적하면서 여러 시간대의 이미지를 얻습니다.

`과거 → 과거 → 현재 → ... → 실제 fate`

그런데 점선으로 **Cutoff**가 있습니다
mitosis/apoptosis가 눈으로도 구별되기 시작합니다.

아직 fate가 형태적으로 드러나기 전에 이 세포가 나중에 죽을지 분열할지 예측



### c. 전체 τ-VAE prediction pipeline


**Timelapse → Probabilistic encoder → Temporal model → Fate prediction**


먼저 각 시점 이미지가 들어옵니다.
시점의 이미지가 각각 encoder를 통과합니다.

각 이미지를 그대로 temporal model에 넣는 것이 아니라,
이미지를 **low-dimensional representation**으로 바꿉니다.

그림 가운데의 색 점들이 각 시점의 latent representation이라고 보면 됩니다.
**latent trajectory**로 바뀝니다.

이 temporal sequence를 **TCN(Temporal Convolutional Network)** 이 봅니다. 논문에서는 약 **8.5시간에 해당하는 128 time steps**를 receptive field로 사용합니다.

마지막 prediction head가

- apoptosis
- mitosis
- other

중 하나를 출력합니다.

그림에서는 예측 결과가 **Apoptosis**로 .
그리고 오른쪽 Ground truth와 비교해서 loss를 계산하며 학습합니다.


**세포와 주변 환경이 시간에 따라 어떻게 변했는지를 latent trajectory로 만든 뒤, TCN으로 미래 fate를 예측한다.**


### d. β-VAE로 이미지 자체를 학습한다

이 부분은 fate prediction이 아니라 **이미지 representation learning 단계**입니다.

먼저 실제 이미지 x가 들어갑니다.

β-VAE이기 때문에 encoder가 단순히 하나의 값 z를 내는 게 아니라

$q_\phi(z|x)$

라는 **latent probability distribution**을 만듭니다.

그리고 그 분포에서 z를 sample합니다.

논문에서는 latent dimension이 **32차원**입니다.

즉 이미지 한 장이

 $x \rightarrow z \in \mathbb{R}^{32}$

로 바뀝니다.

이 z를 decoder에 넣으면 다시 이미지 x'를 복원합니다.

`x → Encoder → z → Decoder → x'`

왼쪽의

**Real image**와 **Synthesized image**

가 비슷하도록 reconstruction loss를 줍니다.



그림의 **KL loss**도 중요합니다.

β-VAE는 latent distribution이 기본 Gaussian prior

$p(z)=N(0,I)$

와 크게 벗어나지 않도록 KL divergence를 사용합니다.

그래서 latent space가 완전히 뒤죽박죽되는 대신 **연속적이고 구조화된 representation**을 갖도록 유도합니다.



### e. Latent를 PCA로 다시 해석 가능한 feature로 만든다

여기서 d에서 얻은 32차원 latent representation을 그대로 TCN에 넣지 않습니다.

저자들은 약 **120만 장의 cell image**를 β-VAE로 encoding합니다.

그래서 아주 많은

$z_1,z_2,z_3,\dots$ 

가 latent space에 생깁니다.

그림의 회색 점들이 그것입니다.

이 전체 latent dataset에 **PCA**를 적용합니다.

그래서 새로운 축

- PC0
- PC1
- PC2
- ...
- PC31

을 만듭니다.


$z\in R^{32}$ 를 $PC0,PC1,\dots,PC31$ 로 projection합니다.

중요한 것은 **차원 수를 줄이려고 PCA를 쓴 것이 핵심이 아니라는 점**입니다.

32 latent → 32 PC입니다.

PCA의 주된 목적은 **β-VAE latent를 사람이 해석하기 쉬운 방향으로 회전시키는 것**







>**① a — 문제 정의**  
`주변 세포와 함께 살아가는 세포`  
↓  
미래에
`mitosis / apoptosis`
중 무엇이 되는가?  
>**② b — 미래 정보 제거**
`과거 interphase 영상`
→ **cutoff**
→ 미래 fate는 모델에게 보여주지 않음
>**③ d — 각 frame을 β-VAE로 encoding**
$Image_t \rightarrow z_t$   
>**④ e — latent를 PCA feature로 변환**
$z_t \rightarrow [PC0_t,PC1_t,\cdots,PC31_t]$   
> **⑤ c — 시간 sequence를 TCN에 넣음**
$PC_{t-128:t} \rightarrow TCN \rightarrow Apoptosis/Mitosis/Other$
{: .prompt-info }







![alt](/assets/img/biophysic-fig4.png)

> **① 모델이 학습한 PC들이 실제로 어떤 생물물리적 특성을 의미하는가?**  
> **② 그중 어떤 특성이 언제 cell fate 예측에 중요한가?**



### a. 각 PC가 실제로 무엇을 의미하는가?

$PC0, PC1, PC2,\dots$

직접 측정할 수 있는 여러 physical parameter와 각 PC의 **Pearson correlation**을 계산합니다.

- Solidity
- Orientation
- Brightness
- Eccentricity
- Aspect ratio
- Cell density
- Cell size
- Cell type

입니다.

결과적으로 강하게 연결되는 것들을 보면

$\boxed{PC0 \approx Cell\ type}$
$\boxed{PC1 \approx Cell\ density}$
$PC2 \approx Nuclear\ orientation$
$PC3 \approx Nuclear\ aspect\ ratio$



β-VAE가 이미지만 보고 unsupervised하게 latent를 학습시켰는데, PCA를 해보니 **실제 물리적 특성과 대응되는 축들이 자연스럽게 나타난 것**입니다.

**AI가 학습한 representation이 실제 biological/physical property와 연결된다.**



### b. PC0와 PC1을 실제 이미지로 확인

a에서 파악한 것을 
b는 이것을 **시각적으로 확인**하는 부분입니다.

가로축으로 PC0를 바꾸고,

$PC0:-3\rightarrow+3$

세로축으로 PC1을 바꿉니다.

$PC1:-3\rightarrow+3$

그러면 reconstructed image가 실제로 변합니다.

**latent space에서 해당 PC 값을 실제로 움직여 보면서 이미지가 어떻게 변하는지도 확인**



### c. PC0~PC3의 의미를 더 직접적으로 검증

여기가 a를 훨씬 직관적으로 보여줍니다.

위에 있는 작은 이미지들을 먼저 보면 됩니다.

PC 값을

$-5 \rightarrow 0 \rightarrow +5$

정도로 변화시키면서 해당 PC를 가진 이미지들을 확인합니다.


**PC0 → Cell type**

PC0가 변하면 이미지가
WT ↔ scribkd 방향으로 바뀝니다.

그래서 아래 그래프에서도 PC0 magnitude에 따라 scribkd 비율이 크게 변합니다.
관련 있다고 볼 수 있다

**PC1 → Cell density**
이게 **이 논문에서 가장 중요한 PC**입니다.

PC1이 증가할수록 주변에 보이는 nucleus 수가 증가합니다.

세포들이 더 빽빽하게 있습니다.
관련 있다고 볼 수 있다

논문에서는 local density와 nuclear area가 관련된 정보가 이 component에 반영되어 있다고 설명합니다.


**PC2 → Orientation**
PC2가 변하면 중앙 nucleus의 방향이 달라진다.


**PC3 → Aspect ratio**
PC3가 변하면 nucleus가

둥근 형태 ↔ 길쭉한 형태

로 변합니다.

이것도 관련있다는 것을 볼 수 있다

사람이 해석할 수 있는 representation이 만들어졌다



### d. 어떤 PC가 fate prediction에 가장 중요한가?


**Feature ablation**을 합니다.

특정 PC를 **Gaussian noise로 교체**합니다.

그리고 prediction accuracy가 얼마나 떨어지는지 봅니다.


처음에는 모든 PC가 있기 때문에 accuracy가 높습니다.

그런데 PC를 하나씩 제거합니다.
중요하지 않은 PC부터 제거하면 accuracy가 별로 떨어지지 않습니다.
반대로 **중요한 PC를 제거하면 accuracy가 크게 떨어집니다.**



PC1 하나만 남겨도 약

$Accuracy\approx43\%$

3-class random chance는 33% 이므로 PC1 하나만으로도 fate에 대한 상당한 정보가 있다는 겁니다.

처음부터 density를 feature로 넣은 것이 아니라 모델이 이미지에서 스스로 학습했다

![alt](/assets/img/biophysic-fig3.png)

### e. Apoptosis가 일어나는 세포에서는 시간에 따라 무엇을 보는가?

d에서는 **WHAT:** 어떤 feature가 중요한가? 를 봤다면,

e/f에서는 **WHEN:** 그 feature가 언제 중요한가? 를 봅니다.

e는 **실제로 나중에 apoptosis가 발생한 하나의 trajectory 예시**입니다.



e 첫 번째 줄: 실제 영상
0h가 실제 apoptosis event에 가까운 시점입니다.


e 두 번째 줄: PC feature saliency

각 시간 × 각 PC에 대해
이 PC가 이 시점의 prediction에 얼마나 영향을 줬는가?
를 gradient 기반 saliency로 계산합니다.

밝은 부분일수록 해당 시간/PC가 prediction에 중요했다는 의미입니다.

단순히 PC 값이 크다는 뜻과 **saliency가 높다는 것은 다릅니다.**


e 세 번째 줄: PC value
중요한 PC들의 실제 값이 시간에 따라 어떻게 변했는지를 보여줍니다.


e 마지막 줄: Fate prediction probability

- Mitosis
- Apoptosis
- Other

중 무엇이라고 예측하는지 보여줍니다.
scribkd cell의 경우 **최대 약 8시간 전부터 apoptosis prediction이 나타날 수 있었습니다.**



![alt](/assets/img/biophysic-fig2.png)

### f. Mitosis trajectory

f는 e와 구조가 완전히 같습니다.
차이는 이 세포의 실제 fate가 **Mitosis**라는 것입니다.

전체 데이터 분석에서는 **mitosis prediction은 apoptosis보다 훨씬 늦게**, 대략 event **2시간 전** 정도에 나타나는 경향이 있었습니다.

그래서

$Apoptosis:\sim8h\ before$

vs.

$Mitosis:\sim2h\ before$

라는 시간적 차이가 있습니다.


![alt](/assets/img/biophsic-fig1.png)

### g. 이번에는 PC가 아니라 원본 이미지의 어디를 봤는가?

지금까지는
어떤 **PC**가 중요한가를 봤다.

그런데 원래 입력은 이미지.
그래서 gradient를 encoder까지 다시 backpropagation해서
**원본 이미지의 어떤 pixel이 prediction에 영향을 주었는가?**
를 계산한다.



첫 번째 줄: **Raw image**

두 번째: **GFP channel saliency** : 초록색 세포: **원본의 초록색 정보 중 어디가 중요했나?**

세 번째: **RFP channel saliency** : 보라색 세포: **원본의 자홍색 정보 중 어디가 중요했나?**



- **nearby cells**
- central cell의 **nuclear geometry**
    - aspect ratio
    - convexity 등

이 중요한 영역으로 나타난다.
다만 저자들도 pixel saliency는 PC space만큼 **정량적으로 해석하기 어렵다**고 명시합니다.



> GFP, RFP 동시에 보여주면 안되는지
>**각 채널의 어느 부분이 예측에 민감했는지를 보기 위해 분리해서 보여주는 것**입니다
{: .prompt-info }








> 주장
{: .prompt-warning }

## 핵심 주장 / 방법론

핵심 주장은 deep learning을 단순 segmentation/classification 도구로 쓰는 데서 멈추지 않고, time-lapse microscopy로부터 생물학적 mechanism을 학습하고 해석할 수 있다는 것이다. 저자들은 cell competition에서 loser cell의 fate가 시간에 따른 local cellular neighbourhood와 morphology에 의해 결정된다고 보고, 이를 end-to-end로 학습하는 `tau-VAE`를 만든다.


1. MDCK WT와 scrib-kd cell competition 영상을 4분 간격으로 촬영하고, single-cell trajectory와 fate(apoptosis/mitosis)를 만든다.
2. 각 timepoint에서 중심 cell 주변의 glimpse를 추출한다.
3. β-VAE가 image glimpse를 low-dimensional latent로 압축하고 복원한다.
4. latent를 PCA로 투영해 PC0, PC1 같은 해석 가능한 축을 만든다.
5. PC time series를 TCN에 넣어 fate를 예측한다.
6. latent PC ablation, saliency, biochemical perturbation, discriminator network로 모델이 무엇을 배웠는지 해석한다.

모델이 expert feature engineering 없이 local cell density/crowding 관련 표현(주로 PC1)을 fate prediction의 핵심 변수로 학습했다






