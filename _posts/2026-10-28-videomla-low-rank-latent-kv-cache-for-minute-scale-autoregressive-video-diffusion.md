---
type: literature
title: "VideoMLA: Low-Rank Latent KV Cache for Minute-Scale Autoregressive Video Diffusion"
author: "Hidir Yesiltepe, Jiazhen Hu, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, Hoda Eldardiry, Pinar Yanardag"
year: "2026"
citekey: "yesiltepeVideoMLALowRankLatent2026"
status: unread
tags:
  - literature
source: "http://arxiv.org/abs/2605.30351"
accept:
date: 2026-10-28 11:05:12 +0100
---




> **공식 링크**
>
> - Project page: [VideoMLA](https://videomla.github.io/)
> - Code: [yesiltepe-hidir/VideoMLA](https://github.com/yesiltepe-hidir/VideoMLA)
> - Paper: [arXiv:2605.30351](https://arxiv.org/abs/2605.30351)
> - 공식 저장소는 VideoMLA를 `NeurIPS 2026` 논문으로 표기하고, project page·checkpoint·학습 및 추론 코드를 함께 제공한다.
{: .prompt-info }

> **PDF**
>
{: .prompt-tip }

> [Preprint PDF](zotero://select/library/items/D8T5ZA2U)



> **서지정보**
>
{: .prompt-info }

> Yesiltepe, Hidir, Jiazhen Hu, Tuna Han Salih Meral, 기타. “VideoMLA: Low-Rank Latent KV Cache for Minute-Scale Autoregressive Video Diffusion”. arXiv:2605.30351. Preprint, arXiv, 2026년 5월 28일. [https://doi.org/10.48550/arXiv.2605.30351](https://doi.org/10.48550/arXiv.2605.30351).

  

> **초록(Abstract)**
>
{: .prompt-info }

> Long-rollout causal video diffusion has converged on a fixed-size sliding-window KV cache, with recent progress innovating within this layout by changing which tokens occupy the window or how their positions are encoded. The per-head KV layout itself, a dominant contributor to streaming memory and latency, has been mostly left unchanged. In this paper, we present the first study of Multi-Head Latent Attention (MLA) in video diffusion. VideoMLA replaces per-head keys and values with a shared low-rank content latent and a shared decoupled 3D-RoPE positional key, reducing per-token KV memory by 92.7% at every cached layer. We further investigate why MLA succeeds in video diffusion even though the spectral assumption often used to motivate it in language models does not hold: pretrained video attention is not low-rank, with 99%-energy effective rank far above any practical latent dimension. VideoMLA retains quality at compression ratios where direct spectral approximation would predict large reconstruction error. We show that the MLA bottleneck, rather than the pretrained spectrum, determines the effective rank: both spectral and random initialization occupy nearly the full rank budget from initialization, and training preserves this budget while adapting within it. On VBench, VideoMLA matches short-horizon streaming video diffusion baselines, achieves the best overall score at long horizons among evaluated methods, and improves throughput by 1.23x on a single B200.

  

## 한줄 요약
VideoMLA는 per-head KV를 그대로 저장하는 대신 공유 low-rank content latent와 decoupled 3D-RoPE positional key만 cache에 저장해 KV memory를 92.7% 줄이면서 장기 video quality를 유지한다.

autoregressive 비디오 확산 모델의 효율적인 장기 생성을 위해 기존의 dense KV 캐시 구조를 개선한 VideoMLA를 제안한다.   
multi head latent attention MLA 구조를 도입하여 per token KV 메모리 사용량을 절감하고 추론 처리량을 향상시켰다. 

## 하이라이트 & 내 메모
### 핵심 하이라이트
- Wan-2.1 T2V-1.3B의 self-attention에 MLA를 적용해 모든 head의 K/V를 하나의 shared content latent로 압축한다.
- cache에는 content latent와 positional RoPE key만 저장하고, attention 시 per-head key/value를 필요에 따라 복원하거나 projection을 score/output에 흡수한다.
- dense KV의 3,072 scalars/token을 224 scalars/token으로 줄여 cached KV memory를 92.7% 절감한다.
- Video attention은 일반적인 spectral low-rank 가정과 달리 높은 effective rank를 가지지만, 실제 성능은 pretrained spectrum보다 MLA bottleneck의 rank budget에 의해 결정된다고 분석한다.
- 3D-RoPE positional key와 content latent를 분리하고, long rollout에서는 local window-relative re-indexing으로 학습 시 positional regime를 유지한다.
- VBench 장기 구간에서 경쟁 방법과 비슷한 품질을 유지하면서 B200 한 장에서 throughput 1.23배 향상을 보고한다.

### 내 메모
이 논문의 중요한 점은 “video attention이 원래 low-rank라서 압축된다”가 아니라, dense attention을 그대로 보존하지 않아도 주변 diffusion pipeline이 요구하는 K/V 인터페이스를 bottleneck 안에서 다시 학습할 수 있다는 점이다. 따라서 압축 성능은 SVD reconstruction error만으로 예측할 수 없다. 실제로는 latent dimension, positional channel allocation, 재학습·distillation 과정이 함께 결정한다.

  
![Pasted image 20261028101819](/assets/img/pasted-image-20261028101819.png)

기존 transformer 의 multi-head attention 을 어떻게 multi head latent attention 구조로 바꿔서 KV caches 메모리를줄이는지 보여준다. 

영상의 내용 정보와 위치 정보를 분리한다. 

먼저 transformer에서는각 토큰마다 Query, Key, value를 계산한다.   
autoregressive video generation 에서는 다음 프레임을 생성할 때 이전 프레임들을 참고해야하기에 이전 토큰들의 K,V를 메모리에 저장한다. 이게 KV cache

영상은 이미지보다 토큰이 훨씬 많기 때문에 메모리 사용량이 커진다. 

기존에는 K,V를 모두 저장했는데 VideoMLA에서는 K,V의 내용을 공유하는 작은 latent 하나와 RoPE용 Key만 저장한다.   
이를 통해 토큰당 KV Cache 크기를 92.7% 줄였다. 

제일 밑에 **Video Latent x_t**

이는 원본 픽셀이 아니라 비디오 모델 내부의 latent token.  
KV down: Key와 Value에 필요한 내용을 압축한다.   
Key rotation: 위치 정보용 Key 생성  
Query down: 현재 토큰의 Query 표현 압축

왼쪽 **KV Down -> Key up/value up**  
기존 x_t에서 K와 V를 각각 계산했지만 videoMLA는 먼저 하나의 압축된 표현을 만든다.   
압축 표현을 사용해서 각각 K와 V를 만든다. 

K와 V를 다시 만들 수 있기 때문에 캐시에는 최종 K와 V를 따로 저장할 필요가 없다.   
Key up을 통해 만든 K에는 RoPE가 적용되지 않는다.   
NoPE는 n opositional encoding의 약자이다. 

가운데 Key rotaion -> Decoupled 3D RoPE  
이 부분은 위치 정보를 처리하는 경로이다.   
KV를 압축한 c에는 RoPE 위치 정보를 넣지 않기 때문에 별도로 위치용 Key를 계산한다. 

오른쪽 **Query down -> Query Up/Query Rotation**  
쿼리도 비슷하게 만든다.   
먼저 쿼리를 압축하고 내용용, 위치용 두개로 나뉜다.   
K는 RoPE 부분은 head간에 공유하지만 Q의 RoPE 부분은 head 마다 다르게 생성한다.   
Query는 KV vavhe에 저장하지 안흔다. 현재 attention 계산할 때 사용하기 때문이다. 




> **주장**
>
{: .prompt-info }

## 핵심 주장 / 방법론
1. 장기 AR video diffusion의 memory·latency 병목은 cache token 수뿐 아니라 token당 dense per-head KV layout에도 있다.
2. MLA는 모든 head에 공유되는 content latent와 decoupled positional key를 저장해 cache layout을 저차원화한다.
3. attention 계산은 dense MHA와 같은 NoPE+RoPE score를 재구성하므로 주변 causal mask·sink·window pipeline을 유지할 수 있다.
4. 높은 pretrained effective rank에도 불구하고, 학습된 MLA가 rank budget 안에서 video generation에 필요한 정보를 보존할 수 있다.
5. 3-stage training은 MLA projection 적응, consistency distillation, DMD를 순차적으로 사용해 압축 구조와 few-step long-video generation을 함께 맞춘다.


## 내 생각 / 질문 / 반박
92.7% memory reduction은 매우 크지만, cache만 줄어드는 것과 전체 inference cost가 같은 비율로 줄어드는 것은 다르다. attention 시 K/V reconstruction 또는 absorbed projection, 3D-RoPE 처리, 추가 학습 비용이 필요하기 때문이다. 또한 가장 중요한 rank budget을 layer별·head별로 동일하게 두는 것이 최적인지 의문이다. motion-sensitive layer와 appearance-sensitive layer가 서로 다른 latent capacity를 요구할 수 있다.



## 저자가 정말로 증명한 게 뭐고 그냥 주장만 한 게 뭔가?
실험이 보여주는 것은 특정 Wan backbone과 VideoMLA training pipeline에서 높은 compression ratio에서도 VBench 품질과 long-horizon 성능이 유지되고 throughput·batch headroom이 개선된다는 점이다. “video attention은 low-rank가 아니어도 압축 가능하다”는 결론은 이 architecture·데이터·학습 조건에서의 관찰이며, 모든 video transformer와 모든 cache operation으로 일반화되지는 않는다.


## 이 결과가 놀랍나, 아니면 예상대로인가?
cache를 shared latent로 줄이면 memory가 감소하는 것은 예상 가능하다. 놀라운 부분은 pretrained attention의 effective rank가 높고 직접적인 spectral approximation이라면 큰 오차가 예상되는 상황에서도, bottleneck을 재학습한 모델이 장기 video 품질을 유지했다는 점이다. 이는 기능 보존에 필요한 정보와 원래 parameterization의 rank를 구분하게 한다.



## 저자의 결론이 실험 결과보다 과하게 일반화되진 않았나?
그럴 가능성이 있다. 결과는 Wan-1.3B와 특정 rank·RoPE split·B200 환경에 집중되어 있으며, 더 큰 backbone·다른 video tokenizer·다른 attention kernel에서도 같은 memory/quality trade-off가 유지되는지는 확인이 필요하다. 또한 1.23배 throughput은 전체 시스템의 hardware·kernel 구현에 의존한다.



> **방법론**
>
{: .prompt-info }

## 이 방법이 왜 작동하는가, 그냥 "잘 됐다"가 아니라 원리적으로?
video generation에서 attention이 필요로 하는 것은 각 head의 K/V tensor 자체가 아니라 query가 context를 선택하고 value를 출력으로 전달하는 기능이다. VideoMLA는 content 정보를 공유 latent에 모으고, head별 차이는 up-projection과 query-dependent bilinear form으로 복원한다. 위치 정보는 content와 섞지 않고 별도 3D-RoPE key로 유지해 spatial·temporal indexing을 잃지 않는다. 따라서 low-rank cache는 원래 attention matrix를 근사하는 것이 아니라, 학습 가능한 새로운 internal interface로 attention computation을 재parameterize한다.


## 숨겨진 가정이 뭔가?
모든 head가 공유 content latent에서 복원 가능한 공통 정보를 사용하고, 224 scalars/token이 장기 generation에 필요한 appearance·motion·context 정보를 충분히 담는다고 가정한다. positional key와 content latent의 분리가 안정적이며, local re-indexing이 long rollout에서 RoPE phase drift를 막는다고도 가정한다.





## 실험 설정
Wan-2.1 T2V-1.3B를 dense MHA baseline으로 두고 MLA projection dimension, NoPE/RoPE channel split, SVD·random initialization을 비교한다. 학습은 teacher forcing, consistency distillation, DMD의 3단계이며, VBench short/long horizon, user study, throughput·latency·batch memory를 평가한다. 공식 저장소의 기본 long inference는 sink 1개와 local attention size 6을 사용한다.

- baseline이 공정한가?
quality 비교는 같은 Wan 계열과 streaming setup에서 수행되어 구조 비교에 적절하다. 다만 VideoMLA는 추가적인 3-stage training을 거치므로, dense baseline을 동일한 training budget·distillation 단계로 재학습했는지 확인해야 압축 구조의 순수 효과를 분리할 수 있다. throughput은 kernel fusion과 B200 memory system의 영향을 받으므로 다른 hardware에서 재현되어야 한다.



> **한계**
>
{: .prompt-info }

## 직접 명시한 limitation 
Video attention의 high effective rank 때문에 단순 spectral compression만으로는 설명되지 않으며, latent bottleneck과 RoPE split을 잘 선택해야 한다. 작은 latent dimension에서는 quality가 크게 떨어지고, long-horizon 성능은 positional encoding과 cache/window 설정에 의존한다.

## 저자가 명시하지 않은, 추정되는 한계 
공유 latent가 head별 특수성을 제한할 수 있고, layer·head별로 서로 다른 motion·appearance 정보를 같은 budget에 넣으면 일부 조건에서 품질이 급락할 수 있다. cache memory는 줄어도 projection과 attention kernel의 계산량이 늘 수 있으며, 압축 latent에 대한 debugging·interpretability도 dense K/V보다 어렵다.


## 이 방법이 실패하는 경우를 상상할 수 있나?
장면 전환이 급격하거나 드문 motion·세밀한 texture를 오래 유지해야 할 때 작은 shared latent가 필요한 정보를 잃을 수 있다. RoPE re-indexing이 실제 camera·time 관계를 과도하게 단순화하거나, 다른 backbone의 attention head들이 공유 latent로 잘 표현되지 않으면 quality·long-term consistency가 함께 무너질 수 있다.
