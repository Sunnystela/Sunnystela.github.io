---
type: post
title: "[논문정리] LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing"
tags: AI CV
date: 2026-10-14 16:03
math: true
categories: AI CV
---



> **공식 링크 / 확인**
>
> - Project page: [LiveEdit official project page](https://live-edit.github.io/)
> - Paper: [arXiv:2606.26740](https://arxiv.org/abs/2606.26740)
> - Proceedings: [Springer ECCV 2026 chapter](https://link.springer.com/chapter/10.1007/978-3-032-37574-2_35)
> - Code: [GitHub cp-cp/LiveEdit](https://github.com/cp-cp/LiveEdit)
> - title/author 확인: arXiv, official project page, Springer chapter 모두 `LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing`와 `Xinyu Wang, Chongbo Zhao, Fangneng Zhan, Yue Ma`로 일치.
> - accept 확인: Springer chapter 기준 `Computer Vision - ECCV 2026`, conference paper, first online 2026-09-13, pp. 631-649. GitHub README도 `Accepted by ECCV 2026`로 표기.
{: .prompt-info }

> **PDF**
>
{: .prompt-tip }

> [Full Text PDF](zotero://select/library/items/NEKHYVVG)



> **서지정보**
>
{: .prompt-info }

> Wang, Xinyu, Chongbo Zhao, Fangneng Zhan와/과Yue Ma. “LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing”. arXiv:2606.26740. Preprint, arXiv, 2026년 7월 13일. [https://doi.org/10.48550/arXiv.2606.26740](https://doi.org/10.48550/arXiv.2606.26740).

  

> **초록(Abstract)**
>
{: .prompt-info }

> Streaming video editing has made rapid progress, yet practical deployment is still limited by two core issues: maintaining stable backgrounds and non-edited regions over time, and achieving the low latency required for real-time interactive scenarios. Meanwhile, recent streaming video generation methods are mostly developed for synthesis and cannot be directly applied to editing due to the strict preservation requirement and region-specific control. In this work, we present a novel streaming video editing framework that performs causal, frame-by-frame editing with strong content preservation and real-time responsiveness. Our key design is a three-stage distillation pipeline that progressively transfers editing capability from a powerful bidirectional foundation model to an efficient unidirectional streaming editor, enabling stable long-horizon edits without sacrificing visual fidelity. To further support real-time deployment, we introduce an AR-oriented mask cache that reuses region-related computation across frames, substantially reducing redundant processing and accelerating inference. Finally, we establish a dedicated benchmark for streaming video editing. Extensive evaluations demonstrate that our method achieves state-of-the-art visual quality among streaming baselines while drastically boosting inference speed to 12.66 FPS, making it suitable for interactive and augmented reality applications.

  

## 한줄 요약

LiveEdit는 bidirectional video editing model의 품질을 유지하면서도 real-time streaming editing을 가능하게 하기 위해 `Foundation Tuning -> Causal Adaptation/Teacher Forcing -> DMD 4-step Distillation -> AR-oriented Mask Cache`로 이어지는 단계적 distillation pipeline을 제안한 ECCV 2026 논문이다.

  

  
  

## 하이라이트 & 내 메모


![Pasted image 20261014135035](/assets/img/pasted-image-20261014135035.png)
  
기존 diffusion 기반 영상 편집 모델을 어떻게 실시간 스트리밍 영상 편집 모델로 만드는지 보여준다. 

1. 편집 능력 학습: 원본 영상과 텍스트 지시를 받아 원하는 부분을 수정하도록 모델을 학습한다. 
2. 실시간 처리에 맞게 구조 변경: 미래 프레임을 보지 않고 이전 프레임의 정보만 활용하여 순차적으로 편집할 수 있도록 학습한다. 
3. 빠른 영상 생성: diffusion 의 생성 과정을 가속해 실시간 편집이 가능하도록 만들고 AR oriented mask cache로 프레임간 일관성을 유지한다. 


**1.편집능력 학습**  
우선 영상 편집을 잘하는 diffusion 모델을 만든다.   
실시간 처리 속도를 고려하기 보다는 영상 편집 정확성과 품질을 학습한다. 

full attention (bidirectional attention)을 한다. 시간적으로 일관된 편집이 가능하지만 미래 프레임이 필요하기 때문에 실시간 스트리밍에 직접 사용하기에는 어렵다. 

**2.실시간 처리에 맞게 구조 변경**  
stage1에서 학습한 편집 능력을 유지하면서 미래 프레임 없이 동작하도록 바꾼다.   
streaming video editing을 가능하게 만드는 핵심적인 구조 변경이다. 

영상을 작은 chunk로 나누어 처리한다. 논문에서는 chunk당 3개 latent frame을 사용한다. stage1 모델에 casual attention 을 갑자기 적용하면 기존에 학습한 attention 분포가 달라져 편집 품질이 떨어질 수 있다. 

이를 해결하기 위해 1의 가중치를 바탕으로 casual DiT를 초기화하고 teacher forcing으로 다시 학습한다. 

teacher forcing은 학습 중 정답을 조건으로 사용하는 방식이다. 논문에 명시된 학습 손실은 causal mask를 적용한 노이즈 예측 MSE이고 별도 attention map 모방 손실이 제시된 것은 아니다. 

미래 프레임 없이 영상 편집이 가능해지고 이전 chuck의 정보를 활용한 순차적 처리, 학습한 편집 품질 유지된다. streaming 가능하지만 빠르지는 않다. 

**3.DMD**  
diffusion sampling을 100단계에서 4단계로 줄인다.   
여기서 Distribution matching distillation을 사용한다.   
figure에 등장하는 세 모델의 역할을 구분하면

generator는 실제 편집 영상을 생성하는 모델  
real score는 실제 데이터 분포에 대한 기준 제공  
fake score는 generator가 만든 데이터 분포 추정이다. 

generator가 noise와 원본 영상 조건으로 편집된 영상을 생성한다.   
생성 결과가 다시 노이즈를 추가한다.   
real score와 fake score가 각각 노이즈가 추가된 결과를 평가한다.   
두 score차이로 DMD gradient를 계산하여 generator를 업데이트한다.   
동시에 MSE loss를 이용해 편집 결과의 안정성을 유지한다. 

DMD는 적은 생성 단계만 사용하더라도 generator의 출력 분포가 실제 데이터 분포와 가까워지게 만들고자한다. 

**4.ar oriented mask cache**  
앞의 세 학습 단계와 별도로 실제 추론에서 계산량을 더 줄이는 방법이다.   
초록색 스웨터를 노란색으로 바꾸는 경우 얼굴이나 배경은 대부분 그대로 유지되어야한다.   
그런데 모든 프레임의 모든 위치에 완전한 attention 계산을 반복하면 계산 낭비가 생긴다.   
그래서 이전 chunk의 원본과 편집 결과를 비교하여 변경이 발생한 영역을 식별한다.   
모든 연산을 생략하는 것이 아닌 self attention의 중간 특징만 선택적으로 재사용하낟. 





> **주장**
>
{: .prompt-info }

## 핵심 주장 / 방법론

- 핵심 문제의식: 기존 streaming video generation은 "새 영상을 생성"하는 데 초점이 있어, video editing에서 필요한 **원본 배경/비편집 영역 보존**을 직접 만족하지 못한다. 반대로 bidirectional/offline editing model은 품질은 좋지만 전체 sequence를 보고 처리해야 해서 real-time interactive setting에 맞지 않는다.
- LiveEdit의 핵심 주장은 **강한 bidirectional editing teacher를 바로 causal streaming model로 바꾸지 말고, 세 단계로 점진적으로 옮겨야 한다**는 것이다.
- Stage 1, Foundation Tuning: source video latent와 noisy latent를 channel concatenation으로 넣고, bidirectional DiT를 full temporal attention으로 fine-tuning해 offline editing prior를 학습한다.
- Stage 2, Teacher Forcing for Chunk-wise Causal Initial: causal attention mask를 도입해 미래 frame을 보지 않는 chunk-wise streaming 구조로 바꾸되, Stage 1 bidirectional model의 출력을 teacher로 삼아 causal model이 품질을 유지하도록 맞춘다.
- Stage 3, DMD for Streaming Video Editing: Stage 2 causal model을 초기값으로 쓰고, Distribution Matching Distillation을 적용해 4-step generator로 압축한다. 목적은 CFG와 많은 denoising step 없이도 interactive speed를 얻는 것이다.
- AR-oriented Mask Cache: 이전 chunk의 edited output과 source latent 차이에서 editing mask를 만들고, 바뀌지 않는 background token은 self-attention 계산을 재사용한다. 즉, mask는 편집 영역 제어뿐 아니라 **어디를 계산하고 어디를 cache할지 정하는 연산 예산 배분 장치**가 된다.

### 주요 figure 흐름

- **Figure 1 / teaser**: LiveEdit가 local edit와 global edit를 causal streaming 방식으로 처리하면서도, 기존 streaming/video editing baseline보다 낮은 latency와 높은 시각 품질을 목표로 한다는 문제 설정을 보여준다. 여기서 봐야 할 포인트는 "실시간"과 "비편집 영역 보존"이 동시에 요구된다는 점이다.
- **Figure 2 / paradigm comparison**: bidirectional editing은 미래 frame까지 모두 보므로 품질은 좋지만 느리고, 기존 streaming generation은 빠르지만 원본 보존이 약하다. LiveEdit는 Causal DiT와 mask-guided cache를 결합해 두 요구 사이의 절충점을 만든다.
- **Figure 3 / framework**: 논문의 핵심 구조다. Stage 1은 bidirectional DiT를 편집 teacher로 만들고, Stage 2는 causal attention으로 streaming 구조를 맞추며, Stage 3은 DMD로 4-step generator를 만든다. AR-oriented Mask Cache는 chunk 사이에서 편집 영역과 background 영역을 나눠 background token을 재사용한다.
- **Figure 4 / mask cache ablation**: cache를 어디에 적용해야 하는지 보여준다. self-attention에 cache를 걸면 background consistency를 유지하면서 속도를 얻지만, FFN에 cache를 걸면 품질이 깨질 수 있다. 즉, 모든 module의 feature가 같은 방식으로 재사용 가능한 것은 아니다.
- **Comparison figures**: InsV2V, LucyEdit, VideoCoF, StreamDiffusion, StreamV2V와 비교하면서 LiveEdit가 background/non-edited region preservation과 speed를 동시에 노린다는 점을 보여준다. 단순히 예쁜 output보다 "편집하지 말아야 할 영역을 유지하는가"가 핵심 평가축이다.




## 내 생각 / 질문 / 반박

- END_Personalization 관점에서 이 논문은 "identity/reference를 어떻게 넣을 것인가"보다 **streaming AR setting에서 무엇을 보존해야 하는가**를 먼저 정의한다는 점이 중요하다. Personalized AR video에서도 사용자의 얼굴/옷/배경 중 바뀌면 안 되는 영역이 있고, 이 영역은 품질 제약이면서 동시에 cache 가능한 계산 단위가 될 수 있다.
- 가장 흥미로운 부분은 Mask Cache다. 보통 mask는 editing control로만 생각하기 쉬운데, LiveEdit에서는 mask가 "이 영역은 다시 계산해야 하고, 이 영역은 이전 chunk feature를 재사용해도 된다"는 computation routing 역할도 한다.
- 반박/주의점: 12.66 FPS는 논문 setting과 hardware, resolution, chunk size, Wan2.1 기반 모델 조건에 묶여 있다. 모든 interactive video editing setting에서 real-time이라고 일반화하면 안 된다.
- 질문: personalization에서 identity-sensitive region은 cache하면 안 되는가, 아니면 오히려 identity drift를 막기 위해 cache해야 하는가? 예를 들어 얼굴 영역은 편집 대상일 수도 있고 보존 대상일 수도 있어, mask 정의가 task마다 달라질 수 있다.
- 질문: AR-oriented mask가 자동으로 잘 만들어지지 않는 경우, 잘못된 cache가 background preservation이 아니라 artifact persistence를 만들 수 있지 않은가?





## 저자가 정말로 증명한 게 뭐고 그냥 주장만 한 게 뭔가?

- 증명에 가까운 것: Springer/arXiv abstract와 project page 기준, LiveEdit는 streaming baseline들 대비 높은 시각 품질과 12.66 FPS inference speed를 보고하며, 실시간 interactive/AR application에 가까운 성능을 보인다.
- 증명에 가까운 것: three-stage pipeline과 AR-oriented Mask Cache가 결합될 때, bidirectional editing quality를 causal streaming setting으로 옮기는 실용적 방법이 된다는 것을 실험과 ablation으로 보인다.
- 증명에 가까운 것: cache를 모든 module에 동일하게 적용하는 것이 아니라, self-attention 중심으로 적용해야 품질 손상을 줄일 수 있다는 ablation 방향을 제시한다.
- 주장에 가까운 것: "real-time interactive deployment에 적합하다"는 표현은 사용 환경에 따라 달라진다. 실제 AR glasses, mobile device, high-resolution stream, user-in-the-loop latency에서는 별도 검증이 필요하다.
- 주장에 가까운 것: dedicated benchmark가 streaming video editing의 충분한 평가 기준이라는 점. benchmark가 실제 interactive editing의 사용자 만족도와 얼마나 일치하는지는 추가 검증이 필요하다.



## 이 결과가 놀랍나, 아니면 예상대로인가?

- 예상 가능한 점: offline/bidirectional model을 바로 causal로 바꾸면 품질이 무너질 수 있으므로, teacher forcing과 distillation이 필요하다는 방향은 자연스럽다.
- 흥미로운 점: editing mask를 speed-up과 preservation 둘 다에 쓰는 구조다. "편집할 영역만 바꾸고 나머지는 보존"이라는 editing constraint가 곧 "계산을 생략해도 되는 영역"으로 바뀐다.
- 놀라운 점: 단순 few-step distillation만으로 끝내지 않고, chunk-wise causal adaptation과 mask cache까지 함께 넣었다는 점이다. 실시간성은 sampling step 수만 줄인다고 해결되지 않고, temporal architecture와 region-wise computation이 같이 맞아야 한다.





## 저자의 결론이 실험 결과보다 과하게 일반화되진 않았나?

- 약간 과하게 읽을 수 있는 부분이 있다. 논문은 streaming video editing에서 좋은 결과를 보이지만, "real-time AR application에 적합"하다는 말은 실제 device, resolution, user interaction loop, mask generation latency까지 포함해야 완전해진다.
- 특히 long-video inference는 GitHub README에서 rolling attention sinks와 window-relative RoPE를 추가했다고 설명하지만, released checkpoint가 짧은 sequence 학습에 기반하므로 매우 긴 horizon으로 갈수록 품질 저하 가능성이 있다.
- 안전한 해석은 "LiveEdit는 bidirectional editing prior를 causal streaming editor로 옮기는 강한 recipe이며, AR-style 실시간 편집으로 가기 위한 중요한 prototype" 정도다. 바로 production-level AR editor라고 읽으면 과하다.





> **방법론**
>
{: .prompt-info }

## 이 방법이 왜 작동하는가, 그냥 "잘 됐다"가 아니라 원리적으로?

- 이 방법은 한 번에 세 문제를 풀려고 하지 않고 분해한다. 먼저 bidirectional model이 editing prior를 충분히 배우게 하고, 그 다음 causal constraint를 넣어 streaming 구조에 적응시키고, 마지막으로 DMD로 sampling step을 줄인다.
- Stage 1이 필요한 이유는 causal streaming model을 처음부터 학습하면 미래 context 없이 편집 품질을 배우기 어렵기 때문이다. 강한 offline teacher를 먼저 만들면, causal model이 따라갈 target distribution이 생긴다.
- Stage 2가 필요한 이유는 bidirectional attention을 causal attention으로 갑자기 바꾸면 attention distribution과 temporal dependency가 크게 바뀌기 때문이다. teacher forcing은 "미래를 못 보는 model"이 "미래를 보는 teacher"의 출력을 근사하도록 완충 역할을 한다.
- Stage 3 DMD가 필요한 이유는 real-time editing에서는 denoising step 수 자체가 latency를 지배하기 때문이다. 100-step/CFG 기반 model을 4-step generator로 압축해야 interactive speed가 가능해진다.
- Mask Cache가 작동하는 이유는 video editing에서 대부분의 frame 영역이 매 순간 새로 편집되는 것이 아니기 때문이다. 바뀌지 않는 background/non-edited region은 이전 chunk의 feature를 재사용해도 시각적 일관성을 해치지 않을 수 있다.
- 단, cache는 self-attention처럼 temporal/spatial context redundancy가 큰 module에 맞고, FFN처럼 token-wise transformation이 품질에 민감한 module에는 부적합할 수 있다.


## 숨겨진 가정이 뭔가?

- 편집 영역과 보존 영역을 mask로 충분히 안정적으로 구분할 수 있다는 가정.
- chunk-wise causal model이 bidirectional teacher의 품질을 충분히 근사할 수 있다는 가정.
- source video와 edit instruction 사이의 correspondence가 명확해서, 바뀌어야 할 영역과 유지되어야 할 영역이 크게 충돌하지 않는다는 가정.
- background/non-edited region의 feature는 chunk 사이에서 재사용 가능할 만큼 temporal redundancy가 크다는 가정.
- 4-step DMD generator가 다양한 instruction과 scene distribution에서도 충분한 fidelity를 유지한다는 가정.
- real-time FPS가 실제 interactive product에서 중요한 latency budget 전체를 대표한다는 가정. 실제로는 preprocessing, mask extraction, UI, video decoding/encoding, device transfer까지 포함해야 한다.






## 실험 설정

- Venue/status: ECCV 2026 conference paper. Springer chapter는 first online 2026-09-13, pp. 631-649로 등재되어 있다.
- Base model: official README 기준 Wan2.1-T2V-1.3B와 Self-Forcing codebase 위에 구축된다.
- Training pipeline: Foundation Tuning, Teacher Forcing for chunk-wise causal initialization, DMD for streaming video editing의 3단계.
- Inference target: causal chunk-by-chunk video-to-video editing. official README의 기본 inference는 released `ar-forcing_002000.pt` checkpoint를 사용한다.
- 핵심 지표/주장: project page와 abstract 기준 4-step distilled diffusion generation, causal DiT chunk-wise processing, Mask Cache token reuse, 12.66 FPS streaming inference.
- Baselines: project page에서 InsV2V, LucyEdit, VideoCoF, StreamDiffusion, StreamV2V와 qualitative comparison을 보여준다.
- Ablation: project page의 mask cache layer figure는 no cache, cache on self-attention, cache on FFN을 비교해 cache를 어디에 적용해야 하는지 보여준다.

- baseline이 공정한가?

- 공정성은 대체로 reasonable해 보이지만, editing target이 "source preservation + instruction following + speed"를 동시에 요구하므로 baseline별 목표가 완전히 같지는 않을 수 있다. 특히 offline editing baseline은 품질에는 강하지만 latency에서는 불리하고, streaming generation baseline은 speed에는 맞지만 preservation constraint가 약할 수 있다.
- 따라서 비교를 읽을 때는 "누가 더 예쁜가"보다 "실시간 조건에서 편집 영역만 바꾸고 나머지를 얼마나 유지하는가"로 봐야 한다.



> **한계**
>
{: .prompt-info }

## 직접 명시한 limitation 

- 공개 README의 long-video inference 설명에서는 rolling local-attention window를 사용하며, released checkpoint가 짧은 sequence에 맞춰 학습되었기 때문에 긴 horizon으로 extrapolate할 때 visual quality가 저하될 수 있다고 적는다.
- paper abstract 수준에서는 practical deployment가 background/non-edited region preservation과 low latency 때문에 어렵다고 전제한다. 즉, 이 두 문제가 아직 streaming video editing의 핵심 병목임을 인정한다.

## 저자가 명시하지 않은, 추정되는 한계 

- AR-oriented mask가 틀리면 cache가 오히려 오류를 고정할 수 있다. 잘못 보존된 feature가 다음 chunk로 계속 전달되면 artifact persistence가 생길 수 있다.
- local edit와 global edit가 섞이는 경우, "보존할 영역"과 "바꿀 영역"의 경계가 모호해져 mask cache가 불안정해질 수 있다.
- identity/person personalization에서는 얼굴, 옷, 손, 배경 등 어떤 영역을 보존해야 하는지가 instruction마다 달라진다. LiveEdit의 mask cache가 identity preservation까지 직접 해결한다고 보기는 어렵다.
- training cost와 data filtering이 적지 않다. 실시간 inference를 위해 사전에 강한 teacher와 distillation pipeline을 학습해야 한다.
- 12.66 FPS가 실제 사용자의 체감 latency를 보장하지는 않는다. I/O, decoding, UI interaction, mask computation이 합쳐지면 전체 latency는 달라질 수 있다.


## 이 방법이 실패하는 경우를 상상할 수 있나?

- edit instruction이 전체 scene style을 바꾸는 global edit라면, background token을 cache하는 것이 오히려 편집 강도를 제한할 수 있다.
- small object, hand, face detail처럼 identity-critical 영역이 mask에서 background로 분류되면, 필요한 편집이 누락되거나 기존 artifact가 유지될 수 있다.
- 빠르게 움직이는 camera나 object가 많으면 chunk 사이의 feature reuse가 잘못된 correspondence를 만들 수 있다.
- 장기 streaming에서 rolling window 밖의 context가 필요한 경우, causal attention과 limited cache만으로는 identity나 layout consistency를 유지하기 어렵다.
- 사용자가 interactive하게 instruction을 바꾸는 경우, 이전 cache를 얼마나 폐기하고 얼마나 재사용할지 결정하기 어렵다.



> **마무리**
>
{: .prompt-info }

## 한 문장 설명

LiveEdit는 offline bidirectional video editor의 품질을 causal streaming editor로 옮기고, mask cache로 비편집 영역 계산을 재사용해 real-time video editing을 가능하게 하려는 ECCV 2026 논문이다.


## 이 논문이 틀렸다면 어디가 가장 먼저 무너질까?

가장 먼저 무너질 지점은 **mask cache가 보존 영역을 안정적으로 구분하고 재사용해도 품질이 유지된다**는 가정이다. 이 가정이 틀리면 LiveEdit의 속도 이득은 artifact persistence나 identity/background drift를 만드는 원인이 되고, real-time editing이라는 장점이 품질 손실로 상쇄된다.

