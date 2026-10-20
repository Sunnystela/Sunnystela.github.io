---
type: literature
title: "[논문정리] AnyView: Synthesizing Any Novel View in Dynamic Scenes"
author: "Basile Van Hoorick, Dian Chen, Shun Iwase, Pavel Tokmakov, Muhammad Zubair Irshad, Igor Vasiljevic, Swati Gupta, Fangzhou Cheng, Sergey Zakharov, Vitor Campagnolo Guizilini"
year: "2026"
citekey: "vanhoorickAnyViewSynthesizingAny2026"
status: unread
tags:
  - literature
source: "https://arxiv.org/abs/2601.16982v2"
accept: "ECCV 2026 (European Conference on Computer Vision); official project page and official code repository identify the paper as an ECCV 2026 paper."
date: 2026-10-20 17:57:18 +0200
math: true
---




> **공식 링크 / acceptance 확인**
>
> - Project page: [AnyView: Synthesizing Any Novel View in Dynamic Scenes](https://tri-ml.github.io/AnyView/)
> - ECCV venue 확인: project page와 [official code repository](https://github.com/TRI-ML/AnyView-DVS) 모두 `ECCV 2026`으로 표기
> - Paper: [arXiv:2601.16982](https://arxiv.org/abs/2601.16982)
> - Paper PDF: [AnyView PDF](https://tri-ml.github.io/AnyView/AnyView.pdf)
> - Code / benchmark: [TRI-ML/AnyView-DVS](https://github.com/TRI-ML/AnyView-DVS)
> - 현재 확인 가능한 공식 자료에는 ECCV 2026 venue 표기는 있으나, proceedings volume·page·발표 형식은 확인되지 않아 추가로 추정하지 않음.
{: .prompt-info }

> **PDF**
>
{: .prompt-tip }

> [Full Text PDF](zotero://select/library/items/QUJ93L9A)



> **서지정보**
>
{: .prompt-info }

> Van Hoorick, Basile, Dian Chen, Shun Iwase, 기타. “AnyView: Synthesizing Any Novel View in Dynamic Scenes”. arXiv.Org, 2026년 1월 23일. [https://arxiv.org/abs/2601.16982v2](https://arxiv.org/abs/2601.16982v2).

  

> **초록(Abstract)**
>
{: .prompt-info }

> Modern generative video models excel at producing convincing, high-quality outputs, but struggle to maintain multi-view and spatiotemporal consistency in highly dynamic real-world environments. In this work, we introduce $\textbf{AnyView}$, a diffusion-based video generation framework for $\textit{dynamic view synthesis}$ with minimal inductive biases or geometric assumptions. We leverage multiple data sources with various levels of supervision, including monocular (2D), multi-view static (3D) and multi-view dynamic (4D) datasets, to train a generalist spatiotemporal implicit representation capable of producing zero-shot novel videos from arbitrary camera locations and trajectories. We evaluate AnyView on standard benchmarks, showing competitive results with the current state of the art, and propose $\textbf{AnyViewBench}$, a challenging new benchmark tailored towards $\textit{extreme}$ dynamic view synthesis in diverse real-world scenarios. In this more dramatic setting, we find that most baselines drastically degrade in performance, as they require significant overlap between viewpoints, while AnyView maintains the ability to produce realistic, plausible, and spatiotemporally consistent videos when prompted from $\textit{any}$ viewpoint. Results, data, code, and models can be viewed at: https://tri-ml.github.io/AnyView/

  

## 한줄 요약
AnyView는 한 개의 동적 입력 비디오와 입력·목표 카메라의 Plücker ray 조건만으로, 명시적 3D 재구성이나 test-time optimization 없이 극단적으로 다른 시점의 시간 동기화 비디오를 생성한다.
  

## 하이라이트 & 내 메모
### 핵심 하이라이트
- 목표는 dynamic view synthesis: 한 카메라 trajectory의 비디오가 주어졌을 때 같은 장면·동작을 다른 camera trajectory에서 생성한다.
- Cosmos latent diffusion transformer를 기반으로 입력·목표 카메라의 intrinsics/extrinsics를 dense Plücker representation으로 인코딩한다.
- depth map을 명시적 conditioning으로 넣거나 test-time 3D reconstruction을 수행하지 않고, 대규모 2D·3D·4D 데이터에서 implicit 4D representation을 학습한다.
- 12개 4D 데이터셋을 robotics, driving, 3D, other 네 도메인으로 균형 샘플링하고, 새 Kubric-5D와 AnyViewBench를 함께 제안한다.
- 좁은 시점 변화가 아니라 입력·목표 시점 overlap이 거의 없는 extreme camera displacement에서 기존 방법이 입력 시점을 복사하거나 붕괴하는지 평가한다.

### 내 메모

![Pasted image 20261019164805](/assets/img/pasted-image-20261019164805.png)  
**초록색**: input camera trajectory: 실제 원본 영상을 촬영한 카메라의 위치와 방향 변화  
**주황색**: target caemra tajectory: 새롭게 생성하려는 영상의 카메라 위치와 방향 변화

input과 target이 같은 시간의 장면을 서로 다른 위치에서 관찰한다. 

미리 프레임을 예측하는 일반적인 video prediction과 다르게 anyview는 같은 시간에 다른 카메라에서 무엇이 보일지 예측한다. 

**Plücker Embeddings**: 카메라가 정보를 어떻게 표현할지  
일반적인 카메라 정보는 위치, 회전, 초점 거리 같은 숫자로 표현할 수 있지만 이 논문에서는 카메라 정보를 각 픽셀에서 3D 공간으로 나가는 광선으로 표현한다. 

r은 광선의 방향 벡터로 3차원  
o는 카메라 중심  
m은 광선의 공간적 위치를 표현하는 moment벡터 3차원

이렇게 하면 모델이 픽셀의 색상 뿐만 아니라 그 픽셀이 3D 공간의 어느 방향과 위치를 바라보는지에 대한정보도 받는다. 

원본 카메라와 목표 카메라 모두 embedding을 계산해 입력해가며 각 픽셀마다 6채널을 갖는 맵이 된다. 

RGB Frames 는 장면의 실제 시각적 정보  
Plücker Embeddings는 픽셀별 3D 광선 정보  
Video Tokenizer는 두 정보를 압축된 latent token으로 변환  
Token Stack은 두 시점의 정보를 하나의 transformer입력으로 구성한다. 

input view는 관찰한 깨끗한 RGB영상을 사용하고   
target view는 생성 과정에서 노이즈가 섞인 RGB latent를 사용한다. 

RGB와 Plucker 정보를 각각 인코딩한 뒤, 동일한 시점에 속한 두 종류의 latent 를 채널 방향으로 연결한다. 

**Diffusion Transformer**  
실제 생성이 일어나는 부분이다.   
파란색 박스가 핵심 생성 모델이다.   
AnyView는 NVIDIA cosmos기반의 latent diffusion transformer를 사용한다. 

여기 두개 정보가 들어간다. 

3D RoPE  
transformer가 토큰의 시간 및 영상 내 공간 위치를 구분하도록 하는 위치 인코딩이다  
여기서 3D는 실제 물체의 3D 좌표를 직접 복원하는 것이 아닌 영상 latent 세 축 시간, 높이, 너비를 의미한다. 

Timestep  
현재 diffusion 의 노이즈 제거 단계를 나타내는 정보다. 영상 속 몇 번째 프레임인지 나타내는 시간 인덱스와는 다르다.   
모델은 영상 정보와 목표 카메라 plucker 정보를 참고하면서 target view의 노이즈를 점진적으로 제거한다. 

단순히 RGB 영상을 복사하거나 시점을 바꾸는 것이 아닌 모델이 학습한 장면과 움직임에 관한 지식을 활용해 목표 시점의 영상을 생성한다. 

**Video Decoder -> Prediction / GT**  
마지막으로 diffusion transformer가 만든 Target RGB latnet를 video decoder로 복원하면 실제 RGB 동영상이 나온다. 




이 논문의 핵심은 geometry를 제거한 것이 아니라 geometry를 “복원 단계”에서 “조건부 생성이 학습하는 implicit representation”으로 옮긴 것이다. 따라서 explicit 3D가 필요 없다는 말은 geometry가 불필요하다는 뜻이 아니라, 여러 도메인과 시간축을 본 diffusion model이 camera ray 조건과 appearance·motion prior를 함께 사용해 geometry를 암묵적으로 추정한다는 뜻이다.

**Figure 흐름 메모:** 입력 비디오와 입력 camera trajectory → 각 픽셀의 ray direction·moment를 담은 Plücker embedding 생성 → 입력 view와 target view 조건을 sequence 차원에서 함께 구성 → Cosmos latent diffusion transformer가 target-view video latent를 생성 → latent decoder가 시간 동기화된 RGB 비디오를 출력한다. Figure 1의 비교에서는 depth reprojection·point cloud 또는 기존 trajectory-aware baseline이 큰 시점 이동에서 입력 view를 복사하거나 구조를 잃는 반면, AnyView는 geometry·appearance·dynamic motion을 함께 유지하는지를 보여준다.
  
![Pasted image 20261020174653](/assets/img/pasted-image-20261020174653.png)

AnyView가 기존 학습 방식보다 새로운 카메라 시점을 얼마나 정확하게 생성하는지 비교한 정량 평가 결과

1. from scratch/ gcd data, ours: 세가지 학습 설정 비교
2. in distribusion/zero shot: 학습에 사용한 데이터 분포와 새로운 분포에서의 성능 비교
3. psnr/ssim/lpips: 생성된 영상과 실제 정답 영상의 유사도를 평가하는 지표

PSNR: 생성 영상과 gt 픽셀 단위 오차  
SSIM: 두 영상의 밝기, 대비, 구조적 유사성  
LPIPS: 딥러닝 특징 기반의 지각적 차이 (낮을수록 좋음)

from scratch: anyview의 전체 데이터  
GCD data: Kubric 4D, parallel domain 4D 데이터 사용   
ours: 다양한 도메인의 전체 데이터 사용

사전학습된 video foundation model이 중요한지, 다양한 데이터셋을 함께 학습하는 것이 일반화 성능을 높이는지 보고자함

in distribusion : 학습 과정에서 접한 데이터 분포와 유사한 평가 환경. 정확히 같은 훈련 영상을 테스트하는 것이 아님

결과적으로  
video foundation model의 사전 학습이 중요하다. 이미 다양한 영상에서 외형, 움직임, 시간적 패턴을 학습한 cosmos모델을 활용하기 때문에 제한된 3D/4D 데이터만으로 학습하는 것보다 일반적으로 유리하다. 

단일한 합성 데이터 환경에서 잘 동작하는 모델ㅇ리라도 자율주행, 로봇, 사람의 행동 등 서로 다른 환경에서는 성능이 떨어질 수 있다. 

Anyview는 2D, 3D, 4D 데이터를 결합해 이러한 일반화 문제를 완화하려고 한다. 




> **주장**
>
{: .prompt-info }

## 핵심 주장 / 방법론
1. 극단적 dynamic view synthesis는 explicit scene reconstruction이나 test-time optimization 없이도 end-to-end로 학습할 수 있다.
2. 카메라 intrinsics와 extrinsics를 dense per-pixel Plücker representation으로 통합하면, 입력과 목표 시점의 ray geometry를 동일한 conditioning 형식으로 제공할 수 있다.
3. 입력·목표 view를 sequence dimension에서 함께 처리하는 latent diffusion transformer가 관측된 motion과 보이지 않는 영역의 appearance를 implicit 4D representation으로 모델링한다.
4. 다양한 4D 데이터와 일부 2D·3D supervision을 혼합해 특정 카메라 설정에 과적합되지 않는 generalist 모델을 학습한다.
5. AnyViewBench는 큰 순간적 viewpoint change를 중심으로 driving·robotics·human activity 등 여러 도메인의 zero-shot 일반화를 측정한다.


## 내 생각 / 질문 / 반박
“any viewpoint”라는 표현은 물리적으로 모든 시점을 복원한다는 뜻이 아니라, 학습 분포가 허용하는 camera ray와 scene prior를 이용해 plausible한 영상을 생성한다는 뜻으로 읽어야 한다. 목표 시점에 관측 정보가 전혀 없을 때 ground truth를 정확히 복원할 수 없으므로, realism·temporal consistency와 exact reconstruction을 분리해 평가해야 한다. 또 explicit geometry를 쓰지 않는 대신, 모델 크기와 데이터 다양성에 geometry prior를 흡수한 것은 아닌지 확인할 필요가 있다.



## 저자가 정말로 증명한 게 뭐고 그냥 주장만 한 게 뭔가?
증명에 가까운 것은 공개된 benchmark 조건에서 입력 시점과 목표 시점의 변화가 커질수록 기존 baseline보다 AnyView의 생성 결과가 더 plausible하고 시간적으로 일관되며, 여러 도메인에서 동작한다는 실험적 비교다. 반면 “임의의 시점”과 “advanced common-sense reasoning”은 관측되지 않은 영역을 그럴듯하게 채우는 생성 능력에 대한 표현이며, 보이지 않는 장면의 실제 정답을 복원했다는 의미로 해석할 수 없다.


## 이 결과가 놀랍나, 아니면 예상대로인가?
camera ray를 명시적으로 조건으로 주고 충분한 multi-view dynamic 데이터를 학습하면 좁은 시점 변화보다 넓은 시점 extrapolation이 나아질 것이라는 방향은 예상 가능하다. 놀라운 점은 별도의 depth reprojection이나 3D optimization 없이도, 동적 객체와 배경의 시간 일관성을 한 latent diffusion model 안에서 함께 유지하려는 설계를 실제 extreme benchmark까지 확장했다는 것이다.



## 저자의 결론이 실험 결과보다 과하게 일반화되진 않았나?
그렇다. AnyViewBench에서 좋은 결과를 내더라도 실제 카메라가 장면 밖으로 이동하거나, 가림·반사·투명성·새로운 객체처럼 training prior로 설명하기 어려운 경우까지 보장하지 않는다. 특히 “any”는 camera trajectory의 형식적 범위와 benchmark 범위 안에서 이해해야 하며, 정확한 geometry 또는 물리적 미래 예측의 보장은 아니다.



> **방법론**
>
{: .prompt-info }

## 이 방법이 왜 작동하는가, 그냥 "잘 됐다"가 아니라 원리적으로?
Plücker embedding은 카메라 pose를 단순한 전역 벡터가 아니라 각 픽셀 ray의 방향과 moment로 바꿔, 모델이 입력·목표 영상의 대응 관계를 image token과 같은 공간에서 볼 수 있게 한다. diffusion transformer는 입력 video latent에서 시간적 motion·appearance를 추출하고, 목표 ray 조건에 맞는 target video latent를 생성한다. multi-domain 4D 학습은 특정 장면의 explicit geometry를 저장하기보다 “이 motion과 appearance가 다른 ray에서 어떻게 보이는가”라는 조건부 분포를 학습하게 한다.


## 숨겨진 가정이 뭔가?
카메라 파라미터가 정확히 알려져 있고 입력과 목표 영상이 시간적으로 동기화되며, 훈련 데이터가 새로운 시점에서 필요한 motion·appearance prior를 충분히 포함한다고 가정한다. 또한 모델이 ray 조건과 scene identity를 혼동하지 않고, 관측되지 않은 영역을 생성할 때도 장면의 object permanence와 dynamics를 안정적으로 유지한다고 가정한다.





## 실험 설정
프로젝트 페이지는 DROID, Kubric, Ego-Exo4D 및 driving·robotics·human activity를 포함한 여러 데이터와 AnyViewBench를 제시한다. AnyViewBench는 zero-shot과 in-distribution split을 나누고, 입력 video·target video·양쪽 camera parameter를 제공한다. 비교 대상은 depth reprojection/point cloud 기반 방법과 trajectory-aware video generation baseline이며, 평가는 좁은 변화뿐 아니라 extreme camera displacement와 복잡한 trajectory를 포함한다.

- baseline이 공정한가?
AnyViewBench에서 모든 방법에 입력·목표 camera parameter와 동일한 target을 제공한다면 conditioning 비교는 공정하다. 다만 AnyView는 대규모 4D 데이터와 Cosmos backbone을 활용하므로, explicit geometry baseline과의 비교에서 데이터 규모·사전학습·계산량을 동일하게 맞췄는지 확인해야 한다. 또한 perceptual plausibility가 높은 결과와 pixel-level ground truth 복원의 우열은 별도 지표로 봐야 한다.



> **한계**
>
{: .prompt-info }

## 직접 명시한 limitation 
극단적 시점에서 보이지 않는 영역은 ground truth와 정확히 일치하지 않을 수 있고, 결과가 plausible하더라도 입력 영상과 완전히 동일한 scene geometry를 보장하지 않는다. 기존 baseline도 큰 viewpoint change에서 입력 시점을 복사하거나 extrapolation에 실패하며, 이 문제 자체가 AnyViewBench의 핵심 난점이다.

## 저자가 명시하지 않은, 추정되는 한계 
implicit representation은 명시적 3D 구조처럼 object identity·occlusion·metric depth를 검사하거나 편집하기 어렵다. camera calibration 오차, 시간 동기화 오차, 학습 분포 밖의 motion은 ray conditioning을 잘못된 문제로 만들 수 있다. 또한 “no geometric assumptions”라고 해도 Plücker ray와 multi-view video 데이터가 강한 geometric inductive signal을 제공한다.


## 이 방법이 실패하는 경우를 상상할 수 있나?
입력 카메라와 목표 카메라 사이에 공통 관측이 거의 없고, 객체가 빠르게 변형되거나 사라지며, 반사·투명·자기 가림이 많은 경우 실패할 수 있다. 카메라 파라미터가 부정확하거나 목표 trajectory가 훈련 분포에서 크게 벗어나면 motion이 시간축에서 흔들리거나 background와 object가 섞일 가능성이 높다.


> **내 연구 연결**
>
{: .prompt-info }

## 현재 진행중인 연구와 연결될만한 점
세포 영상에서 관측되지 않은 시점 대신 관측되지 않은 미래를 생성한다는 관점으로 연결할 수 있다. AnyView의 핵심처럼 입력 trajectory와 target trajectory를 명시하면, 세포의 현재 morphology·motion 상태를 조건으로 미래 trajectory를 생성하는 문제를 구성할 수 있다. 다만 세포에서는 camera ray보다 cell identity·neighbor interaction·lineage state가 핵심 조건이므로, implicit 4D prior를 biology-aware state representation으로 바꿔야 한다.


## 관련될 수 있는 개념/질문 (후보 제안, 질문 형태로)
시점 변화의 크기와 예측 불확실성을 함께 추정할 수 있는가? / 명시적 geometry 없이 생성된 dynamic scene의 temporal consistency를 어떤 correspondence metric으로 검증할 것인가? / 관측되지 않은 영역의 plausibility와 실제 정답 복원 성능을 어떻게 분리할 것인가? / multi-domain training에서 domain 균형 샘플링이 camera generalization과 motion generalization 중 무엇을 더 개선하는가?


## 연결된 노트

- 관련 프로젝트: 0. 창고

- 관련 개념: 카메라 조건은 명시적 3D 대신 생성 모델의 시점 일반화를 조절한다

  

> **마무리**
>
{: .prompt-info }

## 한 문장 설명
AnyView는 Plücker camera conditioning과 multi-domain 4D 학습을 이용해, 한 동적 입력 비디오에서 큰 시점 변화에도 시간적으로 일관된 target-view 비디오를 생성한다.


## 이 논문이 틀렸다면 어디가 가장 먼저 무너질까?
극단적 viewpoint change에서 AnyView의 개선이 실제 camera conditioning의 효과가 아니라, 더 큰 Cosmos backbone·더 많은 4D 데이터·더 강한 생성 prior의 효과라는 통제 실험에서 먼저 무너질 것이다.


## 다음 액션

- [ ] 영구노트로 승격할 아이디어가 있는지 확인하여 작성
- [x] 카메라 조건은 명시적 3D 대신 생성 모델의 시점 일반화를 조절한다 영구노트로 승격
