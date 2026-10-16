---
type: literature
title: "[논문정리] Repurposing Geometric Foundation Models for Multi-view Diffusion"
author: "Wooseok Jang, Seonghu Jeon, Jisang Han, Jinhyeok Choi, Minkyung Kwon, Seungryong Kim, Saining Xie, Sainan Liu"
year: "2026"
citekey: "jangRepurposingGeometricFoundation2026"
status: unread
tags:
  - literature
source: "http://arxiv.org/abs/2603.22275"
accept: "ECCV 2026 (European Conference on Computer Vision), accepted paper; ECCV 공식 accepted-papers 목록의 3D Graphics and Geometric Representations 트랙에 등재됨. 공식 목록에는 Poster Session 4, 위치 #85로 표시됨."
date: 2026-10-16 16:49:34 +0200
math: true
---




> **공식 링크 / acceptance 확인**
>
> - Project page: [Repurposing Geometric Foundation Models for Multi-view Diffusion](https://jeonseonghu.github.io/projects/repurposing-gfm/)
> - ECCV acceptance: [ECCV 2026 Accepted Papers](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)
> - Paper: [arXiv:2603.22275](https://arxiv.org/abs/2603.22275)
> - Code: [cvlab-kaist/GLD](https://github.com/cvlab-kaist/GLD)
> - acceptance 표기 기준: ECCV 공식 accepted-papers 목록에서 제목과 저자 전원이 일치하는 항목을 확인함. author/project page의 발표 형식 표기와 충돌하는 정보는 공식 ECCV 목록의 poster 표기를 우선함.
{: .prompt-info }


> **PDF**
>
{: .prompt-tip }

> [Preprint PDF](zotero://select/library/items/I3SJLMU4)



> **서지정보**
>
{: .prompt-info }

> Jang, Wooseok, Seonghu Jeon, Jisang Han, 기타. “Repurposing Geometric Foundation Models for Multi-view Diffusion”. arXiv:2603.22275. Preprint, arXiv, 2026년 3월 23일. [https://doi.org/10.48550/arXiv.2603.22275](https://doi.org/10.48550/arXiv.2603.22275).

  

> **초록(Abstract)**
>
{: .prompt-info }

> While recent advances in generative latent spaces have driven substantial progress in single-image generation, the optimal latent space for novel view synthesis (NVS) remains largely unexplored. In particular, NVS requires geometrically consistent generation across viewpoints, but existing approaches typically operate in a view-independent VAE latent space. In this paper, we propose Geometric Latent Diffusion (GLD), a framework that repurposes the geometrically consistent feature space of geometric foundation models as the latent space for multi-view diffusion. We show that these features not only support high-fidelity RGB reconstruction but also encode strong cross-view geometric correspondences, providing a well-suited latent space for NVS. Our experiments demonstrate that GLD outperforms both VAE and RAE on 2D image quality and 3D consistency metrics, while accelerating training by more than 4.4x compared to the VAE latent space. Notably, GLD remains competitive with state-of-the-art methods that leverage large-scale text-to-image pretraining, despite training its diffusion model from scratch without such generative pretraining.

  

## 한줄 요약
기하 정보를 이미 보존하는 foundation-model feature space에서 diffusion을 수행하면, multi-view 생성의 3D 일관성을 별도 geometry 조건으로 억지로 주입하기보다 latent 자체의 구조로 학습시킬 수 있고, VAE latent 대비 수렴도 빨라진다.
  
기존 diffusion 모델의 구조를 크게 바꾸지 않고 사용하는 latent space를 바꾸는 것만으로 생성 품질과 학습 효율을 개선한다. 

기존 latent diffusion 모델은 일반적으로 VAE가 압축한 이미지 표현을 사용한다. 하지만 VAE의 latent space는 이미지 복원에 초점을 맞추기 때문에 여러 카메라 시점 사이의 3d 기하학적 일관성을 충분히 표현하지 못할 수 있다. 



## 하이라이트 & 내 메모
### 핵심 하이라이트
- GLD는 view-independent VAE latent 대신 Depth Anything 3(DA3)·VGGT의 기하적으로 정렬된 feature space를 diffusion latent로 재사용한다.
- frozen geometric encoder의 feature에서 RGB를 복원하는 decoder를 먼저 학습해, 해당 feature가 생성에 필요한 photometric 정보를 충분히 담는지 검증한다.
- 전체 feature pyramid를 확산시키지 않고 boundary layer를 선택해 diffusion하고, 나머지 깊은 feature는 frozen encoder의 전파로 얻어 계산량을 줄인다.
- 생성된 feature는 RGB decoder뿐 아니라 원래 geometric decoder에도 입력할 수 있어, 추가 geometry 학습 없이 depth·camera pose·point cloud를 얻는다.
- VAE·RAE 대비 2D image quality와 3D consistency가 개선되고, VAE latent 대비 training convergence가 4.4배 이상 빠르다는 것이 핵심 실험 결과다.

### 내 메모
이 논문의 중요한 전환은 “diffusion 모델에 geometry 조건을 추가할 것인가”가 아니라 “생성 대상 latent가 처음부터 geometry를 보존하도록 만들 것인가”이다. 다만 빠른 수렴은 foundation model이 이미 제공하는 correspondence prior를 학습 초기에 활용한 결과이지, diffusion 자체가 geometry를 새롭게 발견했다는 증거는 아니다.

**Figure 흐름 메모:** 입력 multi-view 이미지 → frozen geometric foundation encoder → multi-level geometric features → boundary level(논문에서는 level 1)을 diffusion으로 생성 → 필요하면 cascade로 level 0을 생성하고 나머지 feature를 전파 → RGB decoder로 novel-view RGB 복원. 같은 생성 feature를 geometric decoder에 통과시키면 depth·camera pose·point cloud가 별도 학습 없이 나온다. 따라서 Figure의 핵심 비교는 (a) VAE latent에서 RGB만 생성하는 기존 흐름과 (b) geometry-aware latent에서 RGB와 geometry를 동시에 decode하는 GLD 흐름이다.
  



주로 2D 이미지 생성용 VAE 잠재 공간에서 학습되어 시점간 기하학적 일관성을 유지하는데 한계가 있었다.   
이를 해결하기 위해 외부 기하학적 조건을 주입하는 방식이 사용되었으나 이 연구는 기하학적 정보가 이미 내재된 geometric latent space를 직접 활용하는 것이 더 효율적이고 성능이 뛰어나다는 점을 입증한다. 

기하학적 정보가 풍부한 geometric foundation models의 특징 공간을 novel view synthesis를 위한 잠재 공간으로 재활용하는 geometric latent diffusion 프레임워크를 제안한다. 

기존 2d vae잠재 공간 대신 기하학적 일관성이 보장된 다층 특징을 사용하여 학습 수렴 속도를 4.4배 이상 가속화하고 외부 조건 없이도 높은 3d 일관성과 픽셀 수준의 복원 품질을 달성했다. 

![Pasted image 20261015172315](/assets/img/pasted-image-20261015172315.png)

**(a) VAE lantet diffusion 기존 방식**  
기존 모델에서는 VAE가 각 이미지를 압축하고 diffusion model이 새로운 카메라 시점에 해당하는 latent를 생성한다.   
VAE가 주로 이미지 재구성을 위해 학습되었다. multi view관계는 diffusion model이 학습과정에서 추가로 익혀야한다. 

**(b) geometric latent diffusion 제안 방식**  
geometric foundation model이 제공하는 feature를 새로운 latent representation으로 사용한다. 

RGB decoder: geometric feature에서 RGB 이미지를 복원하도록 새롭게 학습한다.   
3D Decoder: 이미 학습된 DA3의 geometry decoder를 재사용하며 추가학습 없이 depth와 raymap을 출력한다. 

이에 따라 RGB생성과 3D 복원을 동일한 latent representation에서 수행할 수 있다. 

(c) training efficiency 4.4 배 빠른 학습 수렴  
그래프의 x축은 학습 step(천단위) y축은 PSNR이다. 

주황색으로 GLD가 회색 VAE와 DINO보다 더 높은 PSNR에 빠르게 도달한다. 이를 기준으로 VAE보다 4.4배 빠른 수렴을 보고한다. 

그렇지만 추론 속도까지 4.4배 빨라졌다는 의미는 아니고 실제 생성시에는 GLD가 두단계의 sampling을 수행하므로 더 느리다. 


![Pasted image 20261016145553](/assets/img/pasted-image-20261016145553.png)


**(a) geometric latent diffusion framework**  
**주황색 부분 multi view diffusion models**  
geometric encoder로 depth anything 3(DA3)를 사용한다.   
이는 이미 여러 시점의 3D 구조를 학습한 모델이며 내부 transformer에서 네 단계의 feature를 추출할 수 있다. 

여기에서 level은 단순히 이미지 해상도를 4번 줄인 단계가 아니라 DA3 transformer의 서로 다른 깊이에서 추출한 표현이다. 

논문은 네개의 level을 모두 diffusion 으로 생성하지 않는다. 계산 비용도 크고 여러 level이 서로 불일치할 수 있기 때문이다. 


대신 두 모델을 사용한다. 

Main Diffusion Model ($M_1$)  
level 1의 multi view feature를 생성한다.   
구조와 appearance 정보가 균형을 이루는 latent를 생성한다.   
이 feature가 전체 모델의 기준점이 된다. 


Cascaded Diffusion Model ($M_{1\rightarrow0}$)  
이미 생성된 level 1을 조건으로 level 0를 생성한다.   
상대적으로 세밀한 이미지 정보와 질감을 보완한다.   
level0 과 level 1사이의 불일치를 줄인다. 


독립적으로 $F_0, F_1$을 생성하면 두 latent가 서로 다른 장면을 표현할 수 있다.   
cascaded 방식은 먼저 만들어진 구조에 맞추어 세부 정보를 생성하도록 한다. 


**보라색 부분 feature propagation**  
level 1에서 level 2, 3로 파란색 화살표가 내려간다.   
나머지 두 feature를 새로운 diffusion model로 만드는 대신, 기존 DA3 encoder의 나머지 layer에 통과시킨다. 

이때 DA3의 가중치는 frozen되어있다.   
level 2,3는 stochastic sampling이 아닌 level 1에 의해 결정되는 feature이다. 

이렇게 하게 된다면
- 4개 diffusion으로 학습하는 것보다 비용이 적다. 
- 깊은 feature가 level 1에 종속되어 일관성을 유지한다. 
- 사전학습된 DA3의 3D feture processing능력을 그대로 사용한다. 

**하늘색 부분 RGB Decoder**   
생성된 feature 4개가 준비된다.   
이들을 학습된 RGB Decoder에 입력한다. 

결과는 새로운 카메라 시점의 RGB 이미지이다. 

RGB decoder는 사전에 DA3 feature에서 이미지를 재구성하도록 학습한다. 반면 3D Decoder는 사전 학습된 DA3 의 decoder를 추가 학습 없이 재사용할 수 잇다는 차이가 있다. 

**Figure 2(b) — Multi-view Diffusion Model Architecture**

**input features** 

noisy, source, cascaded

target행의 source feature는 zero padding 되어있다. target RGB는 아직 없으므로 target 의 실제 source feature를 제공할 수 없기 때문이다. 

source이미지가 있다고 해서 source와 target의 joint feature까지 아는 것은 아니다.   
DA3의 multi view attention은 여러 이미지를 함께 입력하면 source feature도 변화시킨다. 따라서 이 모델은 source feature까지 포함하는 joint latnet 전체를 생성하도록 학습한다. 

**camera + timestep conditioning**

camera: source 및 target의 카메라 pose, plucker ray embedding과 PRoPE로 위치 및 방향 정보를 반영한다.   
timestep t : 현재 diffusion/flow 과정에서 어느 노이즈 단계에 있는지를 알려준다.   
AdaLN: timestep 및 조건 정보를 transformer 블록에 주입하는 방식이다. 

카메라 conditioning 덕분에 단순히 새로운 이미지를 만드는 것이 아니라 지정된 위치에서 촬영된 이미지를 생성할 수 있다. 

Enc. / Dec. 와 3D attention

Dec. 는 RGB Decoder가 아니라 diffusion 내부의 velocity decoder이다. 

3D attention은 서로 다른 view의 token들이 서로 정보를 주고 받도록 하는 역할을 한다.   
마지막 출력은 RGB 이미지가 아니라 flow matching에서 사용하는 velocity field이다. 

sampling이 끝나 latent가 완성된 이후에야 RGB Decoder를 사용한다. 






> **주장**
>
{: .prompt-info }

## 핵심 주장 / 방법론
1. NVS의 latent는 압축률만 좋은 VAE보다 cross-view correspondence와 3D 구조를 보존해야 한다.
2. GLD는 geometric foundation model을 새로 end-to-end 학습하지 않고 frozen feature space를 latent로 repurpose한다.
3. Stage 1에서 RGB decoder를 학습해 feature space의 photometric reconstructability를 확인한다.
4. Stage 2에서 boundary feature에 multi-view diffusion을 적용하고, cascade·feature propagation으로 계층적 feature를 복원한다.
5. 생성 결과를 RGB decoder와 geometric decoder 양쪽에 연결해 novel view synthesis와 zero-shot geometry decoding을 함께 수행한다.


## 내 생각 / 질문 / 반박
이 방식은 “좋은 생성 prior”보다 “좋은 좌표계”가 먼저일 수 있음을 보여준다. 하지만 DA3·VGGT의 correspondence가 깨지는 희소 시점, 심한 가림, 조명 변화에서는 latent의 기하 prior 자체가 잘못된 구조를 고정할 수 있다. 서로 다른 foundation model feature를 같은 latent처럼 취급해도 되는지, feature normalization과 decoder capacity가 성능을 얼마나 설명하는지 확인하고 싶다.



## 저자가 정말로 증명한 게 뭐고 그냥 주장만 한 게 뭔가?
증명에 가까운 것은 동일한 데이터·평가 조건에서 GLD가 VAE/RAE baseline보다 2D·3D 지표와 수렴 속도에서 우수하고, 생성 feature를 geometric decoder에 연결해 depth·pose를 산출할 수 있다는 점이다. “기하적으로 정렬된 latent가 일반적으로 최적이다”라는 주장은 DA3·VGGT, 선택한 데이터셋, 구현 설정 밖으로는 아직 일반화된 법칙이 아니다.


## 이 결과가 놀랍나, 아니면 예상대로인가?
부분적으로 예상된다. 이미 cross-view 구조를 담은 representation에서 diffusion하면 correspondence를 새로 학습할 부담이 줄기 때문이다. 놀라운 부분은 generative text pretraining 없이도 경쟁력 있는 RGB 품질과 geometry decoding을 동시에 얻었다는 점이며, 이는 photometric detail이 geometric feature에 충분히 남아 있음을 시사한다.



## 저자의 결론이 실험 결과보다 과하게 일반화되진 않았나?
그럴 가능성이 있다. 실험은 특정 geometric foundation model과 multi-view 벤치마크에 집중되어 있으므로, 모든 3D 생성 문제에서 geometry-aware feature가 VAE보다 우월하다고 해석하면 과하다. 또한 4.4배는 최종 inference latency가 아니라 training convergence 비교로 읽어야 한다.



> **방법론**
>
{: .prompt-info }

## 이 방법이 왜 작동하는가, 그냥 "잘 됐다"가 아니라 원리적으로?
VAE latent에서는 서로 다른 시점의 같은 3D 점이 단지 appearance가 비슷한 위치로 압축될 뿐, 동일한 좌표라는 보장이 없다. 반면 geometric foundation feature는 여러 시점의 대응·깊이·camera 구조를 feature 관계 안에 이미 encode한다. Diffusion은 이 좌표계에서 feature를 생성하므로, 모델이 매 샘플마다 geometry를 처음부터 추론하는 대신 기존 correspondence manifold 위에서 denoising하게 된다. frozen geometric decoder를 재사용할 수 있는 것도 생성 공간과 geometry 해석 공간이 일치하기 때문이다.


## 숨겨진 가정이 뭔가?
기하 foundation model의 feature가 충분히 photometric하게 표현 가능하고, 입력 시점의 overlap이 decoder가 요구하는 correspondence를 제공하며, diffusion이 학습한 feature가 frozen decoder가 기대하는 feature 분포를 유지한다고 가정한다. 또한 boundary layer 하나에서의 생성이 더 깊은 feature 전체를 안정적으로 결정할 수 있어야 한다.





## 실험 설정
주요 비교는 VAE·RAE·DINO latent 기반 diffusion 및 text-to-image pretraining을 사용하는 최신 multi-view 방법들이다. GLD는 RealEstate10K·DL3DV 등의 multi-view 데이터에서 RGB 품질과 3D consistency를 평가하고, ETH3D에서 depth/geometry decoding도 비교한다. DA3와 VGGT를 geometric backbone으로 사용하며, DA3는 RGB와 geometry decoder 양쪽 경로를 제공한다.

- baseline이 공정한가?
VAE/RAE와의 latent 비교 및 수렴 속도 비교는 핵심 가설에 적절하지만, foundation model의 대규모 사전학습을 baseline의 generative pretraining과 동일한 자원으로 환산하기 어렵다. 따라서 “더 좋은 latent”와 “더 강한 사전학습 prior”의 기여를 완전히 분리했다고 보기는 어렵다.



> **한계**
>
{: .prompt-info }

## 직접 명시한 limitation 
희소한 시점·심한 occlusion·큰 조명 변화처럼 foundation model의 correspondence가 불안정한 경우 생성 artifact와 잘못된 geometry가 남을 수 있다. 또 cascade sampling 때문에 inference는 단일-stage VAE 방식보다 느리다.

## 저자가 명시하지 않은, 추정되는 한계 
foundation model의 feature·decoder 버전에 대한 의존성이 크고, frozen decoder의 표현 범위를 넘어서는 새로운 appearance를 만들기 어렵다. boundary layer 선택과 cascade 설계가 특정 backbone에 맞춰져 있을 수 있으며, 4.4배 수렴 향상이 실제 wall-clock training cost·메모리·decoder preprocessing까지 포함한 비교인지는 별도로 확인해야 한다.


## 이 방법이 실패하는 경우를 상상할 수 있나?
관측 시점 사이에 공통 영역이 거의 없거나, 반사·투명·동적 물체처럼 단일 정적 3D correspondence가 성립하지 않으면 실패할 수 있다. 또한 geometric encoder가 잘못 추정한 depth를 diffusion이 수정하지 못하고 일관되게 유지할 가능성도 있다.


  

> **마무리**
>
{: .prompt-info }

## 한 문장 설명
GLD는 geometric foundation model의 correspondence-aware feature를 diffusion latent로 재사용해, multi-view RGB 생성과 3D geometry 복원을 하나의 표현 공간에서 수행한다.


## 이 논문이 틀렸다면 어디가 가장 먼저 무너질까?
VAE·RAE 대비 4.4배 빠른 수렴과 3D consistency 우위가 backbone 사전학습·decoder 설계·학습 budget을 통제한 뒤에도 유지되는지에서 먼저 무너질 것이다.

