---
layout: post
title: "[논문간단정리:수정예정] TOKENFLOW: CONSISTENT DIFFUSION FEATURES FOR CONSISTENT VIDEO EDITING"
date: 2026-05-11 12:50
categories: MyStudy
tags: AI video
math: true
---


>TOKENFLOW: CONSISTENT DIFFUSION FEATURES FOR CONSISTENT VIDEO EDITING(2023).Michal Geyer, Omer Bar-Tal, Shai Bagon, Tali Dekel
{: .prompt-tip }


TokenFlow: Consistent Diffusion Features for Consistent Video Editing은 텍스트 기반 비디오 편집의 핵심 과제인 temporal consistency을 텍스트-이미지 Diffusion 모델의 diffusion feature space에서 일관성을 강제함으로써 해결하는 프레임워크를 제안합니다. 기존 텍스트-이미지 Diffusion 모델을 비디오에 프레임별로 적용하면 시각적 품질은 높지만, 불일치한 패턴이나 flickering과 같은 시간적 불일치가 발생합니다. 이 연구는 비디오의 자연스러운 시간적 중복성이 Diffusion 모델의 내부 특징 표현에서도 유사하게 나타난다는 핵심 관찰에 기반하며, 이 특징 공간의 일관성을 유지함으로써 고품질의 시간적으로 일관된 비디오 편집이 가능함을 보여줍니다.

TokenFlow의 핵심 방법론은 다음과 같이 두 가지 주요 단계로 구성되며, 각 denoising timestep마다 반복적으로 수행됩니다.

1.  **Pre-processing: DDIM Inversion 및 Extracting Diffusion Features**
*   주어진 입력 비디오 $I = [I_1, ..., I_n]$에 대해, 각 프레임 $I_i$를 DDIM inversion을 사용하여 노이즈가 있는 latent space 시퀀스 $[x_{i1}, ..., x_{iT}]$로 변환합니다.
*   각 디노이징 스텝 $t$에서, $x_{it}$를 Diffusion 모델 $\epsilon_\theta$에 입력하고, 모델 내의 **모든 layer의 self-attention 모듈에서 tokens $\phi(x_{it})$를 추출합니다.** 이 토큰들은 원본 비디오의 프레임 간 특징 대응 관계를 설정하는 데 사용됩니다. 즉, 원본 비디오의 특징 공간에서 인접 프레임 간의 유사한 시각적 요소들이 어떻게 매핑되는지를 학습합니다.

2.  **Iterative Denoising Process**: Diffusion 모델의 reverse diffusion 과정인 $t=T$부터 $t=1$까지 각 스텝에서 다음 두 가지 단계를 번갈아 수행합니다.
*   **Keyframe Sampling and Joint Editing, Section 4.1:**
    *   현재 노이즈가 있는 비디오 $J_t$에서 무작위로 소수의 키프레임 인덱스 $\kappa$를 샘플링합니다.
    *   선택된 키프레임 $\{J_i\}_{i \in k}$ 는 extended-attention 블록을 사용하여 공동으로 디노이징되고 편집됩니다. 이는 표준 self-attention을 확장하여, 각 키프레임의 Query $Q_i$가 모든 키프레임의 키(Key) $K_{i_j}$와 값(Value) $V_{i_j}$를 참조하여 정보를 집계하도록 합니다. 수식으로는 다음과 같습니다:

    $$A = \text{Softmax}\left(\frac{Q_i K^\top}{\sqrt{d}}\right)$$
    
    여기서 $K = [K_{i_1}, ..., K_{i_k}]$는 모든 키프레임의 키를 연결한 것이며, $Q_i$는 단일 키프레임의 쿼리입니다. 이를 통해 키프레임들 간에 일관된 전역적 외형(global appearance)이 형성됩니다.
    *   이 공동 편집된 키프레임들로부터 "기준 토큰(base tokens)" $T_{\text{base}} = \{\phi(J_i)\}_{i \in \kappa}$을 추출합니다.

*   **TokenFlow Propagation, Section 4.2:**
    *   $T_{\text{base}}$를 기반으로, 원본 비디오 특징에서 미리 계산된 프레임 간 대응 관계를 활용하여 편집된 토큰을 비디오의 나머지 모든 프레임으로 전파합니다.
    *   각 원본 프레임의 토큰 $\phi(x_{it})$과 그에 인접한 (가장 가까운 과거와 미래) 키프레임들의 토큰 $\phi(x_{i^+t}), \phi(x_{i^-t})$ 간의 **Nearest Neighbor, NN 필드 $\gamma_{i^\pm}$를 계산합니다.** 이는 공간적 위치 $p$와 $q$에 대해 cosine distance를 사용하여 $q$를 찾는 과정입니다:

    $$\gamma_{i^\pm}[p] = \arg\min_q D(\phi(x_i)[p], \phi(x_{i^\pm})[q])$$

    *   이 NN 필드 $\gamma_{i^\pm}$를 사용하여, 편집된 프레임의 토큰 $T_{\text{base}}$를 나머지 프레임으로 선형 결합하여 전파합니다. 각 프레임 $i$의 특정 공간 위치 $p$에 대한 토큰은 다음과 같이 계산됩니다:
    
    $$F_\gamma(T_{\text{base}}, i, p) = w_i \cdot \phi(J_{i^+})[\gamma_{i^+}[p]] + (1 - w_i) \cdot \phi(J_{i^-})[\gamma_{i^-}[p]]$$

    여기서 $w_i$는 프레임 $i$와 인접 키프레임들($i^+, i^-$) 사이의 거리에 비례하는 스칼라 값으로, 부드러운 전환을 보장합니다. 이 과정을 통해 각 프레임의 self-attention 블록의 출력을 원본 비디오의 시간적 일관성을 반영하도록 조작합니다.
    *   최종적으로, 이 토큰들을 사용하여 전체 비디오 $J_t$를 디노이징하여 $J_{t-1}$을 얻습니다.

이 접근 방식은 이미지 편집 기술과 결합하여 작동하며, 추가적인 training이나 fine-tuning 없이도 시간적 일관성을 크게 향상시킵니다. TokenFlow는 Diffusion 모델의 내부 특징 공간에서 시간적 중복성을 활용하여, 원본 비디오의 움직임과 의미적 레이아웃을 보존하면서 고품질의 텍스트 기반 비디오 편집을 가능하게 합니다.