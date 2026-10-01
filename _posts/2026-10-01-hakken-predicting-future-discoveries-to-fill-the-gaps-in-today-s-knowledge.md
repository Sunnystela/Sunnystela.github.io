---
type: post
title: "[논문정리] Hakken: Predicting future discoveries to fill the gaps in today's knowledge"
date:   2026-10-01 10:17
categories: AI LLM
tag: LLM
math: true
---



> **공식 링크 / 확인**
>
> - arXiv: [2609.04494](https://arxiv.org/abs/2609.04494)
> - PDF: [arXiv PDF](https://arxiv.org/pdf/2609.04494)
> - DOI: [10.48550/arXiv.2609.04494](https://doi.org/10.48550/arXiv.2609.04494)
> - title/author 확인: arXiv metadata와 PDF first page의 title/author가 현재 노트와 일치한다.
> - venue 확인: 2026-09-03 arXiv preprint이며, 신뢰 가능한 공개 자료 기준으로 학회/저널 accept 정보는 확인되지 않는다.
> - 별도 project website / code repository는 확인하지 못했고, 공식 공개 자료는 arXiv abstract/PDF이다.
{: .prompt-info }

### 관련 프로젝트 : C. AI Agent, RAG, 0. Horizon Europe AI Agent

> **PDF**
{: .prompt-tip }

> [Preprint PDF](zotero://select/library/items/Y2F2PN5H)



> **서지정보**
>
{: .prompt-info }

> Besold, Tarek R., Uchenna Akujuobi, Pablo Sanchez, 기타. “Hakken: Predicting future discoveries to fill the gaps in today’s knowledge”. arXiv:2609.04494. Preprint, arXiv, 2026년 9월 3일. [https://doi.org/10.48550/arXiv.2609.04494](https://doi.org/10.48550/arXiv.2609.04494).

  

> **초록(Abstract)**
>
{: .prompt-info }

> We present Hakken, a domain-agnostic prediction and explanation system performing knowledge prediction, i.e., growing scientific knowledge by establishing novel relationships, ones that are not limited to the deductive hull of previous knowledge. Hakken uses a transformer-based prediction model built on temporal sequences of knowledge graphs extracted from vast bodies of research publications, fused with an LLM's semantic knowledge, to predict the presence and define the type of as-yet undocumented relationships between scientific concepts. It then calls a model-agnostic explanation framework to provide accompanying information for each prediction that allows scientists to evaluate the suggested new relationship. While general purpose, we demonstrate Hakken's practical capabilities by applying it to the biomedical domain. There, Hakken's prediction model establishes a new benchmark for time-aware multi-label relation prediction, and we show that the model's output stays coherent and informative over extended time spans in historic data. In addition, we scored 1.5 million above-confidence-threshold hypotheses related to aging, qualitatively validated batches of these predictions with biologists and progressed three of them for empirical validation in wet-lab. Two predictions with potentially significant impact in the context of drug discovery and repurposing were confirmed, introducing previously undocumented interactions between TP53 and BAMBI, and between RAF1 and TNF, to biomedical science.

  
### TODO



## 한줄 요약

>Hakken은 논문에서 추출한 temporal knowledge graph와 LLM semantic representation을 결합해 아직 문헌에 기록되지 않은 과학 관계를 예측하고, PHELInE로 설명을 붙인 뒤 일부 biomedical hypothesis를 wet-lab 검증까지 보낸 AI for Science 시스템이다.

아직 예측되지 않은 과학적 관계를 AI가 미리 예측해서 새로운 연구 가설을 제안한다는 논문

A유전자가 B유전자 발현에 영향을 줄 것이다

그런 관계를 예측하고 실제 wet lab 실험까지 해서 일부를 검증했다. 

과거 논문들이 시간에 따라 어떻게 쌓였는지 학습해서   
문헌에 존재하지 않는 결과를 예측한다.   
이미 존재하는 지식을 복원하는 게 아니라 미래에 발견될 가능성이 있는 관계를 예측하는 것이 목표다 



  
  

## 하이라이트 & 내 메모

핵심 모델: THiGERLLM

Temporal Hierarchical Graph-based Encoder Representation with LLM  
prediction


PHELInE = explanation


![Pasted image 20260916133258](/assets/img/pasted-image-20260916133258.png)  
Hakken 안의 예측 모델인 THiGERLLM의 전체 forward 흐름

> **THiGERLLM**
>
> THiGERLLM은 temporal knowledge graph를 처리하는 graph branch와 literature text를 처리하는 language branch로 구성된다.   
> Graph branch에서 얻은 temporal graph representation을 LLM token으로 변환해 language branch에도 주입하고 마지막에 두 branch의 relation score를 ensemble하여 최종 multi-label relation을 예측한다.  
> 관계 예측하고 그 예측에 대한 점수도 같이 낸다. 
{: .prompt-info }


entity pair (s,o)에 대해

> **entity pair?**
>
> 그냥 두개 entity를 한쌍으로 묶은 것이다
{: .prompt-info }

그래프에서 본 구조적, 시각적 evidence와   
논문 text에서 본 semantic evidence를 따로 계산한 뒤   
마지막에 합쳐서 어떤 relation이 성립할지 예측한다. 

가장 아래 왼쪽 노란 박스에서 시작한다. 


**Temporal Knowledge Graph**  
입력은 시간별 그래프 snaphot인 G1, G2,,, GT이고 특정 entity pair (u,v)와 그 두 entity의 T개 시점에 걸친 neighborhood를 가져온다. 

>그래프의 snapshot?  
>각각 몇년  
>(u,v)



**Graph Branch**

**Hierarchiacal Graph Transformer**  
이 블록은 시간별 graph 정보를 합쳐서 네가지 출력을 만든다. 

$z_{pos}$ : 실제 traget entity apair (u,v)의 임베딩  
$z_{neg}$ : contrasive learning 용 negative pair  
$z_{temporal}$ : 시간별 정보를 남겨둔 temporal embeddings  
$E_{rel}$ : 각 relation/label의 embedding


z_pos, E_rel은 위쪽 graph scoring head로 간다.   
기본 relation score은 

$z_{pos}​⋅E_{rel}​$

history correction 도 추가된다. 


**Graph -> LLM Bridge**

$z_{pos}$는 pair projector 로 들어가고  
$z_temporal$은 temporal mapper로 들어간다. 

**stop-gradient (detach)**  
LM 쪽 loss가 graph encoder까지 그대로 역전파되지 않도록 graph representation 을 detach해서 넘기는 설계이다. 


**pair projector**  
target pair의 graph embedding을 LLM embedding 차원으로 바꾼다.   
graph vector를 LLM이 받을 수 있는 representation으로 바꾼다. 

**temporal mapper**  
$z_{temporal}$ 시간대별 graph embedding들을 LLM 쪽 text/token space에 맞게 바꾼다. 

pair 을 대표하는 vector과 시간 변화 정보도 별도로 압축해서 LLM에 넣는다. 


**temporal context**  
LM branch는 graph 정보뿐만 아니라 실제 textual evidence도 같이 받는다. 

**decoder LLM**  
casual self-attention 으로 전체 context를 업데이트한다.   
두 entity marker에 대응하는 hidden representation을 이용하여 entity marker pooling을 한다. 

두 entity 관계를 판단하는데 필요한 LM presentation을 $h_{pooled}$ 하나로 만든다. 


History-aware cross-attention  
으로 처리해서   
history_ligits를 만들고  
prediction에 correction을 더한다. 


Text scoring Head  
$h_{pooled}$ 를 받아 relation별 logit을 계산하고 history correction을 더한다. 

독립적으로 graph logits랑 text logits가 생기게 된다. 

**Joint-training & inference**

위에서 만들어진 두 개를 저기에 넣게 된다.   
두 branch의 prediction을 calibration한 뒤 ensemble한다. 


**Multi-label relation prediction**  
최종 출력이   
$y^​∈\{0,1\}^L$

relation이 L개라면 각각 0또는 1을 예측하여 한 pair에 relation 하나만 나오는게 아니라 여러개가 동시에 선택될 수 있다. 



![Pasted image 20260916170603](/assets/img/pasted-image-20260916170603.png)

THiGERLLM이 왜 이런 새로운 관계를 예측했는지 설명하는 PHELInE explainability framework이다.   
위위 그림는 예측기였고 이번 figure는 그 예측이 나온 뒤 설명 경로를 찾는 후처리 단계인 것이다. 

기존 Knowledge Graph에서 어떤 기존 사실들의 경로 때문에 이 예측이 나온 것처럼 보이는지 찾아준다. 

내부 gradient를 분석하는 대신 GraphSAGE surrogate model을 하나 만들어서 후보 경로의 영향력을 빠르게 측정한다. 


> **큰 흐름 정리**
>
> ### OFFLINE
> 1. THiGERLLM이 여러 triple에 점수를 냄
> 2. GraphSAGE가 그 점수를 모방하도록 학습
> 3. surrogate freeze
>
> ### ONLINE
> 4. 새 hypothesis 입력
> 5. KG에서 연결 path 후보 생성
> 6. 각 path를 넣거나 제거한 graph를 surrogate에 넣음
> 7. sufficiency / necessity 계산
> 8. influence가 큰 path를 ranking
> 9. 서로 비슷하지 않은 top path들을 최종 explanation으로 선택
{: .prompt-info }



**맨 위에 기호들 정리**

G  
전체 knowledge graph

$f_θ$​  
원래 예측기, 즉 THiGERLLM scorer

$f_{ϕ^∗}​$  
THiGERLLM을 흉내 내도록 학습한 frozen surrogate model

E  
하나의 설명 후보를 이루는 triple들의 집합

$G_E$   
E라는 path만 포함한 subgraph

$G\E$  
전체 graph G에서 E의 triple들을 제거한 graph


먼저 THiGERLLM이 hypoyhesis를 하나 만든다. 




**Hypothesis (s,r,o): (TP53, AFFECTS_EXPRESSION_OF, BAMBI)**

앞에서 배운 THiGERLLM이 먼저   
(TP53, AFFECTS_EXPRESSION_OF, BAMBI) 라는 새로운 relation을 높은 confidence로 예측한 상황인 것이다. 


TP53 ── affects_expression_of ──> BAMBI

이런 관계가 발견될 가능성이 높다고 예측하는 것이다.  

여기서 PHELInE가 작동을 하는 것이다.   
이게 가설을 만드는 것은 아니고  
설명을 생성하는 것  
왜 이 관계를 높게 봤는가를

### **OFFLINE**

**Training data만들기**

**Training data: $(s,r,o)^+, (s,r,o)^-$**  
**scored by THiGERLLM**

$(s,r,o)^+$ = **positive triple**  
실제 training KG에 존재하는, 관찰된 관계  
$(s,r,o)^-$ = **negative sample**  
비교를 위해 만든 관계 없는/관찰되지 않은 triple

여기에 많은 positive/negative triple을 THiGERLLM에 넣어서   
fθ​(s,r,o) : score  
을 얻는다. 

GraphSAGE에게 정답 biological relation을 직접 가르치는 것이 아닌 몇 점을 줄 것인지를 가르치는 것이다

> **GraphSAGE?**
>
> 그래프에서 각 노드가 주변 이웃 노드들의 정보를 모아서 자기 임베딩을 만드는 GNN 모델이다. Graph SAmple and affreGate  
> haken이 새로 주장한건 아니다.   
> 각 노드 주변에서 일정수의 이웃만 샘플링해서 aggregation 하는 것이다.   
> GraphSAGE가 PHELInE내에서 THiGERLLM의 대리모델인 것이다. 
{: .prompt-warning }

> **negative 데이터도 함께 학습하는 이유**
>
> 어떤 triple에 높은 점수를 주고 어떤 triple에 낮은 점수를 주는지 보기 위해. 전체적인 점수 체계를 모사하기 위해.   
> 틀렸다고 외우게 하는 데이터가 아닌 낮거나 다양한 점수를 주는 것을 배운다. 
{: .prompt-warning }

 **Train GraphSAGE $f_\phi$ to mimic the scores of THiGERLLM**  
$f_ϕ​(s,r,o,G)≈f_θ​(s,r,o)$  
![Pasted image 20260917165510](/assets/img/pasted-image-20260917165510.png)  
으로 MSE로 THiGERLLM score을 따라 하도록 GraphSAGE를 학습시킨다. 

> **왜 설명 모델을 따로 두는 것인가?**
>
> 후보 explanation path가 수백, 수천개일 수 있고   
> 이 path 삭제한 경우, 저 path만 있는 경우 등 모두 평가해야하기 때문이다.   
> 매번 THiGERLLM을 retraining하면 너무 비싸다.   
> 그래서 작은 GraphSAGE를 THiGERLLM을 흉내내는 대리 모델로 만든 것이다. 
{: .prompt-info }

PHELInE는 매번 거대한 THiGERLLM을 직접 뜯어보지 않는다.   
여러 triple을 THiGERLLM을 넣는다.   
각각에 THiGERLLM이 점수를 준다고 할때 이 점수들을 training target으로 삼아 graphSAGE surrogate model을 학습한다. 

GraphSAGE에게 생물학적으로 맞는 답을 직접 맞히라고 가르치는 것이 아니라 THiGERLLM이 주는 점수를 최대한 똑같이 하라고 학습시키는 것이다. 


**$f_{ϕ^∗}$​ frozen surrogate**  
GraphSAGE 학습이 끝나면 더이상 학습시키지 않는다. 

이후 hypothesis가   
몇개 들어와도 똑같은 surrogate를 재사용한다. 

> **surrogate?**
>
> 대리모델이라는 뜻으로 원래 모델이 너무 크거나 느리거나 비싸서 직접 계속 쓰기 어려울때 원래 모델의 출력을 최대한 비슷하게 다라 하도록 만든 더 단순한 모델을 말한다. 
{: .prompt-warning }



### **ONLINE**

여기는 새 hypothesis가 들어올때마다 수행한다.   
Knoewledge graph에서 candidate explanation 찾기  
이제 실제로 하나의 새 hypothesis를 설명하는 것이다. 

1. **Candidate generation**
s->o를 연결하는 기존 사실들의 path를 찾는다.   
둘 사이에 직적 관계가 아직 문헌에서 안알려졌다고 할때

KG는 대신 다른 candiate E_1 점수가 있을 것이다. 

> **KG?**
>
> Knowledge Graph  
> 개념들을 점으로 놓고 그 사이의 관계를 선으로 연결한 데이터 구조이다.   
> TP53, Gene A, Cancer 그런게 node/entity가 되는 것이고  
> regulates, associated with, increase expression of 이런게 edge/relation이 되는 것이다. 
{: .prompt-warning }

A--B  
B--C

A--C

여러 경로를 만들어서 PHELInE는 subject와 object를 연결하는 relational path들을 eumerate한다. 

기존 KG에는 이 직접 relation이 없지만 둘 사이를 연결하는 간접 path들이 있을 수 있다. 



2. **Query $f_{\phi^*}$ — no retraining**
각 candidate를 surrogate에 넣는다.   
각 $E_i$ 에 대해 이 path만 남기는 것과 이 path를 전체 KG에서 없애는 작업을 한다.   
이 경로 하나만으로 prediction을 설명할 수 있는지 이 경로가 없어지면 prediction이 약해지는지 보는 것이다. 


![Pasted image 20260929165435](/assets/img/pasted-image-20260929165435.png)

각각 sufficiency와 necessity이다. 

Sufficiency: 이 경로로 충분한가  
경로만 남기고 surrogate에게 다시 묻는다. 점수가 높다면 그 path자체가 hypothesis를 설명하기 충분하다고 본다. 


A--B  
B--C

A--D  
D--C

A-B-C  
A-D-C

TP53  
TP43



Neccesity: 이 경로가 꼭 필요한가  
반대로 경로를 삭제해서 prediction score를 다시 구한다. 없앴더니 점수가 많이 떨어진다면 필요한 path라는 의미이다. 

모든 candidate path에 대해 score를 계산한다. 


3. **Influence scoring & reranking**
점수가 높은 path가 hypothesis에 훨씬 큰 영향을 미치는 explanation 후보가 된다. 

rank by influence ; pick top diverse paths

각 paht영향력을 비교해서 높은 것부터 정렬한다.   
top5가 모두 같은 것을 보여줄 수도 있다.   
PHELInE는 intermediate entity와 relation이 다른 path를 선호하도록 reranking한다.   
논문에서 different intermediate entities를 포함하는 distinct relational paths를 우선한다. 다양한 reasoning patterns를 제공한다. 

THiGERLLM이 무슨 관계가 있을지 예측하고 PHELInE는 왜 그런 예측을 했을지 기존 지식 경로로 설명한다. 


> **sufficiency점수와 necessity점수를 둘다 같이 ranking을 하는 것인지.**
>
> 아니다. 각각 candidate explanation $E_i$ 에 대해 두 종류 점수를 계산한다. 하나를 선택하거나 둘을 결합해서 사용한다. 어떻게 결합하는지 고정 공식은 아니다. 선택된 function을 정의하고 그 점수로 top k 를 뽑은 다음 reranking한다.   
> top k 를 뽑고 비슷한 path가 몰리지 않도록 diversity reranking을 해서 최종 explanation 하는 것이다. 
{: .prompt-warning }

> **diverse reranking 방식?**
>
> 정확한 공식이 본문에 쓰여있지는 않지만  
> A-B-C될때 이 B가 모두 같은 계열로 뽑힐 수도 있다.   
> B가 다른 결과를 선호하기 때문에 점수가 조금 낮더라고 서로 다른 reasoning path를 보여주는 것이다.   
> 설명 공간의 coverage를 넓히는 것이다.   
> explanation 이 편향돼 보이고 다른 가능한 매커니즘을 놓칠 수 있어서 서로 다른 reasoning route를 일부러 섞어 보여주는 것이다. 
{: .prompt-warning }


> **형태가 (s, r, o) 이어야하는 것인가?**
>
> 그렇다.   
> s가 subject/head entity  
> r이 relation  
> o가 object/tail entity이다
{: .prompt-warning }

![Pasted image 20260929165454](/assets/img/pasted-image-20260929165454.png)  
가장 단순한 방법은 후보 경로를 하나씩 제거해서   
그때마다 THiGERLLM을 다시 학습해서 예측 점수가 얼마나 변하는지 확인한다  
하지만 이렇게 하면 경로마다 전체 모델을 재학습해야하기 때문에 계산 비용이 너무 크다  
그래서 PHELInE는 먼저 THiGERLLM이 여러 positive, negative triple에 대해 출력한 prediction score를 수집한다. 

그리고 이 점수를 따라 하도록 더 가벼운 GraphSAGE surrogate model을 학습한다. 

이 surrogate는 한번 학습한 뒤에는 weight를 고정하고 이후 여러 후보 경로를 평가할때 반복해서 사용한다. 

각 후보 path에 대해 graph에서 해당 path를 포함하거나 제거한 뒤 surrogate의 prediction score가 얼마나 달라지는지 확인한다  
점수 변화가 클수록 해당 path가 원래 예측에 더 큰 영향을 준다고 보고 더 높은 순위로 평가한다. 

THiGERLLM의 내부 weight에 직접 접근할 필요 없이 prediction score만 있으면 된다. 특정 모델 구조에 강하게 의존하지 않는 model-agnostic한 방식으로 사용할 수 있다.

비슷한 경로만 반복해서 보여주는 것을 막기 위해 diversity ranking도 적용한다.

surrogate가 THiGERLLM 자체는 아니기 때문에 approximation error가 발생할 수 있다.   
surrogate가 원래 모델을 얼마나 잘 모사하는지 별도 확인할 필요가 있다. 



## 4. Evaluation in Biomedical Science

### 4.1 Biomedical Data

![Pasted image 20260929165511](/assets/img/pasted-image-20260929165511.png)  
미래에 새로 발견된 관계를 제대로 평가하려면 학습 시점과 평가 시점을 시간으로 분리해야한다.

논문에서는 2020년을 기준으로 둔다.  
과거 데이터로 학습하고 미래에 나타난 관계를 맞혔는지 보는 방식이다. 

평가에서 positive라고 부르는 것은 실제로 생물학적으로 참인 관계라기보다 2020년 이후 문헌에 실제로 보고된 관계라는 뜻이다.

2020년 이후에도 문헌에 안 나온 relation이라고 해서 반드시 false는 아니다


![Pasted image 20260929165528](/assets/img/pasted-image-20260929165528.png)




어떤 데이터로 hakken을 만들고 평가했는지 설명한다.   
PMC open access의 full text, MEDLINE의 title/abstract 그리고 상업 라이선스 데이터셋으로부터  
처음에 약 2억 4933만개 raw triple이 있었고 cleaning후 최종적으로 

- 254,806 entities
- 7,127,960 triples
- 23 relation types
- 20 macro-domains
가 남는다.   
각 triple에는 그 관계가 처음 문헌에 연도 timestamp도 붙어있어서 temporal KG를 만들 수 있다. 


> **openaccess라고 모두 가능한게 아닌 것으로 안다. 어떻게 했는지**
>
> PMC open access의 full text 랑 MEDLINE 의 title/abstract , select commercailly licensed data 를 모았다고만 나와있다. 
{: .prompt-warning }

> **어떻게 cleaning한 것인지**
>
> 1. preliminary expert analysis
>   biomedical 전문가들이 raw dataset의 type들을 검토했다. 얼마나 관련있는지 보고 관계를 우선순위화한다. 같은 의미인데 표현이나 방향이 다른 관계를 하나의 schema로 통일한다. 
> 2. domain search
>   각 entity에 ontology hierarchy가 있으니 그 ancestry를 계속 위로 따라가서 가장 상위의 biological domain을 찾는다. 
> 3. edge cleaning
>   실제 triple을 크게 줄이는 단계이다.   
>   exact duplicate를 제거한다.   
>   동일한 triple이 여러 시점에 있으면 가장 이른 timestamp만 유지한다.   
>   invalid entry   
>   극도로 드문 relation 제거한다.   
>   contradictory/inconsistent relation 정리한다. 
> 4. node cleaning
>   domain이 없거나 inconsistent한 entity에는 데이터에 같이 들어있던 role tag를 이용해서 valid domain을 다시 할당한다. 
{: .prompt-warning }

### 4.2 Model Benchmarking

THiGERLLM 성능 비교  
한 시점을 cutoff로 잡아서   
진짜 미래 예측처럼 temporal split을 쓴다

THiGERLLM을  
random  
complEx  
kNN  
MLP  
Rule-based momdel  
tNodeEmbed  
기존 THiGER

을 비교한다.   
결과는 THiGERLLM이 모든 metric을 압도하는 것은 아니다. macro recall이 크게 좋아지고 macro F1도 약간 좋아지지만 weighted F1과 mean nDCG는 기존 THiGER가 더 높다. 


### 4.3 back-testing non historic data

모델이 정말 미래를 어느정도 예측하는지 아니면 split 하나에서만 잘 된 것인지

historical back-testing을 한다.   
>historical back testing

그리고 미래를 더 멀리 볼수록 recall은 떨어지지만 10념 horizon에서도 급격히 random 수준으로 무너지지 않고 recall감소가 대략 8% 정도로 보고된다.   
ranking metric인 mean nDCG도 비교적 안정적이다. 

현재 문헌 구조안에 몇년 뒤 등장할 관계를 예측할 수 있는 신호가 어느정도 존재한다. 

10년 뒤 문헌에 기록될 relation을 어느 정도 예측하는 것이다. 

![Pasted image 20261001105344](/assets/img/pasted-image-20261001105344.png)  
2020년까지의 지식만 학습했을때 2020년 이후 처음 등장한 relation type를 얼마나 잘 예측하는가를 여러 baseline으로 비교한 표이다. 

| 지표           | 의미                                  | 높을수록          |
| ------------ | ----------------------------------- | ------------- |
| `Prec_macro` | relation 종류별 precision을 계산한 뒤 평균    | 좋음            |
| `Rec_macro`  | relation 종류별 recall을 계산한 뒤 평균       | 좋음            |
| `F1_macro`   | macro precision/recall의 균형          | 좋음            |
| `Prec_w`     | relation 빈도를 고려한 weighted precision | 좋음            |
| `Rec_w`      | weighted recall                     | 좋음            |
| `F1_w`       | weighted F1                         | 좋음            |
| `nDCG_mean`  | 정답 relation을 얼마나 높은 순위에 올리는지        | 100에 가까울수록 좋음 |

THiGER과 THiGERLLM을 비교하면 된다. 새 모델이 THiGERLLM으로 이것이 미래에 등장하는 relation을 훨씬 더 많이 찾아냈다. 이걸 less frequent rare relation 에 대한 recall 향상으로 해석한다. 

각각 baseline 역할  
random: 랜덤 예측으로 하한선.  
ComplEx: 전통적인 KG embedding model  
KNN: 비슷한 entity pair의 relation 활용  
MLP: 일반적인 neural classifier  
RuleBased: node/neighbor relation pattern 기반  
TNodeEmbed: temporal/structural 모델  
THiGER: 기존 저자들의 temporal/structural 모델  
THiGERLLM: THiGER 계열에 textual/LLM semantic signal을 추가한 새 모델

baseline이 단순 모델부터 KG, temporal model, 이전 SOTA까지 단게적으로 들어가있다. 

THiGERLLM은 text/LLM 정보를 추가하면서 rare relation 을 포함한 전체 relation type에 대한 recall을 크게 높였지만, precision frequent-relation 성능 및 ranking에서는 기존 THiGER 보다 낮다. recall precision trade off가 나타난다. 






### 4.4 Wet lab validation

이전까지는 모두 retrospective  
>retrospective?

그래서 아직 문헌에 없는 relation 을 하나 만들어냈을 때 실제 생물학 실험에서도 잘 맞을지를 보게 된다. 


aging 관련 gene 목록을 전문가에게 받아서 1385개 entity를 대상으로 예측한다. 여기서 threshold넘긴 hypothesis가 154만개가 나온다. 

이 154만개를 실험할 수 없으니 여러단계로 줄인다. 


3개중에 2개가 실험적으로 지지된다.   
그렇지만 이 3개가 154만개에서 여러 filtering, 전문가 selection을 거쳐 고른 3개이기때문에 selection bias가 매우 크다. 

hakken 이 실제 실험으로 가져갈만한 novel hypothesis를 생성할 수 있다는 proof of concept를 보여준다. 정도로 이해가 맞다.

![Pasted image 20260918171813](/assets/img/pasted-image-20260918171813.png)

Fig.3:  
과거 시점까지만 학습한 hakken이 그 뒤 2년, 4년, 6년, 8년, 10년 뒤에 실제로 문헌에 나타나는 relation을 얼마나 잘 미리 맞추는지를 보는 것이다. 

세로로 각각 1990, 2000, 2010 이다.   
또 가로로 각각 micro recall, macro recall, mean nDCG를 보여준다. 

x축은 years after training cutoff는 cutoff이후 몇년 뒤의 미래 relation을 보는지다.   
2년 간격의 disjoint bin으로 나눈다.   
cutoff가 2000년이면 x=2는 2000-2002년에 처음 문헌에 등장한 관계를 보는 것.  
각 구간이 따로 떨어진 interval이다. 

**파란선**인 실제 모델의 성능이고  
**주황선**은 random empirical scores baseline  
주황선이 단순 랜덤은 아니고 THiGERLLM 모델이 내는 score 분포 자체는 유지하고 그 score 를 어떤 relation을 줄지만 섞는 것이다.  
점수 자체는 유지하면서   
입력 정보랑 relation 사이 실제 패턴을 제거했을때 나오는 성능

원래 THiGERLLM이

- TP53–BAMBI, `affects_expression_of` → 0.9
- RAF1–TNF, `decreases_expression_of` → 0.8
- A–B, `treats` → 0.2

라고 냈다면, 주황선은 점수 0.9, 0.8, 0.2 자체는 유지하되

- TP53–BAMBI → 0.2
- RAF1–TNF → 0.9
- A–B → 0.8

처럼 **어떤 관계에 어떤 점수가 붙는지 랜덤하게 바꾸는 방식**



**위에 3개는 micro recall이다.**   
모든 relation instance를 한꺼번에 놓고 계산한 recall이다.   
전체 관계를 다 합쳐서 실제로 미래에 등장한 관계들 중 몇 %를 미리 맞췄는지 보는 것이다

시간이 멀어질수록 내려간다.   
미래가 멀수록 에측이 어려워진다.   
micro recall은 frequent relation type이 결과를 많이 좌우할 수 있다. 그래서 저자들도 micro curve만으로는 temporal effect와 dataset composition effect를 분리하기 어렵다고 한다. 


**macro recall**  
relation 종류별로 따로 recall을 구한다음 평균내는 것이다.   
흔한 relation만 많이 맞춘 건지 희귀도 고르게 맞추는지 본다  
종류 전반에서, rare/low support relation을 놓치지 않고 찾아낸다.

**nDCG**  
모델이 유망하다고 위에 올린 hypothesis 들이 실제로 나중에 발견된 관계와 잘 맞는지  
시간이 지나도 모델의 우선순위 ranking은 안정적이다



너무 복잡다.  
우선 정리하면  
미래가 멀수록 recall은 감소한다.   
10년 hoizon 에서도 random baseline보다 높고 predictive signal이 남는다.   
1990, 2000, 2010 cutoff에서 비슷한 패턴을 보여 temporal behavior가 비교적 안정적이다. 






![Pasted image 20260922145652](/assets/img/pasted-image-20260922145652.png)

![Pasted image 20260929165552](/assets/img/pasted-image-20260929165552.png)  
Hakken이 예측한 RAF1 decreases expression of TNF hypothesis를 wet lab에서 시험한 것이다.   
RAF1을 inhibitor로 막았을때 TNF가 어떻게 변하는지를 본다

RAF1이 평소 TNF를 낮추고 있으므로 RAF1을 억제했을때 TNF가 올라가는 현상을 기대할 수 있다


그래프 y축을 본다. +되면 TNF는 증가한다. 0이면 TNF 변화가 없다. 아래 -는 TNF가 감소한다. 


맨 밑에 x축에 +는 약물 농도가 높아진다는 표시이다. 

> **LPS?**
>
> TNF를 증가시키는 물질로 positive control이다. 
{: .prompt-info }

RAF1을 막는 약을 두개 사용했다.   
첫번째가 GW5074이다. 

RAF를 억제했을때 TNF가 증가한다가 나와야한다.   
AZ628은 감소가 나왔다.   
가설과 반대가 된다.   
다른 실험으로 추가 검증을 증가하게 된다. 

> **왜 추가검증을 하는지. 이미 반대된거 아닌지**
>
> 간접적으로 한 것이기 때문이다. 가설이 틀려서인지 약물 특이 효과때문인지 구분하기 위함  
> 그런데? 그것도 약물로 한것. 다른 약으로 한번더 확인한거지 약물 confound를 완전히 제거한 독립 직접 검증이 아니다. 
{: .prompt-warning }

![Pasted image 20260922145702](/assets/img/pasted-image-20260922145702.png)  
![Pasted image 20260929165604](/assets/img/pasted-image-20260929165604.png)


TP53이 BAMBI 발현에 영향을 준다는 Hakken 가설을 실험한 것이다. 

Nutlin이라는 약으로 P53단백질을 증가시킨 뒤에 BAMBI mRNA가 변하는지 확인한다. 

y축에 특별한 처리하지 않은 vehicle control 과 비교해서 BAMBI 발현이 몇% 변했는지이다.   
같은 약물 조건으로 서로 다른 cell density에서 시험한다. 


Nutlin을 먼저보면 낮은 농도에서는 결과가 별로 일정하지 않다. 어떤 cell density는 감소하고 어떤건 거의 변화가 없다.   
농도를 높이면 BAMBI가 올라간다. 

둘 사이에 functional relationahip이 있을 가능성을 본다. 

맨 오른쪽 것은   
positive control이다.   
발현을 유도할 수 있다는 것은 알려진 사실이다.   
그래서 정말 증가할 수 있는 실험인지 확인하기 위해 넣는다. 

별은 위와 마찬가지로 통계적으로 유의한지를 보여준다. 

그래서 세 cell density에서 모두 BAMBI 증가가 관찰된다.   
하지만 매우 깔끔하지는 않다. 낮은 nutlin농도에서 오히려 감소하는 경우도 있고 증가폭도 엄청 크진않다.   
그래서 관계가 약하고 조건 의존적일 수 있다고 적혀있다. 




![Pasted image 20260922145535](/assets/img/pasted-image-20260922145535.png)  
Hakken 이 많은 후보 중에서 실제 wet lab 검증 대상으로 최종 선택한 3개의 hypothesis를 정리한 표이다. 

E1, E2 가 관계를 이루는 두 entity이다  
relation이 hakken이 예측한 관계 종류  
recency E1, E2는  각 entity관련 문헌이 얼마나 최근에 활발했는지 보는 보조 지표  
MPL은 minimum path length, 기존 KG에서 두 entity 사이의 최단 경로 길이다  
confidence는 hakken이 그 hypothesis에 부여한 confidence score

각각 SOAT1이 STAT3의 transcriptional activity에 영향을 준다  
RAF1이 TNF expression을 감소시킨다  
TP53가 BAMBI expression에 영향을 준다  
가설이다. 


이 세개는 150만개가 넘는 predicted hypothesis에서 여러 filtering을 거쳐 추린 후보들 중 biomedical expert가 confidence 0.8 이상인 것들을 중심으로 실제 실험 대상으로 선택한 것이다. 

recency는 각 entity에 대해 그 node와 연결된 publication year들의 median을 recency score로 사용했다고 한다.   
처음 발견된 연도가 아닌 그 entity가 최근 문헌에서 얼마나 다루고 있는지 보는 보조 지표이다. 




![Pasted image 20260922150344](/assets/img/pasted-image-20260922150344.png)



첫번째 줄을 보면

SOAT -> STAT3  
SOAT1이 STAT3의 전사활성에 영향을 준다  
검증 결과 denied  
실험 결과 이 가설을 뒷받침하는 결과가 나오지 않았다는 의미다

RAF1 -> TNF  
결과 supported 로 기능적 관계를 지지했다는 뜻이다

TP53 -> BAMBI  
이것도 지지하는 결과가 나왔다는 뜻이다.

그래서 여기서 말하고자 하는 것은 THiGERLLM의 high-confidence hypothesis가 전부 맞는 건 아니지만 실제 실험으로 검증 가능한 새 가설을 만들어냈고 일부는 wet-lab에서 지지됐다를 보인다



![Pasted image 20260929165614](/assets/img/pasted-image-20260929165614.png)
- THiGERLLM이 graph 정보만 보는 게 아니라 text/LLM 정보를 같이 쓰면서 macro recall이 올라갔고, 특히 데이터가 적은 rare relation들을 더 잘 잡았다
- fig3에서 과거 시점까지만 학습시켰을때 장기적으로도 유망한 hypothesis를 위쪽에 올리는 ranking능력은 꽤 좋다
- 예측 뒷받침하는 candidate path를 같이 보여준다. 연구자가 모든 예측을 직접 하는 것이 아닌 어떤 예측부터 실험할지 우선순위를 정할때 참고가능하다

아직 해결되지 않은 문제로
- 정답 데이터 자체가 미래에 계속 추가되기 때문에 완전하지 않다
- 라이선스 데이터가 섞여있어 같은 데이터 조건으로 재현하기 어렵다
- 관계 있다는 증거를 얻었지만 어떤 경로로 어떻게 변화시킨다까지는 더 실험이 필요하다

놓치는 후보를 줄이면서 유망한 hypothesis를 우선순위화해주고 최종 검증은 사람이 하는 시스템이다. 


> **왜 GraphSAGE는 되고 THiGERLLM은 안 되냐**
>
{: .prompt-warning }
GraphSAGE가 특별해서 되는 게 아니야.  
**GraphSAGE는 그냥 싸고 graph를 직접 입력으로 받는 작은 대체 모델이라서 쓰는 거야.**

생각해보자.  
원래 THiGERLLM이 이런 점수를 냈어.
```
A --- X --- B
```

이 graph를 보고

```
A와 B의 관계 점수 = 0.9
```

라고 했다고 하자.

이제 `A-X-B` 경로가 중요한지 알고 싶어.

그럼 가장 정직한 방법은:

```
A-X-B 경로를 없앤 graph를 만든다
↓
THiGERLLM을 그 graph로 다시 학습한다
↓
점수 다시 본다
```

문제는 **THiGERLLM 재학습이 너무 비싸다**는 거야.

그래서 저자들이 하는 건 이거야.

```
THiGERLLM의 점수들을 많이 모음
↓
GraphSAGE가 그 점수들을 흉내내도록 학습
↓
GraphSAGE 고정
```

이제 경로를 지우고 GraphSAGE에 넣어봐.

```
원래 graph → GraphSAGE → 0.88
경로 제거 graph → GraphSAGE → 0.35
```

그러면 저자들은

> “아, 진짜 THiGERLLM을 다시 학습했어도 점수가 많이 떨어졌을 가능성이 있겠구나”

라고 **근사해서 쓰는 거야.**

여기서 제일 중요한 점:

**GraphSAGE가 실제 재학습을 한 건 아니야.**  
그냥 **graph가 바뀌면 출력도 바뀌는 작은 모델**이라서, 그 변화량을 proxy로 쓰는 거야.

그래서 네 말대로

> “그럼 THiGERLLM 자체에 graph를 바꿔 넣으면 되잖아?”

이것도 가능해.

다만 그 경우는:

```
같은 THiGERLLM weight
+
다른 graph input
```

일 뿐이야.

즉  
**“지금 이 모델이 경로 제거에 얼마나 민감한가”**  
를 보는 거고,

PHELInE가 노리는 건  
**“학습 데이터에서 그 경로가 없었다면 모델이 어떻게 달라졌을까”**  
를 싸게 흉내내는 거야.

비유 하나만 할게.

학생이 어떤 책으로 공부해서 시험 90점을 받았어.

어떤 문단 X가 중요했는지 알고 싶어.

방법 1:  
시험 볼 때 문단 X만 가리고 다시 시험봄.  
→ 이미 공부한 기억은 남아있음.

방법 2:  
처음부터 문단 X 없는 책으로 다시 공부시킴.  
→ 진짜로 X 없이 배운 결과.

PHELInE가 궁금한 건 2번이야.  
근데 학생을 매번 다시 공부시키기 너무 힘드니까,  
**학생을 흉내내는 작은 모형(GraphSAGE)** 을 만들어서 빠르게 실험하는 거야.


**GraphSAGE는 “재학습을 실제로 하는 모델”이 아니라, “재학습했을 때 생길 변화”를 싸게 대신 추정하려고 쓰는 대리모델이야.**









