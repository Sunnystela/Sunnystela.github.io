---
type: literature
title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
author: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
year: "2021"
citekey: lewisRetrievalAugmentedGenerationKnowledgeIntensive2021
status: reading
tags:
  - literature
source: http://arxiv.org/abs/2005.11401
accept: NeurIPS 2020 (Advances in Neural Information Processing Systems 33); arXiv:2005.11401v4 last revised 2021-04-12
date: 2026-09-17 17:23:16 +0200
---


> **공식 링크 / 확인**
>
> - NeurIPS Proceedings: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
> - arXiv: [2005.11401](https://arxiv.org/abs/2005.11401)
> - PDF: [arXiv PDF](https://arxiv.org/pdf/2005.11401)
> - Implementation/Models: [Hugging Face Transformers RAG](https://huggingface.co/docs/transformers/model_doc/rag)
> - title/author 확인: NeurIPS proceedings와 arXiv v4의 title/author가 현재 노트의 title/author와 일치한다.
> - venue 확인: NeurIPS proceedings는 `Advances in Neural Information Processing Systems 33 (NeurIPS 2020)`로 등재하고, arXiv comments도 `Accepted at NeurIPS 2020`로 표기한다.
{: .prompt-info }

### 관련 프로젝트 : C. AI Agent, RAG

> **PDF**
>
{: .prompt-tip }

> [Preprint PDF](zotero://select/library/items/262CHWGB)



> **서지정보**
>
{: .prompt-info }

> Lewis, Patrick, Ethan Perez, Aleksandra Piktus, 기타. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”. arXiv:2005.11401. Preprint, arXiv, 2021년 4월 12일. [https://doi.org/10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401).

  

> **초록(Abstract)**
>
{: .prompt-info }

> Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) -- models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.

  

## 한줄 요약

RAG는 BART 같은 parametric seq2seq generator에 DPR 기반 Wikipedia vector index라는 non-parametric memory를 결합해, knowledge-intensive task에서 답변 품질과 provenance, 지식 갱신성을 함께 개선하려는 원 논문이다.

## 하이라이트 & 내 메모

![Pasted image 20260902225444](/assets/img/pasted-image-20260902225444.png)


RAG는 먼저 사용자의 입력 질문 x를 Query Encoder에 넣어 질문의 의미를 나타내는 벡터 q(x)로 변환한다. 한편 문서들은 미리 각각 임베딩 d(z)로 변환되어 Document Index에 저장되어 있다. 이후 Retriever는 질문 벡터 q(x)와 문서 벡터 d(z)의 유사도를 비교하며, MIPS(Maximum Inner Product Search)를 사용해 질문과 가장 관련성이 높은 Top-k 문서 z1, z2, ...를 찾는다. 이때 Retriever p_eta는 각 문서가 현재 질문과 얼마나 관련 있는지를 $p_eta(z|x)$라는 점수 또는 확률로 나타낸다.

검색된 문서들은 질문 x와 함께 Generator p_theta에 전달된다. Generator는 각 검색 문서 z를 참고하여 $p_theta(y|x,z)$, 즉 이 질문과 문서를 바탕으로 특정 답변 y를 생성할 가능성을 계산한다. 이후 Marginalization 단계에서는 Retriever의 문서 관련도와 Generator의 답변 생성 가능성을 함께 고려한다.

최종적으로 $p(y|x) = sum_z p_eta(z|x) × p_theta(y|x,z)$ 형태로 여러 검색 문서의 결과를 종합하여 가장 적절한 답변 y를 생성한다.

입력 질문 x → 질문 임베딩 q(x) → 관련 문서 검색 → Top-k 문서 선택 및 관련도 계산 → 질문과 검색 문서를 Generator에 입력 → 문서별 생성 결과 종합 → 최종 답변 y 생성 순서이다.





> **주장**
>
{: .prompt-info }

## 핵심 주장 / 방법론

핵심 주장은 LLM/seq2seq model이 factual knowledge를 parameter 안에 저장할 수는 있지만, 그 지식을 정확히 수정하거나 출처를 제시하거나 최신 정보로 교체하기 어렵다는 것이다. 그래서 저자들은 parametric memory와 non-parametric memory를 결합한 generation model을 제안한다.

구조는 다음과 같다.

- parametric memory: pre-trained seq2seq model인 BART-large.
- non-parametric memory: Wikipedia 2018 dump를 100-word chunk로 나눈 21M document index.
- retriever: DPR bi-encoder query encoder + fixed document encoder/index.
- search: Maximum Inner Product Search (MIPS)로 top-k document를 검색.
- generator: input query와 retrieved passage를 함께 condition으로 받아 output sequence를 생성.
- training: retrieved document를 latent variable로 보고, top-k 문서에 대한 generator likelihood를 marginalize하며 end-to-end fine-tuning한다.

두 모델이 핵심이다.

- RAG-Sequence: 하나의 retrieved document가 전체 output sequence 생성을 책임진다고 가정한다.
- RAG-Token: token마다 다른 retrieved document를 참고할 수 있게 marginalization한다.

## 내 생각 / 질문 / 반박

C. AI Agent, RAG 프로젝트 관점에서 이 논문은 "논문을 왜 query와 매칭해 저장하는가"에 대한 가장 기본적인 답을 준다. RAG는 모든 논문을 모델에 외우게 하는 것이 아니라, 질문이 들어왔을 때 관련 passage를 찾아 generator가 그 passage를 조건으로 답하게 만드는 구조다.

다만 지금 만들려는 paper agent는 원 논문보다 더 까다롭다. Wikipedia open-domain QA에서는 retrieved passage가 대체로 짧은 factual evidence지만, 논문 QA에서는 method, table, figure caption, experimental condition, limitation이 모두 evidence가 될 수 있다. 따라서 단순 100-word chunk와 dense retrieval만으로는 CardioTox 논문 질문을 충분히 처리하기 어렵다.

내 질문은 다음이다.

- 논문 agent에서 `document` latent variable은 paper chunk인가, figure/table 단위인가, 아니면 claim-evidence pair인가?
- RAG-Token처럼 token마다 다른 근거를 섞는 구조가 긴 과학 답변에서는 citation 혼합 오류를 키우지 않을까?
- domain-specific RAG에서 retriever를 end-to-end로 fine-tune할 때, 어떤 질문/근거 supervision이 필요한가?

## 저자가 정말로 증명한 게 뭐고 그냥 주장만 한 게 뭔가?

증명에 가까운 것:

- RAG가 Natural Questions, WebQuestions, CuratedTrec 등 open-domain QA에서 parametric-only baseline과 기존 retrieve-and-extract 방식보다 강한 성능을 보였다.
- MS-MARCO, Jeopardy Question Generation, FEVER에서도 BART baseline보다 factual/specific generation 또는 classification 성능이 개선됐다.
- RAG-Sequence와 RAG-Token이라는 두 marginalization 방식이 task에 따라 다르게 작동한다.
- non-parametric memory index를 교체하면 world knowledge를 업데이트할 수 있음을 world leader query 실험으로 보였다.

주장에 가까운 것:

- RAG가 hallucination을 줄인다는 결론은 이 실험들에서는 설득력 있지만, 모든 domain-specific scientific QA에 자동으로 일반화되지는 않는다.
- retrieved document가 provenance를 제공한다는 말은 retrieval evidence가 실제 answer claim을 지지할 때만 성립한다.
- end-to-end fine-tuning이 항상 필요한지는 이후 modular/production RAG에서는 별도 판단이 필요하다.

## 이 결과가 놀랍나, 아니면 예상대로인가?

외부 Wikipedia를 붙이면 open-domain QA가 좋아지는 것은 어느 정도 예상 가능하다. 놀라운 지점은 extractive reader 없이 generative seq2seq 모델이 retrieved evidence를 marginalize해서 QA, generation, fact verification을 하나의 recipe로 처리했다는 점이다.

특히 RAG-Token의 Figure 2는 중요하다. generator가 "The Sun Also Rises"와 "A Farewell to Arms"를 생성할 때 서로 다른 document posterior를 높게 쓰는 장면은, generation이 단일 passage 복사가 아니라 parametric memory와 retrieved memory의 협업이라는 점을 보여준다.

## 저자의 결론이 실험 결과보다 과하게 일반화되진 않았나?

조금 과하다. RAG가 provenance를 제공한다고 하지만, 논문이 보여준 것은 top-k document posterior와 FEVER evidence overlap이지, claim-level citation correctness 전체는 아니다.

또한 Wikipedia 기반 benchmark에서는 knowledge source가 비교적 정제되어 있지만, paper agent의 corpus는 PDF, table, appendix, conflicting studies, outdated guideline이 섞인다. 이 환경에서는 RAG 자체보다 corpus curation, chunking, reranking, citation verification이 더 큰 병목이 될 수 있다.

> **방법론**
>
{: .prompt-info }

## 이 방법이 왜 작동하는가, 그냥 "잘 됐다"가 아니라 원리적으로?

RAG는 knowledge storage와 language generation을 분리하기 때문에 작동한다. language model은 fluent generation과 task adaptation을 담당하고, factual knowledge는 외부 index에서 가져온다. 이렇게 하면 지식을 업데이트할 때 model weight를 다시 학습하지 않고 index를 교체할 수 있다.

Figure 흐름은 다음과 같다.

1. Figure 1은 전체 모델 구조다. input query `x`가 query encoder를 거쳐 vector가 되고, document index에서 MIPS로 top-k document `z_i`를 찾는다. generator는 `x`와 각 `z_i`를 함께 받아 output `y`를 만들고, document를 latent variable로 marginalize한다.
2. Figure 1 안의 왼쪽 예시는 task별 입력/출력 형태를 보여준다. question answering은 question을 query로 넣고 answer를 생성하며, fact verification은 claim을 query로 넣고 label을 생성하고, Jeopardy generation은 answer entity를 query로 넣고 question을 생성한다.
3. RAG-Sequence는 retrieved document 하나가 전체 sequence를 설명한다고 보는 구조다. 긴 답변 전체가 하나의 핵심 passage에 의존하는 QA에 잘 맞는다.
4. RAG-Token은 token마다 document posterior를 다시 섞는다. 여러 passage의 정보를 조합해야 하는 generation에서 유리하지만, 근거가 섞여 citation 추적은 어려워질 수 있다.
5. Figure 2는 RAG-Token의 document posterior를 token별로 보여준다. Hemingway 입력에서 특정 책 제목을 생성할 때 해당 정보를 가진 document posterior가 올라가며, 이후에는 parametric memory가 title completion을 이어간다.
6. Figure 3은 test-time retrieved document 수 `K`의 효과를 보여준다. 더 많이 검색하면 recall은 올라가지만, RAG-Token은 일정 지점 이후 성능이 peak를 지나며, retrieval quantity가 항상 generation quality로 이어지지는 않는다.

## 숨겨진 가정이 뭔가?


- Wikipedia index 안에 benchmark 질문의 정답 근거가 충분히 들어 있다는 가정.
- 100-word chunk가 QA/generation에 필요한 evidence 단위로 충분하다는 가정.
- DPR retriever가 Natural Questions/TriviaQA supervision으로 학습됐기 때문에 다른 knowledge-intensive task에도 잘 전이된다는 가정.
- fixed document encoder/index를 유지해도 query encoder와 generator fine-tuning만으로 충분하다는 가정.
- retrieved document posterior가 provenance로 해석 가능하다는 가정.
- generated answer가 retrieved evidence와 parametric memory를 섞을 때, 잘못된 합성이 크게 늘지 않는다는 가정.

## 실험 설정


- baseline이 공정한가?

- Knowledge source: December 2018 Wikipedia dump.
- Index: Wikipedia articles를 disjoint 100-word chunks로 나눠 약 21M documents 구성.
- Retriever: DPR bi-encoder, MIPS/FAISS index, top-k retrieval.
- Generator: BART-large seq2seq model.
- Training: document encoder와 index는 고정하고, query encoder와 BART generator를 fine-tune.
- Tasks: open-domain QA(Natural Questions, TriviaQA, WebQuestions, CuratedTrec), abstractive QA(MS-MARCO), Jeopardy Question Generation, FEVER fact verification.
- Metrics: EM, BLEU-1, Q-BLEU-1, ROUGE-L, label accuracy, human factuality/specificity preference, evidence overlap, diversity.
- Baselines: BART, T5 closed-book, REALM, DPR, task-specific retrieve-and-extract/pipeline systems, BM25/frozen retriever ablations.
- 공정성: 대체로 강한 baseline과 비교하지만 DPR retriever initialization이 QA supervision을 이미 포함하므로, 완전히 retrieval-supervision-free 구조로 읽으면 안 된다. FEVER에서는 gold evidence supervision을 쓰지 않는 점이 장점이다.

> **한계**
>
{: .prompt-info }

## 직접 명시한 limitation 

Broader Impact에서 저자들은 external knowledge source가 완전히 사실적이거나 bias-free일 수 없다고 지적한다. RAG가 Wikipedia에 기반하면 Wikipedia의 오류와 편향도 함께 가져온다.

또한 RAG는 GPT 계열 language model과 유사한 misuse risk가 있다. 더 factual해졌다고 해서 misleading content, impersonation, spam/phishing 생성 가능성이 사라지는 것은 아니다.

## 저자가 명시하지 않은, 추정되는 한계 

RAG의 provenance는 document-level posterior에 가깝고, claim-level evidence verification은 아니다. 논문 agent에서는 한 답변 안의 각 문장마다 근거 passage가 다를 수 있으므로, RAG 원 구조만으로는 citation correctness를 보장하기 어렵다.

또한 Wikipedia 100-word chunk는 비교적 단순한 factual QA에는 맞지만, 논문 PDF의 table, figure, method condition, endpoint definition은 chunk boundary가 틀어지면 의미가 쉽게 깨진다. C. AI Agent, RAG 프로젝트에서는 chunking이 retriever보다 먼저 망가질 수 있다.

## 이 방법이 실패하는 경우를 상상할 수 있나?

검색된 top-k passage에 정답 근거가 없으면 실패한다. RAG는 parametric memory로 일부 답을 맞힐 수 있지만, 그러면 provenance와 faithfulness가 약해진다.

검색된 passage가 맞더라도 generator가 다른 parametric knowledge와 섞어 잘못된 conclusion을 만들면 실패한다. 특히 논문 간 실험 조건이 다른데도 하나의 일반 결론처럼 합성하는 경우가 위험하다.

RAG-Token식 다중 passage 혼합은 답변 다양성과 specificity에는 도움이 될 수 있지만, scientific QA에서는 어떤 claim이 어떤 source에서 왔는지 추적하기 어렵게 만들 수 있다.
