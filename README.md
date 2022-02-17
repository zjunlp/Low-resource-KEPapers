# Low-resource Knowledge Extraction 

The repository is a paper set on low-resource knowledge extraction (NER, RE, EE), which is categorized into three paradigms. 

## Content
* [1 Exploiting Higher-resource Data](#1-Exploiting-Higher-resource-Data)
  * [1.1 Weakly Supervised Augmentation](#Weakly-Supervised-Augmentation)
  * [1.2 Multi-modal Augmentation](#Multi-modal-Augmentation)
  * [1.3 Multi-lingual Augmentation](#Multi-lingual-Augmentation)
  * [1.4 Auxiliary Knowledge Enhancement](#Auxiliary-Knowledge-Enhancement)
* [2 Exploiting Stronger Models](#2-Exploiting-Stronger-Models)
  * [2.1 Meta Learning](#Meta-Learning)
  * [2.2 Transfer Learning](#Transfer-Learning)
  * [2.3 Prompt Learning](#Prompt-Learning)
* [Exploiting Data and Models Together](#Exploiting-Data-and-Models-Together)
  * [3.1 Multi-task Learning](#Multi-task-Learning)
  * [3.2 Formulating KE as QA/MRC](#Formulating-KE-as-QA-and-MRC)
  * [3.3 Retrieval Augmentation](#Retrieval-Augmentation)

## 1 Exploiting Higher-resource Data


### Weakly Supervised Augmentation
* Distant Supervision for Relation Extraction without Labeled Data (ACL 2009) [[paper]](https://aclanthology.org/P09-1113.pdf)
* BOND: BERT-Assisted Open-Domain Named Entity Recognition with Distant Supervision (KDD 2020) [[paper]](https://dl.acm.org/doi/abs/10.1145/3394486.3403149)
* Automatically Labeled Data Generation for Large Scale Event Extraction (ACL 2017) [[paper]](https://aclanthology.org/P17-1038.pdf)
* Gradient Imitation Reinforcement Learning for Low Resource Relation Extraction (EMNLP 2021) [[paper]](https://aclanthology.org/2021.emnlp-main.216.pdf)


### Multi-modal Augmentation
* Visual Attention Model for Name Tagging in Multimodal Social Media (ACL 2018) [[paper]](https://aclanthology.org/P18-1185.pdf)
* Cross-media Structured Common Space for Multimedia Event Extraction (ACL 2020) [[paper]](https://aclanthology.org/2020.acl-main.230.pdf)
* Joint Multimedia Event Extraction from Video and Article (EMNLP 2021 finding) [[paper]](https://aclanthology.org/2021.findings-emnlp.8.pdf)
* Multimodal Relation Extraction with Efficient Graph Alignment (MM 2021) [[paper]](https://dl.acm.org/doi/10.1145/3474085.3476968)

### Multi-lingual Augmentation
* Neural Relation Extraction with Multi-lingual Attention (ACL 2017) [[paper]](https://aclanthology.org/P17-1004.pdf)
* Improving Low Resource Named Entity Recognition using Cross-lingual Knowledge Transfer (IJCAI 2018) [[paper]](https://www.ijcai.org/Proceedings/2018/0566.pdf)
* Event Detection via Gated Multilingual Attention Mechanism (AAAI 2018) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11919)
* Cross-lingual Transfer Learning for Relation Extraction Using Universal Dependencies (Computer Speech & Language, 2022) [[paper]](https://www.sciencedirect.com/science/article/pii/S0885230821000711)


### Auxiliary Knowledge Enhancement
#### (1) Text
* Improving Event Detection via Open-domain Trigger Knowledge (ACL 2020) [[paper]](https://aclanthology.org/2020.acl-main.522.pdf)
* MapRE: An Effective Semantic Mapping Approach for Low-resource Relation Extraction (EMNLP 2021) [[paper]](https://aclanthology.org/2021.emnlp-main.212.pdf)

#### (2) KG
* DOZEN: Cross-Domain Zero Shot Named Entity Recognition with Knowledge Graph (SIGIR 2021) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3404835.3463113)
* Leveraging FrameNet to Improve Automatic Event Detection (ACL 2016) [[paper]](https://aclanthology.org/P16-1201.pdf)


#### (3) Ontology & Rule
* Logic-guided Semantic Representation Learning for Zero-Shot Relation Classification (COLING 2020) [[paper]](https://aclanthology.org/2020.coling-main.265.pdf)
* OntoED: Low-resource Event Detection with Ontology Embedding (ACL 2021) [[paper]](https://aclanthology.org/2021.acl-long.220.pdf)
* Low-resource Extraction with Knowledge-aware Pairwise Prototype Learning (Knowledge-Based Systems, 2022) [[paper]](https://www.sciencedirect.com/science/article/pii/S0950705121008467)



## 2 Exploiting Stronger Models

### Meta Learning

#### For Low-resource NER
* MetaNER: Named Entity Recognition with Meta-Learning (WWW 2020) [[paper]](https://dl.acm.org/doi/10.1145/3366423.3380127)


#### For Low-resource RE
* Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs (ICML 2020) [[paper]](http://proceedings.mlr.press/v119/qu20a/qu20a.pdf)
* Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification (AAAI 2019) [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/4604)
* Enhanced Meta-Learning for Cross-Lingual Named Entity Recognition with Minimal Resources (AAAI 2020) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6466)
* Meta-Information Guided Meta-Learning for Few-Shot Relation Classification (COLING 2020)[[paper](https://aclanthology.org/2020.coling-main.140.pdf)]


#### For Low-resource EE
* Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection (WSDM 2020) [[paper]](https://dl.acm.org/doi/abs/10.1145/3336191.3371796)
* Adaptive Knowledge-Enhanced Bayesian Meta-Learning for Few-shot Event Detection (ACL 2021 finding) [[paper]](https://aclanthology.org/2021.findings-acl.214.pdf)
* Few-Shot Event Detection with Prototypical Amortized Conditional Random Field (ACL 2021 finding) [[paper](https://aclanthology.org/2021.findings-acl.3.pdf)]


### Transfer Learning

#### (1) Class-related Semantics
* Zero-Shot Transfer Learning for Event Extraction (ACL 2018) [[paper]](https://aclanthology.org/P18-1201.pdf)
* Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks (NAACL 2019) [[paper]](https://aclanthology.org/N19-1306.pdf)
* Relation Adversarial Network for Low Resource Knowledge Graph Completion (WWW 2020) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3366423.3380089)
* Graph Learning Regularization and Transfer Learning for Few-Shot Event Detection (SIGIR 2021) [[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463054)]

#### (2) Pre-trained Language Representations
* Matching the Blanks: Distributional Similarity for Relation Learning (ACL 2019) [[paper]](https://aclanthology.org/P19-1279.pdf)
* Exploring Pre-trained Language Models for Event Extraction and Generation (ACL 2019) [[paper]](https://aclanthology.org/P19-1522.pdf)
* Coarse-to-Fine Pre-training for Named Entity Recognition (EMNLP 2020) [[paper]](https://aclanthology.org/2020.emnlp-main.514.pdf)
* CLEVE: Contrastive Pre-training for Event Extraction (ACL 2021) [[paper]](https://aclanthology.org/2021.acl-long.491.pdf)
* Graph Learning Regularization and Transfer Learning for Few-Shot Event Detection (ACL 2021) [[paper](https://aclanthology.org/2021.acl-long.490.pdf)]

### Prompt Learning

#### (1) Vanilla Prompt Learning
* Template-Based Named Entity Recognition Using BART (ACL 2021) [[paper]](https://aclanthology.org/2021.findings-acl.161.pdf)
* LightNER: A Lightweight Generative Framework with Prompt-guided Attention for Low-resource NER (2021) [[paper]](https://arxiv.org/abs/2109.00720)


#### (2) Augmented Prompt Learning
* PTR: Prompt Tuning with Rules for Text Classification (2021) [[paper]](https://arxiv.org/abs/2105.11259)
* KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction (WWW 2022) [[paper]](https://arxiv.org/abs/2104.07650)
* OntoPrompt: Ontology-enhanced Prompt-tuning for Few-shot Learning (WWW 2022) [[paper]](https://arxiv.org/abs/2201.11332)



## 3 Exploiting Data and Models Together

### Multi-task Learning

#### (1) NER, named entity normalization
* A Neural Multi-Task Learning Framework to Jointly Model Medical Named Entity Recognition and Normalization (AAAI 2019) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/3861)
* MTAAL: Multi-Task Adversarial Active Learning for Medical Named Entity Recognition and Normalization (AAAI 2021) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17714)
* An End-to-End Progressive Multi-Task Learning Framework for Medical Named Entity Recognition and Normalization (ACL 2021) [[paper]](https://aclanthology.org/2021.acl-long.485.pdf)


#### (2) NER, RE
* GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction (ACL 2019) [[paper]](https://aclanthology.org/P19-1136.pdf)
* CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning (AAAI 2020) [[paper]](https://arxiv.org/abs/1911.10438)
* Joint Entity and Relation Extraction Model based on Rich Semantics (Neurocomputing, 2021) [[paper]](https://www.sciencedirect.com/science/article/pii/S0925231220319378?casa_token=jzgLW9J1UKoAAAAA:5vnqqGKt0_-ykbhTp15Bq8mB-8B50cM3LDa10q2h8yc4q4AJVfeEbQV_fyMo2Z92xjl3HPNt6w)


#### (3) NER, RE, EE
* Entity, Relation, and Event Extraction with Contextualized Span Representations (EMNLP 2019) [[paper]](https://aclanthology.org/D19-1585.pdf)


#### (4) Word Sense Disambiguation (WSD), Event Detection (ED)
* Similar but not the Same: Word Sense Disambiguation Improves Event Detection via Neural Representation Matching (EMNLP 2018) [[paper]](https://aclanthology.org/D18-1517.pdf)

### Formulating KE as QA and MRC
* A Unified MRC Framework for Named Entity Recognition (ACL 2020) [[paper]](https://aclanthology.org/2020.acl-main.519.pdf)
* Entity-Relation Extraction as Multi-Turn Question Answering (ACL 2019) [[paper]](http://aclanthology.lst.uni-saarland.de/P19-1129.pdf)
* Zero-Shot Relation Extraction via Reading Comprehension (CoNLL 2017) [[paper]](https://aclanthology.org/K17-1034.pdf)
* Event Extraction as Machine Reading Comprehension (EMNLP 2020) [[paper]](https://aclanthology.org/2020.emnlp-main.128.pdf)
* Event Extraction by Answering (Almost) Natural Questions (EMNLP 2020) [[paper]](https://aclanthology.org/2020.emnlp-main.49.pdf)
* Learning to Ask for Data-Efficient Event Argument Extraction (AAAI 2022, Student Abstract) [[paper]](https://arxiv.org/abs/2110.00479)

### Retrieval Augmentation

#### Retrieval-based Language Models
* Generalization through Memorization: Nearest Neighbor Language Models (ICLR 2020) [[paper]](https://openreview.net/forum?id=HklBjCEKvH)
* Retrieval Augmented Language Model Pre-Training (ICML 2020) [[paper]](http://proceedings.mlr.press/v119/guu20a.html)
* Adaptive Semiparametric Language Models (TACL, 2021) [[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00371/100688/Adaptive-Semiparametric-Language-Models)
* BERT-kNN: Adding a kNN Search Component to Pretrained Language Models for Better QA (EMNLP 2020 finding) [[paper]](https://aclanthology.org/2020.findings-emnlp.307.pdf)
* Efficient Nearest Neighbor Language Models (EMNLP 2021) [[paper]](https://aclanthology.org/2021.emnlp-main.461.pdf)
* End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering (NeurIPS 2021) [[paper]](https://openreview.net/forum?id=5KWmB6JePx)
* Improving Language Models by Retrieving from Trillions of Tokens (2021) [[paper]](https://arxiv.org/abs/2112.04426)


#### Few-shot Settings
* Few-shot Intent Classification and Slot Filling with Retrieved Examples (NAACL 2021) [[paper]](https://aclanthology.org/2021.naacl-main.59.pdf)
* KNN-BERT: Fine-Tuning Pre-Trained Models with KNN Classifier (2021) [[paper]](https://openreview.net/pdf?id=BecjRxs-lY5)

