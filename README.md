# Low-resource Knowledge Extraction

🍎 The repository is a paper set on low-resource knowledge extraction (NER, RE, EE), which is categorized into three paradigms. 

🤗 We strongly encourage the researchers who want to promote their fantastic work for the community to make pull request and update their papers in this repository! 

**Survey Paper**: Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective (2023) \[[paper](https://arxiv.org/abs/2202.08063)\] 

**Slides**: 

- Data-Efficient Knowledge Graph Construction, 高效知识图谱构建 ([Tutorial on CCKS 2022](http://sigkg.cn/ccks2022/?page_id=24)) \[[slides](https://drive.google.com/drive/folders/1xqeREw3dSiw-Y1rxLDx77r0hGUvHnuuE)\] 
- Efficient and Robust Knowledge Graph Construction ([Tutorial on AACL-IJCNLP 2022](https://www.aacl2022.org/Program/tutorials)) \[[paper](https://aclanthology.org/2022.aacl-tutorials.1.pdf), [slides](https://github.com/NLP-Tutorials/AACL-IJCNLP2022-KGC-Tutorial)\] 

**ToolKit**: 

- DeepKE: A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population [[paper](https://aclanthology.org/2022.emnlp-demos.10/), [project](https://github.com/zjunlp/DeepKE)]
- OpenUE: An Open Toolkit of Universal Extraction from Text [[paper](https://aclanthology.org/2020.emnlp-demos.1.pdf), [project](https://github.com/zjunlp/OpenUE)]
- OpenNRE [[project](https://github.com/thunlp/OpenNRE)]


## Content
* [**0. Related Surveys on Low-resource KE**](#Related-Surveys-on-Low-resource-KE)
  * [Knowledge Extraction](#Knowledge-Extraction)
  * [Low-resource NLP](#Low-resource-NLP)
  * [Low-resource Learning](#Low-resource-Learning)
* [**0. Low-resource KE Datasets**](#Low-resource-KE-Datasets)
  * [Low-resource NER](#Low-resource-NER)
  * [Low-resource RE](#Low-resource-RE)
  * [Low-resource EE](#Low-resource-EE)
* [**1. Exploiting Higher-resource Data**](#1-Exploiting-Higher-resource-Data)
  * [1.1 Weakly Supervised Augmentation](#Weakly-Supervised-Augmentation)
  * [1.2 Multi-modal Augmentation](#Multi-modal-Augmentation)
  * [1.3 Multi-lingual Augmentation](#Multi-lingual-Augmentation)
  * [1.4 Auxiliary Knowledge Enhancement](#Auxiliary-Knowledge-Enhancement)
* [**2. Exploiting Stronger Models**](#2-Exploiting-Stronger-Models)
  * [2.1 Meta Learning](#Meta-Learning)
  * [2.2 Transfer Learning](#Transfer-Learning)
  * [2.3 Prompt Learning](#Prompt-Learning)
* [**3. Exploiting Data and Models Together**](#3-Exploiting-Data-and-Models-Together)
  * [3.1 Multi-task Learning](#Multi-task-Learning)
  * [3.2 Task Reformulation](#Task-Reformulation)
  * [3.3 Retrieval Augmentation](#Retrieval-Augmentation)
* [**How to Cite**](#How-to-Cite)



## Related Surveys on Low-resource KE

### Knowledge Extraction
#### NER
* A Survey on Recent Advances in Named Entity Recognition from Deep Learning Models (COLING 2018) \[[paper](https://aclanthology.org/C18-1182.pdf)\]
* A Survey on Deep Learning for Named Entity Recognition (TKDE, 2020) \[[paper](https://ieeexplore.ieee.org/abstract/document/9039685)\]

#### RE
* A Survey on Neural Relation Extraction (Science China Technological Sciences, 2020) \[[paper](https://link.springer.com/article/10.1007/s11431-020-1673-6)\]
* Relation Extraction: A Brief Survey on Deep Neural Network Based Methods (ICSIM 2021) \[[paper](https://dl.acm.org/doi/abs/10.1145/3451471.3451506)\]
* Deep Neural Network-Based Relation Extraction: An Overview (Neural Computing and Applications, 2022) \[[paper](https://link.springer.com/article/10.1007/s00521-021-06667-3)\]

#### EE
* A Survey of Event Extraction From Text (ACCESS, 2019) \[[paper](https://ieeexplore.ieee.org/document/8918013)\]
* What is Event Knowledge Graph: A Survey (TKDE, 2022) \[[paper](https://ieeexplore.ieee.org/abstract/document/9792280)\]
* A Survey on Deep Learning Event Extraction: Approaches and Applications (TNNLS, 2022) \[[paper](https://ieeexplore.ieee.org/abstract/document/9927311)\]

#### General KE
* From Information to Knowledge: Harvesting Entities and Relationships from Web Sources (PODS 2010)  \[[paper](https://dl.acm.org/doi/abs/10.1145/1807085.1807097)\]
* Knowledge Base Population: Successful Approaches and Challenges (ACL 2011) \[[paper](https://aclanthology.org/P11-1115.pdf)\]
* Advances in Automated Knowledge Base Construction (NAACL-HLC 2012, AKBC-WEKEX workshop) \[[paper](https://www.semanticscholar.org/paper/Advances-in-Automated-Knowledge-Base-Construction-Suchanek/709e64be9cc9eb7c8b29bf49237cd2df835efd24)\]
* Information Extraction (IEEE Intelligent Systems, 2015) \[[paper](https://ieeexplore.ieee.org/abstract/document/7243219)\]
* Populating Knowledge Bases (Part of The Information Retrieval Series book series, 2018) \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-93935-3_6)\]
* A Survey on Open Information Extraction (COLING 2018) \[[paper](https://aclanthology.org/C18-1326.pdf)\]
* A Survey on Automatically Constructed Universal Knowledge Bases (Journal of Information Science, 2020) \[[paper](https://journals.sagepub.com/doi/abs/10.1177/0165551520921342)\]
* A Survey on Knowledge Graphs: Representation, Acquisition and Applications (TNNLS, 2021) \[[paper](https://ieeexplore.ieee.org/document/9416312)\]
* A Survey of Information Extraction Based on Deep Learning (Applied Sciences, 2022) \[[paper](https://www.mdpi.com/2076-3417/12/19/9691)\]
* Generative Knowledge Graph Construction: A Review (EMNLP 2022) \[[paper](https://aclanthology.org/2022.emnlp-main.1.pdf)\]
* Multi-Modal Knowledge Graph Construction and Application: A Survey (TKDE, 2022) \[[paper](https://ieeexplore.ieee.org/abstract/document/9961954)\]
* A Survey on Multimodal Knowledge Graphs: Construction, Completion and Applications (Mathematics, 2023) \[[paper](https://www.mdpi.com/2227-7390/11/8/1815)\]
* Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples! (arXiv, 2023) \[[paper](https://arxiv.org/abs/2303.08559)\]
<!--* Knowledge Extraction from Survey Data Using Neural Networks (Procedia Computer Science, 2013) \[[paper](https://www.sciencedirect.com/science/article/pii/S1877050913010995)\]-->

### Low-resource NLP
* A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios (NAACL 2021) \[[paper](https://aclanthology.org/2021.naacl-main.201.pdf)\]
* Few-Shot Named Entity Recognition: An Empirical Baseline Study (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.813.pdf)\]
* A Survey on Low-Resource Neural Machine Translation (IJCAI 2021) \[[paper](https://www.ijcai.org/proceedings/2021/0629.pdf)\]

### Low-resource Learning
* Generalizing from a Few Examples: A Survey on Few-shot Learning (ACM Computing Surveys, 2021) \[[paper](https://dl.acm.org/doi/10.1145/3386252)\]
* Knowledge-aware Zero-Shot Learning: Survey and Perspective (IJCAI 2021) \[[paper](https://www.ijcai.org/proceedings/2021/0597.pdf)\]
* Low-resource Learning with Knowledge Graphs: A Comprehensive Survey (2021) \[[paper](https://arxiv.org/abs/2112.10006)\]



## Low-resource KE Datasets

### Low-resource NER
* {***Few-NERD***}: Few-NERD: A Few-shot Named Entity Recognition Dataset (EMNLP 2021) \[[paper](https://aclanthology.org/2021.acl-long.248.pdf), [data](https://ningding97.github.io/fewnerd/)\]

### Low-resource RE
* {***FewRel***}: FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation (EMNLP 2018) \[[paper](https://aclanthology.org/D18-1514.pdf), [data](https://github.com/thunlp/FewRel)\]
* {***FewRel2.0***}: FewRel 2.0: Towards More Challenging Few-Shot Relation Classification (EMNLP 2019) \[[paper](https://aclanthology.org/D19-1649.pdf), [data](https://github.com/thunlp/FewRel)\]
* {***Entail-RE***}: Low-resource Extraction with Knowledge-aware Pairwise Prototype Learning (Knowledge-Based Systems, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S0950705121008467), [data](https://github.com/231sm/Reasoning_In_KE)\]
* {***LREBench***}: Towards Realistic Low-resource Relation Extraction: A Benchmark with Empirical Baseline Study (EMNLP 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-emnlp.29.pdf), [data](https://github.com/zjunlp/LREBench)\]

### Low-resource EE
* {***FewEvent***}: Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection (WSDM 2020) \[[paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371796), [data](https://github.com/231sm/Low_Resource_KBP)\]
* {***Causal-EE***}: Low-resource Extraction with Knowledge-aware Pairwise Prototype Learning (Knowledge-Based Systems, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S0950705121008467), [data](https://github.com/231sm/Reasoning_In_KE)\]
* {***OntoEvent***}: OntoED: Low-resource Event Detection with Ontology Embedding (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.220.pdf), [data](https://github.com/231sm/Reasoning_In_EE)\]



## 1 Exploiting Higher-resource Data

### Weakly Supervised Augmentation
* Distant Supervision for Relation Extraction without Labeled Data (ACL 2009) \[[paper](https://aclanthology.org/P09-1113.pdf)\]
* Neural Relation Extraction with Selective Attention over Instances (ACL 2016) \[[paper](https://aclanthology.org/P16-1200v2.pdf)\]
* Automatically Labeled Data Generation for Large Scale Event Extraction (ACL 2017) \[[paper](https://aclanthology.org/P17-1038.pdf)\]
* Adversarial Training for Weakly Supervised Event Detection (NAACL 2019) \[[paper](https://aclanthology.org/N19-1105.pdf)\]
* BOND: BERT-Assisted Open-Domain Named Entity Recognition with Distant Supervision (KDD 2020) \[[paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403149)\]
* Gradient Imitation Reinforcement Learning for Low Resource Relation Extraction (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.216.pdf)\]
* Noisy-Labeled NER with Confidence Estimation (NAACL 2021) \[[paper](https://aclanthology.org/2021.naacl-main.269.pdf)\]
* ANEA: Distant Supervision for Low-Resource Named Entity Recognition (ICLR 2021, Workshop of Practical Machine Learning For Developing Countries) \[[paper](https://arxiv.org/pdf/2102.13129.pdf)\]
* MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER  (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.160.pdf)\]
* Mask-then-Fill: A Flexible and Effective Data Augmentation Framework for Event Extraction (EMNLP 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-emnlp.332.pdf)\]
* Finding Influential Instances for Distantly Supervised Relation Extraction (COLING 2022) \[[paper](https://aclanthology.org/2022.coling-1.233.pdf)\]
<!--* Distant Supervision for Relation Extraction with Sentence-Level Attention and Entity Descriptions (AAAI 2017) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/10953)\]-->
<!--* Reinforcement Learning for Relation Classification From Noisy Data (AAAI 2018) \[[paper](https://dl.acm.org/doi/abs/10.5555/3504035.3504744)\]-->
<!--* Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning  (ACL 2018) \[[paper](https://aclanthology.org/P18-1199.pdf)\]-->
<!--* Learning Named Entity Tagger using Domain-Specific Dictionary (EMNLP 2018) \[[paper](https://aclanthology.org/D18-1230.pdf)\]-->

### Multi-modal Augmentation
* Visual Attention Model for Name Tagging in Multimodal Social Media (ACL 2018) \[[paper](https://aclanthology.org/P18-1185.pdf)\]
* Cross-media Structured Common Space for Multimedia Event Extraction (ACL 2020) \[[paper](https://aclanthology.org/2020.acl-main.230.pdf)\]
* Joint Multimedia Event Extraction from Video and Article (EMNLP 2021, Findings) \[[paper](https://aclanthology.org/2021.findings-emnlp.8.pdf)\]
* Multimodal Relation Extraction with Efficient Graph Alignment (MM 2021) \[[paper](https://dl.acm.org/doi/10.1145/3474085.3476968)\]
* Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion (SIGIR 2022) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3477495.3531992)\]
* Good Visual Guidance Makes A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction (NAACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-naacl.121.pdf)\]


### Multi-lingual Augmentation
* Neural Relation Extraction with Multi-lingual Attention (ACL 2017) \[[paper](https://aclanthology.org/P17-1004.pdf)\]
* Improving Low Resource Named Entity Recognition using Cross-lingual Knowledge Transfer (IJCAI 2018) \[[paper](https://www.ijcai.org/Proceedings/2018/0566.pdf)\]
* Event Detection via Gated Multilingual Attention Mechanism (AAAI 2018) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11919)\]
* Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning (COLING 2022) \[[paper](https://aclanthology.org/2022.coling-1.382.pdf)\]
* Cross-lingual Transfer Learning for Relation Extraction Using Universal Dependencies (Computer Speech & Language, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S0885230821000711)\]
* Language Model Priming for Cross-Lingual Event Extraction (AAAI 2022) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21307)\]


### Auxiliary Knowledge Enhancement

#### (1) Text
* Improving Event Detection via Open-domain Trigger Knowledge (ACL 2020) \[[paper](https://aclanthology.org/2020.acl-main.522.pdf)\]
* MapRE: An Effective Semantic Mapping Approach for Low-resource Relation Extraction (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.212.pdf)\]

#### (2) KG
* Leveraging FrameNet to Improve Automatic Event Detection (ACL 2016) \[[paper](https://aclanthology.org/P16-1201.pdf)\]
* DOZEN: Cross-Domain Zero Shot Named Entity Recognition with Knowledge Graph (SIGIR 2021) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463113)\]

#### (3) Ontology & Rule
* Logic-guided Semantic Representation Learning for Zero-Shot Relation Classification (COLING 2020) \[[paper](https://aclanthology.org/2020.coling-main.265.pdf)\]
* OntoED: Low-resource Event Detection with Ontology Embedding (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.220.pdf)\]
* Neuralizing Regular Expressions for Slot Filling (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.747.pdf)\]
* Low-resource Extraction with Knowledge-aware Pairwise Prototype Learning (Knowledge-Based Systems, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S0950705121008467)\]



## 2 Exploiting Stronger Models

### Meta Learning

#### For Low-resource NER
* Few-shot Classification in Named Entity Recognition Task (SAC 2019) \[[paper](https://dl.acm.org/doi/10.1145/3297280.3297378)\]
* Enhanced Meta-Learning for Cross-Lingual Named Entity Recognition with Minimal Resources (AAAI 2020) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/64667)\]
* MetaNER: Named Entity Recognition with Meta-Learning (WWW 2020) \[[paper](https://dl.acm.org/doi/10.1145/3366423.3380127)\]
* Few-Shot Named Entity Recognition via Meta-Learning (TKDE, 2022) \[[paper](https://doi.org/10.1109/TKDE.2020.3038670)\]

#### For Low-resource RE
* Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification (AAAI 2019) \[[paper](https://ojs.aaai.org//index.php/AAAI/article/view/4604)\]
* Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs (ICML 2020) \[[paper](http://proceedings.mlr.press/v119/qu20a/qu20a.pdf)\]
* Enhanced Meta-Learning for Cross-Lingual Named Entity Recognition with Minimal Resources (AAAI 2020) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6466)\]
* Bridging Text and Knowledge with Multi-Prototype Embedding for Few-Shot Relational Triple Extraction (COLING 2020) \[[paper](https://aclanthology.org/2020.coling-main.563.pdf)\]
* Meta-Information Guided Meta-Learning for Few-Shot Relation Classification (COLING 2020) \[[paper](https://aclanthology.org/2020.coling-main.140.pdf)\]
* Pre-training to Match for Unified Low-shot Relation Extraction (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.397.pdf)\]

#### For Low-resource EE
* Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection (WSDM 2020) \[[paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371796)\]
* Adaptive Knowledge-Enhanced Bayesian Meta-Learning for Few-shot Event Detection (ACL 2021, Findings) \[[paper](https://aclanthology.org/2021.findings-acl.214.pdf)\]
* Few-Shot Event Detection with Prototypical Amortized Conditional Random Field (ACL 2021, Findings) \[[paper](https://aclanthology.org/2021.findings-acl.3.pdf)\]


### Transfer Learning

#### (1) Class-related Semantics
* Zero-Shot Transfer Learning for Event Extraction (ACL 2018) \[[paper](https://aclanthology.org/P18-1201.pdf)\]
* Transfer Learning for Named-Entity Recognition with Neural Networks (LREC 2018) \[[paper](https://aclanthology.org/L18-1708.pdf)\]
* Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks (NAACL 2019) \[[paper](https://aclanthology.org/N19-1306.pdf)\]
* Relation Adversarial Network for Low Resource Knowledge Graph Completion (WWW 2020) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3366423.3380089)\]
* Graph Learning Regularization and Transfer Learning for Few-Shot Event Detection (SIGIR 2021) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463054)\]
* LearningToAdapt with Word Embeddings: Domain Adaptation of
Named Entity Recognition Systems (Information Processing and Management, 2021) \[[paper](https://www.sciencedirect.com/science/article/pii/S0306457321000455)\]

#### (2) Pre-trained Language Representations
* Matching the Blanks: Distributional Similarity for Relation Learning (ACL 2019) \[[paper](https://aclanthology.org/P19-1279.pdf)\]
* Exploring Pre-trained Language Models for Event Extraction and Generation (ACL 2019) \[[paper](https://aclanthology.org/P19-1522.pdf)\]
* Coarse-to-Fine Pre-training for Named Entity Recognition (EMNLP 2020) \[[paper](https://aclanthology.org/2020.emnlp-main.514.pdf)\]
* CLEVE: Contrastive Pre-training for Event Extraction (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.491.pdf)\]
* Unleash GPT-2 Power for Event Detection (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.490.pdf)\]


### Prompt Learning

#### (1) Vanilla Prompt Learning
* Template-Based Named Entity Recognition Using BART (ACL 2021, Findings) \[[paper](https://aclanthology.org/2021.findings-acl.161.pdf)\]
* LightNER: A Lightweight Tuning Paradigm for Low-resource NER via Pluggable Prompting (COLING 2022) \[[paper](https://aclanthology.org/2022.coling-1.209.pdf)\]
* Template-free Prompt Tuning for Few-shot NER (NAACL 2022) \[[paper](https://aclanthology.org/2022.naacl-main.420.pdf)\]
* COPNER: Contrastive Learning with Prompt Guiding for Few-shot Named Entity Recognition (COLING 2022) \[[paper](https://aclanthology.org/2022.coling-1.222.pdf)\]
* RelationPrompt: Leveraging Prompts to Generate Synthetic Data for
Zero-Shot Relation Triplet Extraction (ACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-acl.5.pdf)\]
* Dynamic Prefix-Tuning for Generative Template-based Event Extraction (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.358.pdf)\]

#### (2) Augmented Prompt Learning
* PTR: Prompt Tuning with Rules for Text Classification (AI Open, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S2666651022000183)\]
* KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction (WWW 2022) \[[paper](https://dl.acm.org/doi/abs/10.1145/3485447.3511998)\]
* Ontology-enhanced Prompt-tuning for Few-shot Learning (WWW 2022) \[[paper](https://dl.acm.org/doi/10.1145/3485447.3511921)\]
* Unified Structure Generation for Universal Information Extraction (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.395.pdf)\]
* DEGREE: A Data-Efficient Generation-Based Event Extraction Model (NAACL 2022) \[[paper](https://aclanthology.org/2022.naacl-main.138.pdf)\]
* AugPrompt: Knowledgeable Augmented-Trigger Prompt for Few-Shot Event Classification (Information Processing & Management, 2022) \[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0306457322002540)\]
* Schema-aware Reference as Prompt Improves Data-Efficient Relational Triple and Event Extraction (SIGIR 2023) \[[paper](https://arxiv.org/abs/2210.10709)\]



## 3 Exploiting Data and Models Together

### Multi-task Learning

#### (1) NER, Named Entity Normalization (NEN)
* A Neural Multi-Task Learning Framework to Jointly Model Medical Named Entity Recognition and Normalization (AAAI 2019) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/3861)\]
* MTAAL: Multi-Task Adversarial Active Learning for Medical Named Entity Recognition and Normalization (AAAI 2021) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17714)\]
* An End-to-End Progressive Multi-Task Learning Framework for Medical Named Entity Recognition and Normalization (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.485.pdf)\]

#### (2) NER, RE
* GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction (ACL 2019) \[[paper](https://aclanthology.org/P19-1136.pdf)\]
* CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning (AAAI 2020) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6495)\]
* Joint Entity and Relation Extraction Model based on Rich Semantics (Neurocomputing, 2021) \[[paper](https://www.sciencedirect.com/science/article/pii/S0925231220319378?casa_token=jzgLW9J1UKoAAAAA:5vnqqGKt0_-ykbhTp15Bq8mB-8B50cM3LDa10q2h8yc4q4AJVfeEbQV_fyMo2Z92xjl3HPNt6w)\]

#### (3) NER, RE, EE
* Entity, Relation, and Event Extraction with Contextualized Span Representations (EMNLP 2019) \[[paper](https://aclanthology.org/D19-1585.pdf)\]
* InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction (arXiv, 2023) \[[paper](https://arxiv.org/abs/2304.08085)\]

#### (4) Word Sense Disambiguation (WSD), Event Detection (ED)
* Similar but not the Same: Word Sense Disambiguation Improves Event Detection via Neural Representation Matching (EMNLP 2018) \[[paper](https://aclanthology.org/D18-1517.pdf)\]

#### (5) NER, RE, EE & Other Structure Prediction Tasks
* DeepStruct: Pretraining of Language Models for Structure Prediction (ACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-acl.67v2.pdf)\]


### Task Reformulation

#### QA/MRC
* Zero-Shot Relation Extraction via Reading Comprehension (CoNLL 2017) \[[paper](https://aclanthology.org/K17-1034.pdf)\]
* Entity-Relation Extraction as Multi-Turn Question Answering (ACL 2019) \[[paper](http://aclanthology.lst.uni-saarland.de/P19-1129.pdf)\]
* A Unified MRC Framework for Named Entity Recognition (ACL 2020) \[[paper](https://aclanthology.org/2020.acl-main.519.pdf)\]
* Event Extraction as Machine Reading Comprehension (EMNLP 2020) \[[paper](https://aclanthology.org/2020.emnlp-main.128.pdf)\]
* Event Extraction by Answering (Almost) Natural Questions (EMNLP 2020) \[[paper](https://aclanthology.org/2020.emnlp-main.49.pdf)\]
* Learning to Ask for Data-Efficient Event Argument Extraction (AAAI 2022, Student Abstract) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21686)\]

#### Text Generation
* Text2Event: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.217/)\]
* Structured Prediction as Translation between Augmented Natural Languages (ICLR 2021) \[[paper](https://openreview.net/forum?id=US-TP-xnXI)\]
* Unified Structure Generation for Universal Information Extraction (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.395.pdf)\]
* LasUIE: Unifying Information Extraction with Latent Adaptive Structure-aware Generative Language Model (NeurIPS 2022) \[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/63943ee9fe347f3d95892cf87d9a42e6-Abstract-Conference.html)\]
* Universal Information Extraction as Unified Semantic Matching (AAAI 2023) \[[paper](https://arxiv.org/abs/2301.03282)\]
* CODE4STRUCT: Code Generation for Few-Shot Structured Prediction from Natural Language (arXiv, 2022) \[[paper](https://arxiv.org/abs/2210.12810)\]


### Retrieval Augmentation

#### Retrieval-based Low-resource KE
* Few-shot Intent Classification and Slot Filling with Retrieved Examples (NAACL 2021) \[[paper](https://aclanthology.org/2021.naacl-main.59.pdf)\]
* Relation Extraction as Open-book Examination: Retrieval-enhanced Prompt Tuning (SIGIR 2022, Short Paper) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3477495.3531746)\]
* Decoupling Knowledge from Memorization: Retrieval-Augmented Prompt Learning (NeurIPS 2022, Spotlight) \[[paper](https://openreview.net/forum?id=Q8GnGqT-GTJ)\]
* Retrieval-Augmented Generative Question Answering for Event Argument Extraction (EMNLP 2022) \[[paper](https://aclanthology.org/2022.emnlp-main.307.pdf)\]

#### Retrieval-based Language Models in Low-resource Scenarios
* KNN-BERT: Fine-Tuning Pre-Trained Models with KNN Classifier (2021) \[[paper](https://arxiv.org/abs/2110.02523)\]
* Few-shot Learning with Retrieval Augmented Language Models (2022, Meta AI, Atlas) \[[paper](https://arxiv.org/abs/2208.03299)\]

<!--* Improving Neural Language Models with a Continuous Cache (ICLR 2017) \[[paper](https://openreview.net/forum?id=B184E5qee)\]
* Unbounded Cache Model for Online Language Modeling with Open Vocabulary (NeurIPS 2017) \[[paper](https://proceedings.neurips.cc/paper/2017/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf)\]
* Generalization through Memorization: Nearest Neighbor Language Models (ICLR 2020) \[[paper](https://openreview.net/forum?id=HklBjCEKvH)\]
* REALM: Retrieval Augmented Language Model Pre-Training (ICML 2020) \[[paper](http://proceedings.mlr.press/v119/guu20a.html)\]
* Adaptive Semiparametric Language Models (TACL, 2021) \[[paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00371/100688/Adaptive-Semiparametric-Language-Models)\]
* BERT-kNN: Adding a kNN Search Component to Pretrained Language Models for Better QA (EMNLP 2020, Findings) \[[paper](https://aclanthology.org/2020.findings-emnlp.307.pdf)\]
* Efficient Nearest Neighbor Language Models (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.461.pdf)\]
* End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering (NeurIPS 2021) \[[paper](https://openreview.net/forum?id=5KWmB6JePx)\]
* Improving Language Models by Retrieving from Trillions of Tokens (ICML 2022) \[[paper](https://proceedings.mlr.press/v162/borgeaud22a.html)\]-->



## How to Cite

📋 Thank you very much for your interest in our survey work. If you use or extend our survey, please cite the following paper:

```bibtex
@inproceedings{2023_LowResKE,
    author    = {Shumin Deng and
                 Ningyu Zhang and
                 Bryan Hooi},
    title     = {Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective},
    journal   = {CoRR},
    volume    = {abs/2202.08063},
    year      = {2023},
    url       = {https://arxiv.org/abs/2202.08063}
}
```
