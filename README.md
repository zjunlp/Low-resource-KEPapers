# Low-resource Information Extraction 🚀


🍎 The repository is a paper set on low-resource information extraction (NER, RE, EE), which is categorized into three paradigms. 

🤗 We strongly encourage the researchers who want to promote their fantastic work for the community to make pull request and update their papers in this repository! 

📖 **Survey Paper**: Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective (2023) \[[paper](https://arxiv.org/abs/2202.08063)\]

🗂️ **Slides**: 

- Data-Efficient Knowledge Graph Construction, 高效知识图谱构建 ([Tutorial on CCKS 2022](http://sigkg.cn/ccks2022/?page_id=24)) \[[slides](https://drive.google.com/drive/folders/1xqeREw3dSiw-Y1rxLDx77r0hGUvHnuuE)\] 
- Efficient and Robust Knowledge Graph Construction ([Tutorial on AACL-IJCNLP 2022](https://www.aacl2022.org/Program/tutorials)) \[[paper](https://aclanthology.org/2022.aacl-tutorials.1.pdf), [slides](https://github.com/NLP-Tutorials/AACL-IJCNLP2022-KGC-Tutorial)\] 
- Open-Environment Knowledge Graph Construction and Reasoning: Challenges, Approaches, and Opportunities ([Tutorial on IJCAI 2023](https://ijcai-23.org/tutorials/))  \[[slides](https://openkg-tutorial.github.io/)\]



## Content

[**Preliminaries**](#Preliminaries)

* [**🛠️ Low-Resource IE Toolkits**](#%EF%B8%8F-Low-Resource-IE-Toolkits)
  * [Traditional Toolkits](#Traditional-Toolkits)
  * [LLM-Based Toolkits](#LLM-Based-Toolkits)
* [**📊 Low-Resource IE Datasets**](#-Low-Resource-IE-Datasets)
  * [Low-Resource NER](#Low-Resource-NER)
  * [Low-Resource RE](#Low-Resource-RE)
  * [Low-Resource EE](#Low-Resource-EE)
* [**📖 Related Surveys/Analysis on Low-Resource IE**](#-Related-Surveys-and-Analysis-on-Low-Resource-IE)
  * [Information Extraction](#Information-Extraction)
  * [Low-Resource NLP Learning](#Low-Resource-NLP-Learning)


[**🍎Traditional Methods🍎**](#-Traditional-Methods-)

* [**1. Exploiting Higher-Resource Data**](#1-Exploiting-Higher-Resource-Data)
  * [1.1 Weakly Supervised Augmentation](#Weakly-Supervised-Augmentation)
  * [1.2 Multimodal Augmentation](#Multi-Modal-Augmentation)
  * [1.3 Multi-Lingual Augmentation](#Multi-Lingual-Augmentation)
  * [1.4 Auxiliary Knowledge Enhancement](#Auxiliary-Knowledge-Enhancement)
* [**2. Developing Stronger Data-Efficient Models**](#2-Developing-Stronger-Data-Efficient-Models)
  * [2.1 Meta Learning](#Meta-Learning)
  * [2.2 Transfer Learning](#Transfer-Learning)
  * [2.3 Fine-Tuning PLM](#Fine-Tuning-PLM)
* [**3. Optimizing Data and Models Together**](#3-Optimizing-Data-and-Models-Together)
  * [3.1 Multi-Task Learning](#Multi-Task-Learning)
  * [3.2 Task Reformulation](#Task-Reformulation)
  * [3.3 Prompt-Tuning PLM](#Prompt-Tuning-PLM)


[**🍏LLM-Based Methods🍏**](#-LLM-Based-Methods-)
  
* [**Direct Inference Without Tuning**](#Direct-Inference-Without-Tuning)
  * [Instruction Prompting](#Instruction-Prompting)
  * [Code Prompting](#Code-Prompting)
  * [In-Context Learning](#In-Context-Learning)
* [**Model Specialization With Tuning**](#Model-Specialization-With-Tuning)
  * [Prompt-Tuning LLM](#Prompt-Tuning-LLM) 
  * [Fine-Tuning LLM](#Fine-Tuning-LLM)

[**How to Cite**](#How-to-Cite)

<!--  * [Fine-Tuning LLM](#Fine-Tuning-LLM) -->
<!--  * [Retrieval-Augmented Prompting](#Retrieval-Augmented-Prompting)-->


## Preliminaries

## 🛠️ Low-Resource IE Toolkits

### Traditional Toolkits
- DeepKE: A Deep Learning Based Knowledge Extraction Toolkit for Knowledge Base Population [[paper](https://aclanthology.org/2022.emnlp-demos.10/), [project](https://github.com/zjunlp/DeepKE)]
- OpenUE: An Open Toolkit of Universal Extraction from Text [[paper](https://aclanthology.org/2020.emnlp-demos.1.pdf), [project](https://github.com/zjunlp/OpenUE)]
- Zshot: An Open-source Framework for Zero-Shot Named Entity Recognition and Relation Extraction [[paper](https://aclanthology.org/2023.acl-demo.34/), [project](https://github.com/IBM/zshot)]
- OmniEvent [[paper](https://aclanthology.org/2023.findings-acl.586.pdf), [project](https://github.com/THU-KEG/OmniEvent)]
- OpenNRE [[project](https://github.com/thunlp/OpenNRE)]

### LLM-Based Toolkits
- GPT4IE [[project](https://github.com/cocacola-lab/GPT4IE)]
- ChatIE [[paper](https://arxiv.org/abs/2302.10205), [project](https://github.com/cocacola-lab/ChatIE)]
- TechGPT: Technology-Oriented Generative Pretrained Transformer [[project](https://github.com/neukg/TechGPT)] 
- AutoKG: LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities [[paper](https://arxiv.org/abs/2305.13168), [project](https://github.com/zjunlp/AutoKG)]
- KnowLM [[project](https://github.com/zjunlp/KnowLM)] 


## 📊 Low-Resource IE Datasets

### Low-Resource NER
* {***Few-NERD***}: Few-NERD: A Few-shot Named Entity Recognition Dataset (EMNLP 2021) \[[paper](https://aclanthology.org/2021.acl-long.248.pdf), [data](https://ningding97.github.io/fewnerd/)\]

### Low-Resource RE
* {***FewRel***}: FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation (EMNLP 2018) \[[paper](https://aclanthology.org/D18-1514.pdf), [data](https://github.com/thunlp/FewRel)\]
* {***FewRel2.0***}: FewRel 2.0: Towards More Challenging Few-Shot Relation Classification (EMNLP 2019) \[[paper](https://aclanthology.org/D19-1649.pdf), [data](https://github.com/thunlp/FewRel)\]
* {***Wiki-ZSL***}: ZS-BERT: Towards Zero-Shot Relation Extraction with Attribute Representation Learning (NAACL 2021) \[[paper](https://aclanthology.org/2021.naacl-main.272.pdf), [data](https://github.com/dinobby/ZS-BERT)\]
* {***Entail-RE***}: Low-resource Extraction with Knowledge-aware Pairwise Prototype Learning (Knowledge-Based Systems, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S0950705121008467), [data](https://github.com/231sm/Reasoning_In_KE)\]
* {***LREBench***}: Towards Realistic Low-resource Relation Extraction: A Benchmark with Empirical Baseline Study (EMNLP 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-emnlp.29.pdf), [data](https://github.com/zjunlp/LREBench)\]

### Low-Resource EE
* {***FewEvent***}: Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection (WSDM 2020) \[[paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371796), [data](https://github.com/231sm/Low_Resource_KBP)\]
* {***Causal-EE***}: Low-resource Extraction with Knowledge-aware Pairwise Prototype Learning (Knowledge-Based Systems, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S0950705121008467), [data](https://github.com/231sm/Reasoning_In_KE)\]
* {***OntoEvent***}: OntoED: Low-resource Event Detection with Ontology Embedding (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.220.pdf), [data](https://github.com/231sm/Reasoning_In_EE)\]
* {***FewDocAE***}: Few-Shot Document-Level Event Argument Extraction (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.446.pdf), [data](https://github.com/Xianjun-Yang/FewDocAE)\]


## 📖 Related Surveys and Analysis on Low-Resource IE

### Information Extraction
#### NER
* A Survey on Recent Advances in Named Entity Recognition from Deep Learning Models (COLING 2018) \[[paper](https://aclanthology.org/C18-1182.pdf)\]
* A Survey on Deep Learning for Named Entity Recognition (TKDE, 2020) \[[paper](https://ieeexplore.ieee.org/abstract/document/9039685)\]
* Few-Shot Named Entity Recognition: An Empirical Baseline Study (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.813.pdf)\]
* Few-shot Named Entity Recognition: definition, taxonomy and research directions (TIST, 2023) \[[paper](https://dl.acm.org/doi/10.1145/3609483)\]

#### RE
* A Survey on Neural Relation Extraction (Science China Technological Sciences, 2020) \[[paper](https://link.springer.com/article/10.1007/s11431-020-1673-6)\]
* Relation Extraction: A Brief Survey on Deep Neural Network Based Methods (ICSIM 2021) \[[paper](https://dl.acm.org/doi/abs/10.1145/3451471.3451506)\]
* Revisiting Few-shot Relation Classification: Evaluation Data and Classification Schemes (TACL, 2021) \[[paper](https://aclanthology.org/2021.tacl-1.42.pdf)\]
* Deep Neural Network-Based Relation Extraction: An Overview (Neural Computing and Applications, 2022) \[[paper](https://link.springer.com/article/10.1007/s00521-021-06667-3)\]
* Revisiting Relation Extraction in the era of Large Language Models (ACL 2023) [[paper](https://aclanthology.org/2023.acl-long.868.pdf)\]

#### EE
* A Survey of Event Extraction From Text (ACCESS, 2019) \[[paper](https://ieeexplore.ieee.org/document/8918013)\]
* What is Event Knowledge Graph: A Survey (TKDE, 2022) \[[paper](https://ieeexplore.ieee.org/abstract/document/9792280)\]
* A Survey on Deep Learning Event Extraction: Approaches and Applications (TNNLS, 2022) \[[paper](https://ieeexplore.ieee.org/abstract/document/9927311)\]
* Event Extraction: A Survey (2022) [[paper](https://arxiv.org/abs/2210.03419)\]
* Few-shot Event Detection: An Empirical Study and a Unified View (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.628.pdf)\]
* Exploring the Feasibility of ChatGPT for Event Extraction (arXiv, 2023) [[paper](https://arxiv.org/abs/2303.03836)\]
* A Reevaluation of Event Extraction: Past, Present, and Future Challenges (arXiv, 2023) [[paper](https://arxiv.org/abs/2311.09562)\]
<!--* Low Resource Event Extraction: A Survey (2022) [[paper](https://www.cs.uoregon.edu/Reports/AREA-202210-Lai.pdf)\]-->

#### General IE
**Traditional IE** 

* From Information to Knowledge: Harvesting Entities and Relationships from Web Sources (PODS 2010)  \[[paper](https://dl.acm.org/doi/abs/10.1145/1807085.1807097)\]
* Knowledge Base Population: Successful Approaches and Challenges (ACL 2011) \[[paper](https://aclanthology.org/P11-1115.pdf)\]
* Advances in Automated Knowledge Base Construction (NAACL-HLC 2012, AKBC-WEKEX workshop) \[[paper](https://www.semanticscholar.org/paper/Advances-in-Automated-Knowledge-Base-Construction-Suchanek/709e64be9cc9eb7c8b29bf49237cd2df835efd24)\]
* Information Extraction (IEEE Intelligent Systems, 2015) \[[paper](https://ieeexplore.ieee.org/abstract/document/7243219)\]
* Populating Knowledge Bases (Part of The Information Retrieval Series book series, 2018) \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-93935-3_6)\]
* A Survey on Open Information Extraction (COLING 2018) \[[paper](https://aclanthology.org/C18-1326.pdf)\]
* A Survey on Automatically Constructed Universal Knowledge Bases (Journal of Information Science, 2020) \[[paper](https://journals.sagepub.com/doi/abs/10.1177/0165551520921342)\]
* Machine Knowledge: Creation and Curation of Comprehensive Knowledge Bases (Submitted to Foundations and Trends in Databases, 2020) [[paper](https://arxiv.org/abs/2009.11564)\]
* A Survey on Knowledge Graphs: Representation, Acquisition and Applications (TNNLS, 2021) \[[paper](https://ieeexplore.ieee.org/document/9416312)\]
* A Survey of Information Extraction Based on Deep Learning (Applied Sciences, 2022) \[[paper](https://www.mdpi.com/2076-3417/12/19/9691)\]
* Generative Knowledge Graph Construction: A Review (EMNLP 2022) \[[paper](https://aclanthology.org/2022.emnlp-main.1.pdf)\]
* Multi-Modal Knowledge Graph Construction and Application: A Survey (TKDE, 2022) \[[paper](https://ieeexplore.ieee.org/abstract/document/9961954)\]
* A Survey on Multimodal Knowledge Graphs: Construction, Completion and Applications (Mathematics, 2023) \[[paper](https://www.mdpi.com/2227-7390/11/8/1815)\]
* Construction of Knowledge Graphs: State and Challenges (Submitted to Semantic Web Journal, 2023) \[[paper](https://www.semantic-web-journal.net/content/construction-knowledge-graphs-state-and-challenges)\]

**LLM-based IE**

* Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples! (EMNLP 2023, Findings) \[[paper](https://arxiv.org/abs/2303.08559)\]
* Evaluating ChatGPT’s Information Extraction Capabilities: An Assessment of Performance, Explainability, Calibration, and Faithfulness (arXiv, 2023) \[[paper](https://arxiv.org/abs/2304.11633)\] 
* Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.14450)\]
* Improving Open Information Extraction with Large Language Models: A Study on Demonstration Uncertainty (arXiv, 2023) \[[paper](https://arxiv.org/abs/2309.03433)\]
* LOKE: Linked Open Knowledge Extraction for Automated Knowledge Graph Construction (arXiv, 2023) \[[paper](https://arxiv.org/abs/2311.09366)\]
* LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.13168)\]
* Large Language Models and Knowledge Graphs:
Opportunities and Challenges (arXiv, 2023) \[[paper](https://arxiv.org/abs/2308.06374)\]
* Unifying Large Language Models and Knowledge Graphs: A Roadmap (arXiv, 2023) \[[paper](https://arxiv.org/abs/2306.08302)\]
* Trends in Integration of Knowledge and Large Language Models: A Survey and Taxonomy of Methods, Benchmarks, and Applications (arXiv, 2023) \[[paper](https://arxiv.org/abs/2311.05876)\]
* Knowledge Bases and Language Models: Complementing Forces (RuleML+RR, 2023) \[[paper](https://link.springer.com/chapter/10.1007/978-3-031-45072-3_1)\]
* StructGPT: A General Framework for Large Language Model
to Reason over Structured Data (EMNLP 2023) \[[paper](https://arxiv.org/abs/2305.09645)\]
<!--* Knowledge Extraction from Survey Data Using Neural Networks (Procedia Computer Science, 2013) \[[paper](https://www.sciencedirect.com/science/article/pii/S1877050913010995)\]-->

### Low-Resource NLP Learning
* A Survey of Zero-Shot Learning: Settings, Methods, and Applications (TIST, 2019) \[[paper](https://dl.acm.org/doi/10.1145/3293318)\]
* A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios (NAACL 2021) \[[paper](https://aclanthology.org/2021.naacl-main.201.pdf)\]
* A Survey on Low-Resource Neural Machine Translation (IJCAI 2021) \[[paper](https://www.ijcai.org/proceedings/2021/0629.pdf)\]
* Generalizing from a Few Examples: A Survey on Few-shot Learning (ACM Computing Surveys, 2021) \[[paper](https://dl.acm.org/doi/10.1145/3386252)\]
* Knowledge-aware Zero-Shot Learning: Survey and Perspective (IJCAI 2021) \[[paper](https://www.ijcai.org/proceedings/2021/0597.pdf)\]
* Generalizing to Unseen Elements: A Survey on Knowledge Extrapolation for Knowledge Graphs (IJCAI 2023) \[[paper](https://www.ijcai.org/proceedings/2023/0737.pdf)\]
* Zero-shot and Few-shot Learning with Knowledge Graphs: A Comprehensive Survey (Proceedings of the IEEE, 2023) \[[paper](https://arxiv.org/abs/2112.10006)\]
* A Survey on Machine Learning from Few Samples (Pattern Recognition, 2023) \[[paper](https://www.sciencedirect.com/science/article/pii/S0031320323001802)\]


## 🍎 Traditional Methods 🍎

## 1 Exploiting Higher-Resource Data

### Weakly Supervised Augmentation
* Distant Supervision for Relation Extraction without Labeled Data (ACL 2009) \[[paper](https://aclanthology.org/P09-1113.pdf)\]
* Modeling Missing Data in Distant Supervision for Information Extraction (TACL, 2013) \[[paper](https://aclanthology.org/Q13-1030.pdf)\]
* Neural Relation Extraction with Selective Attention over Instances (ACL 2016) \[[paper](https://aclanthology.org/P16-1200v2.pdf)\]
* Automatically Labeled Data Generation for Large Scale Event Extraction (ACL 2017) \[[paper](https://aclanthology.org/P17-1038.pdf)\]
* CoType: Joint Extraction of Typed Entities and Relations
with Knowledge Bases (WWW 2017) \[[paper](https://dl.acm.org/doi/abs/10.1145/3038912.3052708)\]
* Adversarial Training for Weakly Supervised Event Detection (NAACL 2019) \[[paper](https://aclanthology.org/N19-1105.pdf)\]
* Local Additivity Based Data Augmentation for Semi-supervised NER (EMNLP 2020) \[[paper](https://aclanthology.org/2020.emnlp-main.95/)\]
* BOND: BERT-Assisted Open-Domain Named Entity Recognition with Distant Supervision (KDD 2020) \[[paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403149)\]
* Gradient Imitation Reinforcement Learning for Low Resource Relation Extraction (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.216.pdf)\]
* Noisy-Labeled NER with Confidence Estimation (NAACL 2021) \[[paper](https://aclanthology.org/2021.naacl-main.269.pdf)\]
* ANEA: Distant Supervision for Low-Resource Named Entity Recognition (ICLR 2021, Workshop of Practical Machine Learning For Developing Countries) \[[paper](https://arxiv.org/pdf/2102.13129.pdf)\]
* Finding Influential Instances for Distantly Supervised Relation Extraction (COLING 2022) \[[paper](https://aclanthology.org/2022.coling-1.233.pdf)\]
* Better Sampling of Negatives for Distantly Supervised Named Entity Recognition (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.300.pdf)\]
* Jointprop: Joint Semi-supervised Learning for Entity and Relation Extraction with Heterogeneous Graph-based Propagation (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.813.pdf)\]
<!--* Distant Supervision for Relation Extraction with Sentence-Level Attention and Entity Descriptions (AAAI 2017) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/10953)\]-->
<!--* Reinforcement Learning for Relation Classification From Noisy Data (AAAI 2018) \[[paper](https://dl.acm.org/doi/abs/10.5555/3504035.3504744)\]-->
<!--* Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning  (ACL 2018) \[[paper](https://aclanthology.org/P18-1199.pdf)\]-->
<!--* Learning Named Entity Tagger using Domain-Specific Dictionary (EMNLP 2018) \[[paper](https://aclanthology.org/D18-1230.pdf)\]-->

### Multimodal Augmentation
* Visual Attention Model for Name Tagging in Multimodal Social Media (ACL 2018) \[[paper](https://aclanthology.org/P18-1185.pdf)\]
* Visual Relation Extraction via Multi-modal Translation Embedding Based Model (PAKDD 2018) \[[paper](https://link.springer.com/chapter/10.1007/978-3-319-93034-3_43)\]
* Cross-media Structured Common Space for Multimedia Event Extraction (ACL 2020) \[[paper](https://aclanthology.org/2020.acl-main.230.pdf)\]
* Image Enhanced Event Detection in News Articles (AAAI 2020) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6437)\]
* Joint Multimedia Event Extraction from Video and Article (EMNLP 2021, Findings) \[[paper](https://aclanthology.org/2021.findings-emnlp.8.pdf)\]
* Multimodal Relation Extraction with Efficient Graph Alignment (MM 2021) \[[paper](https://dl.acm.org/doi/10.1145/3474085.3476968)\]
* Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion (SIGIR 2022) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3477495.3531992)\]
* Good Visual Guidance Makes A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction (NAACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-naacl.121.pdf)\]


### Multi-Lingual Augmentation
* Low-Resource Named Entity Recognition with Cross-lingual, Character-Level Neural Conditional Random Fields (IJCNLP 2017) \[[paper](https://aclanthology.org/I17-2016.pdf)\]
* Neural Relation Extraction with Multi-lingual Attention (ACL 2017) \[[paper](https://aclanthology.org/P17-1004.pdf)\]
* Improving Low Resource Named Entity Recognition using Cross-lingual Knowledge Transfer (IJCAI 2018) \[[paper](https://www.ijcai.org/Proceedings/2018/0566.pdf)\]
* Event Detection via Gated Multilingual Attention Mechanism (AAAI 2018) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11919)\]
* Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning (COLING 2022) \[[paper](https://aclanthology.org/2022.coling-1.382.pdf)\]
* Cross-lingual Transfer Learning for Relation Extraction Using Universal Dependencies (Computer Speech & Language, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S0885230821000711)\]
* Language Model Priming for Cross-Lingual Event Extraction (AAAI 2022) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21307)\]
* PRAM: An End-to-end Prototype-based Representation Alignment Model for Zero-resource Cross-lingual Named Entity Recognition (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.201.pdf)\]
* Retrieving Relevant Context to Align Representations for Cross-lingual Event Detection (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.135.pdf)\]
* Hybrid Knowledge Transfer for Improved Cross-Lingual Event Detection via Hierarchical Sample Selection (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.296.pdf)\]


### Auxiliary Knowledge Enhancement

#### (1) Textual Knowledge (Type-related Knowledge & Synthesized Data)
* Zero-Shot Relation Extraction via Reading Comprehension (CoNLL 2017) \[[paper](https://aclanthology.org/K17-1034.pdf)\]
* Zero-Shot Open Entity Typing as Type-Compatible Grounding (EMNLP 2018) \[[paper](https://aclanthology.org/D18-1231.pdf)\]
* Description-Based Zero-shot Fine-Grained Entity Typing (NAACL 2019) \[[paper](https://aclanthology.org/N19-1087.pdf)\]
* Improving Event Detection via Open-domain Trigger Knowledge (ACL 2020) \[[paper](https://aclanthology.org/2020.acl-main.522.pdf)\]
* ZS-BERT: Towards Zero-Shot Relation Extraction with Attribute Representation Learning (NAACL 2021) \[[paper](https://aclanthology.org/2021.naacl-main.272.pdf)\]
* MapRE: An Effective Semantic Mapping Approach for Low-resource Relation Extraction (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.212.pdf)\]
* Distilling Discrimination and Generalization Knowledge for Event Detection via Delta-Representation Learning (ACL 2021) \[[paper](https://aclanthology.org/P19-1429.pdf)\]
* MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER  (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.160.pdf)\]
* Mask-then-Fill: A Flexible and Effective Data Augmentation Framework for Event Extraction (EMNLP 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-emnlp.332.pdf)\]
* Low-Resource NER by Data Augmentation With Prompting (IJCAI 2022) [[paper](https://www.ijcai.org/proceedings/2022/0590.pdf)\]
* ACLM: A Selective-Denoising based Generative Data Augmentation Approach for Low-Resource Complex NER (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.8.pdf)\]
* Entity-to-Text based Data Augmentation for various Named Entity Recognition Tasks (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.578.pdf)\]
* Improving Low-resource Named Entity Recognition with Graph Propagated Data Augmentation (ACL 2023, Short) \[[paper](https://aclanthology.org/2023.acl-short.11.pdf)\]
* GDA: Generative Data Augmentation Techniques for Relation Extraction Tasks (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.649.pdf)\]
* Generating Labeled Data for Relation Extraction: A Meta Learning Approach with Joint GPT-2 Training (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.727.pdf)\]
* RE-Matching: A Fine-Grained Semantic Matching Method for Zero-Shot Relation Extraction (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.369.pdf)\]
* S2ynRE: Two-stage Self-training with Synthetic Data for Low-resource Relation Extraction  (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.455.pdf)\]
* Enhancing Few-shot NER with Prompt Ordering based Data Augmentation (arXiv, 2023) [[paper](https://arxiv.org/abs/2305.11791)\]
* STAR: Boosting Low-Resource Event Extraction by Structure-to-Text Data Generation with Large Language Models (arXiv, 2023) [[paper](https://arxiv.org/abs/2305.15090)\]
* SegMix: A Simple Structure-Aware Data Augmentation Method (arXiv, 2023) \[[paper](https://arxiv.org/abs/2311.09505)\]
* GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction (arXiv, 2023) \[[paper](https://arxiv.org/abs/2310.03668)\]

#### (2) Structured Knowledge (KG & Ontology & Logical Rules)
* Leveraging FrameNet to Improve Automatic Event Detection (ACL 2016) \[[paper](https://aclanthology.org/P16-1201.pdf)\]
* DOZEN: Cross-Domain Zero Shot Named Entity Recognition with Knowledge Graph (SIGIR 2021) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463113)\]
* Connecting the Dots: Event Graph Schema Induction with Path Language Modeling (EMNLP 2020) \[[paper](https://aclanthology.org/2020.emnlp-main.50.pdf)\]
* Logic-guided Semantic Representation Learning for Zero-Shot Relation Classification (COLING 2020) \[[paper](https://aclanthology.org/2020.coling-main.265.pdf)\]
* NERO: A Neural Rule Grounding Framework for Label-Efficient Relation Extraction (WWW 2020) \[[paper](https://dl.acm.org/doi/10.1145/3366423.3380282)\]
* Knowledge-aware Named Entity Recognition with Alleviating Heterogeneity (AAAI 2021) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17603)\]
* OntoED: Low-resource Event Detection with Ontology Embedding (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.220.pdf)\]
* Low-resource Extraction with Knowledge-aware Pairwise Prototype Learning (Knowledge-Based Systems, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S0950705121008467)\]
<!--* Neuralizing Regular Expressions for Slot Filling (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.747.pdf)\]-->


## 2 Developing Stronger Data-Efficient Models

### Meta Learning

#### For Low-Resource NER
* Few-shot Classification in Named Entity Recognition Task (SAC 2019) \[[paper](https://dl.acm.org/doi/10.1145/3297280.3297378)\]
* Enhanced Meta-Learning for Cross-Lingual Named Entity Recognition with Minimal Resources (AAAI 2020) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/64667)\]
* MetaNER: Named Entity Recognition with Meta-Learning (WWW 2020) \[[paper](https://dl.acm.org/doi/10.1145/3366423.3380127)\]
* Meta-Learning for Few-Shot Named Entity Recognition (MetaNLP, 2021) \[[paper](https://aclanthology.org/2021.metanlp-1.6.pdf)\]
* Decomposed Meta-Learning for Few-Shot Named Entity Recognition (ACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-acl.124.pdf)\]
* Label Semantics for Few Shot Named Entity Recognition (ACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-acl.155.pdf)\]
* Few-Shot Named Entity Recognition via Meta-Learning (TKDE, 2022) \[[paper](https://doi.org/10.1109/TKDE.2020.3038670)\]
* Prompt-Based Metric Learning for Few-Shot NER (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.451.pdf)\]
* Task-adaptive Label Dependency Transfer for Few-shot Named Entity Recognition (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.203.pdf)\]
* HEProto: A Hierarchical Enhancing ProtoNet based on Multi-Task Learning for Few-shot Named Entity Recognition (CIKM 2023) \[[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614908)\] 
* Meta-Learning Triplet Network with Adaptive Margins for Few-Shot Named Entity Recognition (arXiv, 2023) \[[paper](https://arxiv.org/abs/2302.07739)\]
* Causal Interventions-based Few-Shot Named Entity Recognition (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.01914)\]
* MCML: A Novel Memory-based Contrastive Meta-Learning Method for Few Shot Slot Tagging (arXiv, 2023) \[[paper](https://arxiv.org/abs/2108.11635)\]

#### For Low-Resource RE
* Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification (AAAI 2019) \[[paper](https://ojs.aaai.org//index.php/AAAI/article/view/4604)\]
* Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs (ICML 2020) \[[paper](http://proceedings.mlr.press/v119/qu20a/qu20a.pdf)\]
* Bridging Text and Knowledge with Multi-Prototype Embedding for Few-Shot Relational Triple Extraction (COLING 2020) \[[paper](https://aclanthology.org/2020.coling-main.563.pdf)\]
* Meta-Information Guided Meta-Learning for Few-Shot Relation Classification (COLING 2020) \[[paper](https://aclanthology.org/2020.coling-main.140.pdf)\]
* Prototypical Representation Learning for Relation Extraction (ICLR 2021)  \[[paper](https://openreview.net/forum?id=aCgLmfhIy_f)\]
* Pre-training to Match for Unified Low-shot Relation Extraction (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.397.pdf)\]
* Learn from Relation Information: Towards Prototype Representation Rectification for Few-Shot Relation Extraction (NAACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-naacl.139.pdf)\]
* fmLRE: A Low-Resource Relation Extraction Model Based on Feature Mapping Similarity Calculation (AAAI 2023) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26605)\]
* Interaction Information Guided Prototype Representation Rectification for Few-Shot Relation Extraction (Electronics, 2023) \[[paper](https://www.mdpi.com/2079-9292/12/13/2912)\]
* Consistent Prototype Learning for Few-Shot Continual Relation Extraction (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.409.pdf)\]
* RAPL: A Relation-Aware Prototype Learning Approach for
Few-Shot Document-Level Relation Extraction (EMNLP 2023) \[[paper](https://arxiv.org/abs/2310.15743)\]
* Improving few-shot relation extraction through semantics-guided learning (Neural Networks, 2023) \[[paper](https://www.sciencedirect.com/science/article/pii/S0893608023006196)\]
* Generative Meta-Learning for Zero-Shot Relation Triplet Extraction (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.01920)\]

#### For Low-Resource EE
* Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection (WSDM 2020) \[[paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371796)\]
* Adaptive Knowledge-Enhanced Bayesian Meta-Learning for Few-shot Event Detection (ACL 2021, Findings) \[[paper](https://aclanthology.org/2021.findings-acl.214.pdf)\]
* Few-Shot Event Detection with Prototypical Amortized Conditional Random Field (ACL 2021, Findings) \[[paper](https://aclanthology.org/2021.findings-acl.3.pdf)\]
* Zero- and Few-Shot Event Detection via Prompt-Based Meta Learning (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.440.pdf)\]
* MultiPLe: Multilingual Prompt Learning for Relieving Semantic Confusions in Few-shot Event Detection (CIKM 2023) \[[paper](https://dl.acm.org/doi/abs/10.1145/3583780.3614984)\]


### Transfer Learning

* Zero-Shot Transfer Learning for Event Extraction (ACL 2018) \[[paper](https://aclanthology.org/P18-1201.pdf)\]
* Transfer Learning for Named-Entity Recognition with Neural Networks (LREC 2018) \[[paper](https://aclanthology.org/L18-1708.pdf)\]
* Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks (NAACL 2019) \[[paper](https://aclanthology.org/N19-1306.pdf)\]
* Relation Adversarial Network for Low Resource Knowledge Graph Completion (WWW 2020) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3366423.3380089)\]
* MZET: Memory Augmented Zero-Shot Fine-grained Named Entity Typing (COLING 2020) \[[paper](https://aclanthology.org/2020.coling-main.7.pdf)\]
* LearningToAdapt with Word Embeddings: Domain Adaptation of Named Entity Recognition Systems (Information Processing and Management, 2021) \[[paper](https://www.sciencedirect.com/science/article/pii/S0306457321000455)\]
* MANNER: A Variational Memory-Augmented Model for Cross Domain Few-Shot Named Entity Recognition (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.234.pdf)\]
* Linguistic Representations for Fewer-shot Relation Extraction across Domains (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.414.pdf)\]
* Few-Shot Relation Extraction With Dual Graph Neural Network Interaction (TNNLS, 2023) \[[paper](https://ieeexplore.ieee.org/document/10143375)\]
* Leveraging Open Information Extraction for Improving Few-Shot Trigger Detection Domain Transfer (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.14163)\]


### Fine-Tuning PLM
* Matching the Blanks: Distributional Similarity for Relation Learning (ACL 2019) \[[paper](https://aclanthology.org/P19-1279.pdf)\]
* Exploring Pre-trained Language Models for Event Extraction and Generation (ACL 2019) \[[paper](https://aclanthology.org/P19-1522.pdf)\]
* Coarse-to-Fine Pre-training for Named Entity Recognition (EMNLP 2020) \[[paper](https://aclanthology.org/2020.emnlp-main.514.pdf)\]
* CLEVE: Contrastive Pre-training for Event Extraction (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.491.pdf)\]
* Unleash GPT-2 Power for Event Detection (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.490.pdf)\]
* Efficient Zero-shot Event Extraction with Context-Definition Alignment (EMNLP 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-emnlp.531.pdf)\]
* Few-shot Named Entity Recognition with Self-describing Networks (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.392.pdf)\]
* Query and Extract: Refining Event Extraction as Type-oriented Binary Decoding (ACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-acl.16.pdf)\]
* Unleashing Pre-trained Masked Language Model Knowledge for Label Signal Guided Event Detection (DASFAA 2023) \[[paper](https://link.springer.com/chapter/10.1007/978-3-031-30675-4_42)\]
* A Multi-Task Semantic Decomposition Framework with
Task-specific Pre-training for Few-Shot NER (CIKM 2023) \[[paper](https://arxiv.org/abs/2308.14533)\]
* Continual Contrastive Finetuning Improves Low-Resource Relation Extraction (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.739.pdf)\]
* Unified Low-Resource Sequence Labeling by Sample-Aware Dynamic Sparse Finetuning (EMNLP 2023) \[[paper](https://arxiv.org/abs/2311.03748)\]
* GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer (arXiv, 2023) \[[paper](https://arxiv.org/abs/2311.08526)\]



## 3 Optimizing Data and Models Together

### Multi-Task Learning

#### (1) IE & IE-Related Tasks

**NER, Named Entity Normalization (NEN)**

* A Neural Multi-Task Learning Framework to Jointly Model Medical Named Entity Recognition and Normalization (AAAI 2019) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/3861)\]
* MTAAL: Multi-Task Adversarial Active Learning for Medical Named Entity Recognition and Normalization (AAAI 2021) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17714)\]
* An End-to-End Progressive Multi-Task Learning Framework for Medical Named Entity Recognition and Normalization (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.485.pdf)\]

**Word Sense Disambiguation (WSD), Event Detection (ED)**

* Similar but not the Same: Word Sense Disambiguation Improves Event Detection via Neural Representation Matching (EMNLP 2018) \[[paper](https://aclanthology.org/D18-1517.pdf)\]
* Graph Learning Regularization and Transfer Learning for Few-Shot Event Detection (SIGIR 2021) \[[paper](https://dl.acm.org/doi/pdf/10.1145/3404835.3463054)\]

#### (2) Joint IE & Other Structured Prediction Tasks

**NER, RE**

* GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction (ACL 2019) \[[paper](https://aclanthology.org/P19-1136.pdf)\]
* CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning (AAAI 2020) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6495)\]
* Joint Entity and Relation Extraction Model based on Rich Semantics (Neurocomputing, 2021) \[[paper](https://www.sciencedirect.com/science/article/pii/S0925231220319378?casa_token=jzgLW9J1UKoAAAAA:5vnqqGKt0_-ykbhTp15Bq8mB-8B50cM3LDa10q2h8yc4q4AJVfeEbQV_fyMo2Z92xjl3HPNt6w)\]

**NER, RE, EE**

* Entity, Relation, and Event Extraction with Contextualized Span Representations (EMNLP 2019) \[[paper](https://aclanthology.org/D19-1585.pdf)\]

**NER, RE, EE  & Other Structured Prediction Tasks**

* SPEECH: Structured Prediction with Energy-Based Event-Centric Hyperspheres (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.21.pdf)\]
* Mirror: A Universal Framework for Various Information Extraction Tasks (EMNLP 2023) \[[paper](https://arxiv.org/abs/2311.05419)\]


### Task Reformulation
* Zero-Shot Relation Extraction via Reading Comprehension (CoNLL 2017) \[[paper](https://aclanthology.org/K17-1034.pdf)\]
* Entity-Relation Extraction as Multi-Turn Question Answering (ACL 2019) \[[paper](http://aclanthology.lst.uni-saarland.de/P19-1129.pdf)\]
* A Unified MRC Framework for Named Entity Recognition (ACL 2020) \[[paper](https://aclanthology.org/2020.acl-main.519.pdf)\]
* Event Extraction as Machine Reading Comprehension (EMNLP 2020) \[[paper](https://aclanthology.org/2020.emnlp-main.128.pdf)\]
* Event Extraction by Answering (Almost) Natural Questions (EMNLP 2020) \[[paper](https://aclanthology.org/2020.emnlp-main.49.pdf)\]
* Text2Event: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction (ACL 2021) \[[paper](https://aclanthology.org/2021.acl-long.217/)\]
* Structured Prediction as Translation between Augmented Natural Languages (ICLR 2021) \[[paper](https://openreview.net/forum?id=US-TP-xnXI)\]
* Learning to Ask for Data-Efficient Event Argument Extraction (AAAI 2022, Student Abstract) \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21686)\]
* Complex Question Enhanced Transfer Learning for Zero-shot Joint Information Extraction (TASLP, 2023) \[[paper](https://ieeexplore.ieee.org/abstract/document/10214665)\]
* Weakly-Supervised Questions for Zero-Shot Relation Extraction (EACL 2023) \[[paper](https://aclanthology.org/2023.eacl-main.224.pdf)\]
* Event Extraction as Question Generation and Answering (ACL 2023, Short) \[[paper](https://aclanthology.org/2023.acl-short.143.pdf)\]

### Prompt-Tuning PLM
#### (1) Vanilla Prompt-Tuning
* Template-Based Named Entity Recognition Using BART (ACL 2021, Findings) \[[paper](https://aclanthology.org/2021.findings-acl.161.pdf)\]
* Label Verbalization and Entailment for Effective Zero and Few-Shot Relation Extraction (EMNLP 2021) \[[paper](https://aclanthology.org/2021.emnlp-main.92.pdf)\]
* LightNER: A Lightweight Tuning Paradigm for Low-resource NER via Pluggable Prompting (COLING 2022) \[[paper](https://aclanthology.org/2022.coling-1.209.pdf)\]
* COPNER: Contrastive Learning with Prompt Guiding for Few-shot Named Entity Recognition (COLING 2022) \[[paper](https://aclanthology.org/2022.coling-1.222.pdf)\]
* Template-free Prompt Tuning for Few-shot NER (NAACL 2022) \[[paper](https://aclanthology.org/2022.naacl-main.420.pdf)\]
* RelationPrompt: Leveraging Prompts to Generate Synthetic Data for Zero-Shot Relation Triplet Extraction (ACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-acl.5.pdf)\]
* Prompt for Extraction? PAIE: Prompting Argument Interaction for Event Argument Extraction (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.466.pdf)\]
* Dynamic Prefix-Tuning for Generative Template-based Event Extraction (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.358.pdf)\]
* Prompt-Learning for Cross-Lingual Relation Extraction (IJCNN 2023) \[[paper](https://arxiv.org/abs/2304.10354)\]
* DSP: Discriminative Soft Prompts for Zero-Shot Entity and Relation Extraction (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.339.pdf)\]
* Contextualized Soft Prompts for Extraction of Event Arguments (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.266.pdf)\]
* The Art of Prompting: Event Detection based on Type Specific Prompts (ACL 2023, Short) \[[paper](https://aclanthology.org/2023.acl-short.111.pdf)\]

#### (2) Augmented Prompt-Tuning
* PTR: Prompt Tuning with Rules for Text Classification (AI Open, 2022) \[[paper](https://www.sciencedirect.com/science/article/pii/S2666651022000183)\]
* KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction (WWW 2022) \[[paper](https://dl.acm.org/doi/abs/10.1145/3485447.3511998)\]
* Ontology-enhanced Prompt-tuning for Few-shot Learning (WWW 2022) \[[paper](https://dl.acm.org/doi/10.1145/3485447.3511921)\]
* Relation Extraction as Open-book Examination: Retrieval-enhanced Prompt Tuning (SIGIR 2022, Short) \[[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531746)\]
* Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning (NeurIPS 2022) \[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/97011c648eda678424f9292dadeae72e-Paper-Conference.pdf)\]
* AugPrompt: Knowledgeable Augmented-Trigger Prompt for Few-Shot Event Classification (Information Processing & Management, 2022) \[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0306457322002540)\]
* Zero-Shot Event Detection Based on Ordered Contrastive Learning and Prompt-Based Prediction (NAACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-naacl.196.pdf)\]
* DEGREE: A Data-Efficient Generation-Based Event Extraction Model (NAACL 2022) \[[paper](https://aclanthology.org/2022.naacl-main.138.pdf)\]
* Retrieval-Augmented Generative Question Answering for Event Argument Extraction (EMNLP 2022) \[[paper](https://aclanthology.org/2022.emnlp-main.307.pdf)\]
* Unified Structure Generation for Universal Information Extraction (ACL 2022) \[[paper](https://aclanthology.org/2022.acl-long.395.pdf)\]
* LasUIE: Unifying Information Extraction with Latent Adaptive Structure-aware Generative Language Model (NeurIPS 2022) \[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/63943ee9fe347f3d95892cf87d9a42e6-Abstract-Conference.html)\]
* Universal Information Extraction as Unified Semantic Matching (AAAI 2023) \[[paper](https://arxiv.org/abs/2301.03282)\]
* Universal Information Extraction with Meta-Pretrained Self-Retrieval (ACL 2023) \[[paper](https://aclanthology.org/2023.findings-acl.251.pdf)\]
* RexUIE: A Recursive Method with Explicit Schema Instructor for Universal Information Extraction (EMNLP 2023, Findings) \[[paper](https://arxiv.org/abs/2304.14770)\]
* Schema-aware Reference as Prompt Improves Data-Efficient Relational Triple and Event Extraction (SIGIR 2023) \[[paper](https://arxiv.org/abs/2210.10709)\]
* PromptNER: Prompt Locating and Typing for Named Entity Recognition (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.698.pdf)\]
* Focusing, Bridging and Prompting for Few-shot Nested Named Entity Recognition (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.164.pdf)\]
* Revisiting Relation Extraction in the era of Large Language Models (ACL 2023) [[paper](https://aclanthology.org/2023.acl-long.868.pdf)\]
* AMPERE: AMR-Aware Prefix for Generation-Based Event Argument Extraction Model (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.615.pdf)\]
* BertNet: Harvesting Knowledge Graphs with Arbitrary Relations from Pretrained Language Models (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.309.pdf)\]
* Retrieve-and-Sample: Document-level Event Argument Extraction via Hybrid Retrieval Augmentation (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.17.pdf)\]
* Easy-to-Hard Learning for Information Extraction (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.754.pdf)\]
* DemoSG: Demonstration-enhanced Schema-guided Generation for Low-resource Event Extraction (EMNLP 2023, Findings) \[[paper](https://arxiv.org/abs/2310.10481)\]
* Template-Free Prompting for Few-Shot Named Entity Recognition via Semantic-Enhanced Contrastive Learning (TNNLS, 2023) \[[paper](https://ieeexplore.ieee.org/abstract/document/10264144)\]
* TaxonPrompt: Taxonomy-Aware Curriculum Prompt Learning for Few-Shot Event Classification (KBS, 2023) \[[paper](https://www.sciencedirect.com/science/article/pii/S0950705123000400)\]
* A Composable Generative Framework based on Prompt Learning for Various Information Extraction Tasks (IEEE Transactions on Big Data, 2023) \[[paper](https://ieeexplore.ieee.org/abstract/document/10130644)\]
* Event Extraction With Dynamic Prefix Tuning and Relevance Retrieval (TKDE, 2023) \[[paper](https://doi.org/10.1109/TKDE.2023.3266495)\]
* MsPrompt: Multi-step Prompt Learning for Debiasing Few-shot Event Detection (Information Processing & Management, 2023) \[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0306457323002467)\]
* PromptNER: A Prompting Method for Few-shot Named Entity Recognition via k Nearest Neighbor Search (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.12217)\]
* TKDP: Threefold Knowledge-enriched Deep Prompt Tuning for Few-shot Named Entity Recognition (arXiv, 2023) \[[paper](https://arxiv.org/abs/2306.03974)\]
* OntoType: Ontology-Guided Zero-Shot Fine-Grained Entity Typing with Weak Supervision from Pre-Trained Language Models  (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.12307)\]


## 🍏 LLM-Based Methods 🍏

## Direct Inference Without Tuning

### Instruction Prompting
* Exploring the Feasibility of ChatGPT for Event Extraction (arXiv, 2023) [[paper](https://arxiv.org/abs/2303.03836)\]
* Zero-Shot Information Extraction via Chatting with ChatGPT (arXiv, 2023) \[[paper](https://arxiv.org/abs/2302.10205)\]
* Global Constraints with Prompting for Zero-Shot Event Argument Classification (EACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-eacl.191.pdf)\]
* Revisiting Large Language Models as Zero-shot Relation Extractors (EMNLP 2023, Findings) \[[paper](https://arxiv.org/abs/2310.05028)\]
* Evaluating ChatGPT's Information Extraction Capabilities: An Assessment of Performance, Explainability, Calibration, and Faithfulness (arXiv, 2023) \[[paper](https://arxiv.org/abs/2304.11633)\] 
* LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.13168)\]

### Code Prompting
* Code4Struct: Code Generation for Few-Shot Event Structure Prediction (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.202.pdf)\]
* CodeIE: Large Code Generation Models are Better Few-Shot Information Extractors (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.855.pdf)\]
* ViStruct: Visual Structural Knowledge Extraction via Curriculum Guided Code-Vision Representation (EMNLP 2023) \[[paper](https://arxiv.org/abs/2311.13258)\]
* Retrieval-Augmented Code Generation for Universal Information Extraction (arXiv, 2023) [[paper](https://arxiv.org/abs/2311.02962)\]

### In-Context Learning
* Learning In-context Learning for Named Entity Recognition (ACL 2023) \[[paper](https://aclanthology.org/2023.acl-long.764.pdf)\]
* How to Unleash the Power of Large Language Models for Few-shot Relation Extraction? (ACL 2023, SustaiNLP Workshop) [[paper](https://aclanthology.org/2023.sustainlp-1.13.pdf)\]
* GPT-RE: In-context Learning for Relation Extraction using Large Language Models (EMNLP 2023) [[paper](https://arxiv.org/abs/2305.02105)\]
* Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples! (EMNLP 2023, Findings) \[[paper](https://arxiv.org/abs/2303.08559)\]
* Guideline Learning for In-Context Information Extraction (EMNLP 2023) [[paper](https://arxiv.org/abs/2310.05066)\]
* GPT-NER: Named Entity Recognition via Large Language Models (arXiv, 2023) [[paper](https://arxiv.org/abs/2304.10428)\]
* In-Context Few-Shot Relation Extraction via Pre-Trained Language Models (arXiv, 2023) [[paper](https://arxiv.org/abs/2310.11085)\]
* Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models (arXiv, 2023) [[paper](https://arxiv.org/abs/2311.08921)\]
* Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors (arXiv, 2023) \[[paper](https://arxiv.org/abs/2305.14450)\]
* GPT Struct Me: Probing GPT Models on Narrative Entity Extraction (arXiv, 2023) [[paper](https://arxiv.org/abs/2311.14583)\]
* Improving Open Information Extraction with Large Language Models: A Study on Demonstration Uncertainty (arXiv, 2023) \[[paper](https://arxiv.org/abs/2309.03433)\]
* LOKE: Linked Open Knowledge Extraction for Automated Knowledge Graph Construction (arXiv, 2023) \[[paper](https://arxiv.org/abs/2311.09366)\]

<!--### Retrieval-Augmented Prompting-->



## Model Specialization With Tuning

### Prompt-Tuning LLM
* DeepStruct: Pretraining of Language Models for Structure Prediction (ACL 2022, Findings) \[[paper](https://aclanthology.org/2022.findings-acl.67v2.pdf)\]
* Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors (ACL 2023, Findings) \[[paper](https://aclanthology.org/2023.findings-acl.50.pdf)\]
* Instruct and Extract: Instruction Tuning for On-Demand Information Extraction (EMNLP 2023) \[[paper](https://arxiv.org/abs/2310.16040)\]
* UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition (arXiv, 2023) [[paper](https://arxiv.org/abs/2308.03279)\]
* InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction (arXiv, 2023) \[[paper](https://arxiv.org/abs/2304.08085)\]

### Fine-Tuning LLM
* Fine-Tuning GPT Family (OpenAI, 2023) \[[Documentation](https://platform.openai.com/docs/guides/fine-tuning)\]



## How to Cite

📋 Thank you very much for your interest in our survey work. If you use or extend our survey, please cite the following paper:

```bibtex
@misc{2023_LowResIE,
    author    = {Shumin Deng and
                 Yubo Ma and
                 Ningyu Zhang and
                 Yixin Cao and
                 Bryan Hooi},
    title     = {Information Extraction in Low-Resource Scenarios: Survey and Perspective}, 
    journal   = {CoRR},
    volume    = {abs/2202.08063},
    year      = {2023},
    url       = {https://arxiv.org/abs/2202.08063}
}
```
<!--inproceedings-->
