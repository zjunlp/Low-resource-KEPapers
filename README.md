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

* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()


### Weakly Supervised Augmentation
* Distant Supervision for Relation Extraction without Labeled Data (ACL 2009) [[paper]](https://dl.acm.org/doi/pdf/10.5555/1690219.1690287)
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
#### Text
* Improving Event Detection via Open-domain Trigger Knowledge (ACL 2020) [[paper]](https://aclanthology.org/2020.acl-main.522.pdf)


#### KG
* DOZEN: Cross-Domain Zero Shot Named Entity Recognition with Knowledge Graph (SIGIR 2021) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3404835.3463113)
* Leveraging FrameNet to Improve Automatic Event Detection (ACL 2016) [[paper]](https://aclanthology.org/P16-1201.pdf)


#### Ontology & Rule
* Logic-guided Semantic Representation Learning for Zero-Shot Relation Classification (COLING 2020) [[paper]](https://aclanthology.org/2020.coling-main.265.pdf)
* OntoED: Low-resource Event Detection with Ontology Embedding (ACL 2021) [[paper]](https://aclanthology.org/2021.acl-long.220.pdf)
* Low-resource Extraction with Knowledge-aware Pairwise Prototype Learning (Knowledge-Based Systems, 2022) [[paper]](https://www.sciencedirect.com/science/article/pii/S0950705121008467)



## 2 Exploiting Stronger Models

### Meta Learning

#### For Low-resource NER
* MetaNER: Named Entity Recognition with Meta-Learning (WWW 2020) [[paper]](https://dl.acm.org/doi/10.1145/3366423.3380127)


#### For Low-resource RE
* Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs (ICML 2020) [[paper]](http://proceedings.mlr.press/v119/qu20a/qu20a.pdf)


#### For Low-resource EE
* Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection (WSDM 2020) [[paper]](https://dl.acm.org/doi/abs/10.1145/3336191.3371796)
* Adaptive Knowledge-Enhanced Bayesian Meta-Learning for Few-shot Event Detection (ACL 2021 finding) [[paper]](https://aclanthology.org/2021.findings-acl.214.pdf)


### Transfer Learning

#### Class-related Semantics
* Zero-Shot Transfer Learning for Event Extraction (ACL 2018) [[paper]](https://aclanthology.org/P18-1201.pdf)
* Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks (NAACL 2019) [[paper]](https://aclanthology.org/N19-1306.pdf)
* Relation Adversarial Network for Low Resource Knowledge Graph Completion (WWW 2020) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3366423.3380089)

#### Pre-trained Language Representations
* Matching the Blanks: Distributional Similarity for Relation Learning (ACL 2019) [[paper]](https://aclanthology.org/P19-1279.pdf)
* Coarse-to-Fine Pre-training for Named Entity Recognition (EMNLP 2020) [[paper]](https://aclanthology.org/2020.emnlp-main.514.pdf)
* CLEVE: Contrastive Pre-training for Event Extraction (ACL 2021) [[paper]](https://aclanthology.org/2021.acl-long.491.pdf)


### Prompt Learning

#### Vanilla Prompt Learning
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()

#### Augmented Prompt Learning
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()


## 3 Exploiting Data and Models Together

### Multi-task Learning

#### NER, named entity normalization
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()

#### NER, RE
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()

#### NER, RE, EE
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()

#### word sense disambiguation (WSD), event detection (ED)
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()

### Formulating KE as QA and MRC
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()

### Retrieval Augmentation

#### Retrieval-based Language Models
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()

#### Few-shot Settings
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()
* () [[paper]]()

