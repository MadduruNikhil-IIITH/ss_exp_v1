# Academic References & Literature Notes

This document contains key citations, links, and detailed notes on relevant published work for our SQuAD sentence salience experiment.

---

## 1. Adaptations of SQuAD to Answer Sentence Selection (AS2)

### Citation 1: Modeling Context in Answer Sentence Selection Systems
- **Title**: Modeling Context in Answer Sentence Selection Systems on a Latency Budget
- **Authors**: EACL 2021 / Amazon Science
- **Links**:
  - [arXiv:2102.04351](https://arxiv.org/abs/2102.04351)
  - [GitHub: amazon-science/wqa-squad2-coala](https://github.com/amazon-science/wqa-squad2-coala)
- **Key Concepts**:
  - Outlines the methodology to convert span-based reading comprehension datasets like SQuAD 2.0 into Answer Sentence Selection (AS2) datasets.
  - Proposes the **COALA** model, which utilizes both local context (neighboring sentences) and global context (paragraph representation) to rank sentences.
- **Our Extension**: 
  - Instead of using a simple sliding window of $N$ sentences for local context, we can use the **RST subtree neighborhood** (e.g., selecting the sentence's discourse siblings and parents) as a linguistically motivated context window.

### Citation 2: Multi-Perspective Graph Encoding
- **Title**: Capturing Sentence Relations for Answer Sentence Selection with Multi-Perspective Graph Encoding
- **Authors**: AAAI (Association for the Advancement of Artificial Intelligence)
- **Links**:
  - [AAAI Publication](https://ojs.aaai.org/index.php/AAAI/article/view/6389)
- **Key Concepts**:
  - Moves away from modeling candidate sentences independently. Instead, it constructs a graph over sentences in a paragraph to capture inter-sentence relations and transitions, evaluating on SQuAD and WikiQA.
- **Our Extension**:
  - Rather than a generic fully connected graph, we can use the **RST Discourse Tree** to build a **Rhetorical Discourse Graph (RDG)**. Nodes are sentences, and directed, typed edges represent RST relations (e.g., *Cause*, *Contrast*). Running Graph Neural Networks (GNNs) on this RDG propagates query relevance scores along rhetorical paths.

---

## 2. Sentence Selection for Question Generation (QG)

### Citation 3: Sentence Selection for Question Generation
- **Title**: Learning to Generate Questions by Enhancing Text Generation with Sentence Selection
- **Authors**: arXiv / EMNLP workshop
- **Links**:
  - [arXiv:2104.09341](https://arxiv.org/abs/2104.09341)
- **Key Concepts**:
  - Implements a selector module to identify the most salient context sentences from a passage, then feeds only those selected sentences to a generative Question Generation (QG) model.
  - Demonstrates that selecting high-salience sentences first leads to cleaner, more natural, and less redundant questions.
- **Our Extension**:
  - We can integrate our **DSNB** (Discourse-Semantic Neighborhood Balancing) and **Heuristic-Guided BERT** selectors directly into their QG pipeline, showing that our discourse-aware training methods yield higher precision sentence selection, reducing downstream QG "garbage" generations.

---

## 3. Latest QA & Sentence Selection Advances (2024–2026)

### Citation 4: Context Convergence for Inferential Questions (Latest 2026 Paper)
- **Title**: Context Convergence Improves Answering Inferential Questions
- **Authors**: Jamshid Mozafari, Bhawna Piryani, Adam Jatowt
- **Links**:
  - [arXiv:2605.12370](https://arxiv.org/abs/2605.12370)
  - [GitHub: jamshidmozafari/context-convergence](https://github.com/jamshidmozafari/context-convergence)
- **Key Concepts**:
  - Investigates why LLMs fail at multi-hop inferential questions.
  - Argues that **convergence**—the degree to which individual sentences narrow down potential answer options—is a critical metric for passage construction.
- **Our Extension**:
  - We can use our discourse tree features to measure "rhetorical convergence". Sentences acting as causal/contrastive junctions inside the RST tree provide structural convergence paths that help LLMs answer multi-hop inferential questions.

### Citation 5: FastFiD Inference Efficiency (2024)
- **Title**: FastFiD: Improve Inference Efficiency of Open Domain Question Answering via Sentence Selection
- **Authors**: Huang et al. (Tsinghua University)
- **Links**:
  - [arXiv:2408.06333](https://arxiv.org/abs/2408.06333)
  - [GitHub: thunlp/FastFiD](https://github.com/thunlp/FastFiD)
- **Key Concepts**:
  - Uses an efficient sentence selection mechanism over encoded passages to prune non-salient contexts in open-domain QA, speeding up the generation phase.
- **Our Extension**:
  - Injecting our low-dimensional **RST Heuristic Prior** into the FastFiD sentence selector ensures that context pruning respects rhetorical boundaries, avoiding the accidental removal of crucial explanatory sibling clauses.
