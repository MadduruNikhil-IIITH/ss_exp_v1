# LLM-as-a-Judge Dataset Verification Report

This report validates the exact-index silver annotations of the SQuAD sentence salience dataset against zero-shot predictions from a local **`Qwen/Qwen2.5-1.5B-Instruct`** model.

## 1. Agreement Metrics
- **Sample Size**: 100 sentences (balanced: 50 salient, 50 non-salient)
- **Agreement Rate (Accuracy)**: `0.9100`
- **Cohen's Kappa Score**: `0.8200`
- **Silver Label Quality** (LLM Judge as ground truth):
  - **Precision**: `0.9000`
  - **Recall**: `0.9184`
  - **F1 Score**: `0.9091`

### Confusion Matrix
| | LLM Salient (1) | LLM Non-Salient (0) |
| --- | --- | --- |
| **Silver Salient (1)** | **TP: 45** (Agree) | **FP: 5** (Silver=1, LLM=0) |
| **Silver Non-Salient (0)** | **FN: 4** (Silver=0, LLM=1) | **TN: 46** (Agree) |

## 2. Qualitative Error Analysis

### Category A: Silver Salient (1) but LLM Non-Salient (0)
**Example 1**:
- **Question**: Which political figure supported the Saar Statute referendum?
- **Sentence**: *"In the Paris Agreements of 23 October 1954, France offered to establish an independent "Saarland", under the auspices of the Western European Union (WEU), but on 23 October 1955 in the Saar Statute referendum the Saar electorate rejected this plan by 67.7% to 32.3% (out of a 96.5% turnout: 423,434 against, 201,975 for) despite the public support of Federal German Chancellor Konrad Adenauer for the plan."*
- **LLM Reasoning**: Judgment: No

The given sentence does not provide any information about which political figure supported the Saar Statute referendum. It discusses the outcome and details of the referendum itself without mentioning any specific political figures involved.

**Example 2**:
- **Question**: What plants are the goal of sequencing wild forms of cotton?
- **Sentence**: *"Once the D genome is assembled from all of this raw material, it will undoubtedly assist in the assembly of the AD genomes of cultivated varieties of cotton, but a lot of hard work remains."*
- **LLM Reasoning**: Judgment: No

Reasoning: The given sentence does not provide any information related to the plants being sequenced for wild forms of cotton. It discusses the process of assembling the D genome and its potential benefits, but does not mention anything specific about the plants being sequenced.

**Example 3**:
- **Question**: What comes before state and territorial laws in the 50 U.S states?
- **Sentence**: *"Federal law and treaties, so long as they are in accordance with the Constitution, preempt conflicting state and territorial laws in the 50 U.S. states and in the territories."*
- **LLM Reasoning**: Judgment: No

The given sentence discusses federal preemption over state and territorial laws but does not directly address what comes before them within the 50 U.S. states. It focuses more on the relationship between federal and state laws rather than specifying the order of precedence among state laws themselves.

**Example 4**:
- **Question**: What does a behaviorial component focus on when defining sexual orientation?
- **Sentence**: *"Most definitions of sexual orientation include a psychological component, such as the direction of an individual's erotic desires, or a behavioral component, which focuses on the sex of the individual's sexual partner/s. Some people prefer simply to follow an individual's self-definition or identity."*
- **LLM Reasoning**: Judgment: No

The given sentence discusses both psychological and behavioral components of sexual orientation but does not directly address what a behavioral component focuses on when defining sexual orientation. It instead shifts focus to other aspects like self-definition and identity. Therefore, this sentence is not SALIENT for answering the specific question about

**Example 5**:
- **Question**: What areas does the Brown School Library have represented scholars?
- **Sentence**: *"The school has many nationally and internationally acclaimed scholars in social security, health care, health disparities, communication, social and health policy, and individual and family development."*
- **LLM Reasoning**: Judgment: No

Reasoning: The given sentence lists various fields where scholars are located at the school but does not mention anything about the library's representation of scholars. It focuses solely on the scholarly presence across different departments without providing any information related to the library's collection or representation of scholars.

### Category B: Silver Non-Salient (0) but LLM Salient (1)
**Example 1**:
- **Question**: What is the seeming result of donating genetic cotton sequencing to the public?
- **Sentence**: *"They announced that they would donate their raw reads to the public."*
- **LLM Reasoning**: Judgment: Yes

Reasoning: The sentence "They announced that they would donate their raw reads to the public" directly answers the question by stating what happened regarding the donation of genetic cotton sequencing data to the public.

**Example 2**:
- **Question**: Along with people with diminished immune function, like HIV patients, what group of people is more likely to develop extrapulmonary TB?
- **Sentence**: *"In 15–20% of active cases, the infection spreads outside the lungs, causing other kinds of TB."*
- **LLM Reasoning**: Judgment: Yes

The sentence directly states that extrapulmonary TB occurs more commonly in immunosuppressed persons, which includes individuals with weakened immune systems such as those with HIV. This information is crucial for understanding who is more likely to develop extrapulmonary TB along with people with diminished

**Example 3**:
- **Question**: In which year did Chopin begin experiencing a serious decline in health?
- **Sentence**: *"Chopin's health continued to deteriorate, particularly from this time onwards."*
- **LLM Reasoning**: Judgment: Yes

The sentence "Chopin's health continued to deteriorate, particularly from this time onwards" directly states that Chopin began experiencing a serious decline in health starting from a certain point, providing the necessary information to answer the question about the year when this decline began.

**Example 4**:
- **Question**: What did a study done by Elizabeth Loftus and John palmer show?
- **Sentence**: *"gave higher estimates than those who were asked, "How fast were the cars going when they hit each other?""*
- **LLM Reasoning**: Judgment: Yes

The sentence "gave higher estimates than those who were asked, "How fast were the cars going when they hit each other?"" directly answers the question by providing specific details about the results of the study conducted by Elizabeth Loftus and John Palmer.

