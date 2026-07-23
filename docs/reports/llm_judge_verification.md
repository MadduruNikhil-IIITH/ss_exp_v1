# LLM-as-a-Judge Dataset Verification Report

This report validates the exact-index silver annotations of the SQuAD sentence salience dataset against zero-shot predictions from a local **`Qwen/Qwen2.5-1.5B-Instruct`** model.

## 1. Agreement Metrics
- **Sample Size**: 973 sentences (balanced: 973 salient, 973 non-salient)
- **Agreement Rate (Accuracy)**: `0.7390`
- **Cohen's Kappa Score**: `0.4136`
- **Silver Label Quality** (LLM Judge as ground truth):
  - **Precision**: `0.8179`
  - **Recall**: `0.7906`
  - **F1 Score**: `0.8040`

### Confusion Matrix
| | LLM Salient (1) | LLM Non-Salient (0) |
| --- | --- | --- |
| **Silver Salient (1)** | **TP: 521** (Agree) | **FP: 116** (Silver=1, LLM=0) |
| **Silver Non-Salient (0)** | **FN: 138** (Silver=0, LLM=1) | **TN: 198** (Agree) |

## 2. Qualitative Error Analysis

### Category A: Silver Salient (1) but LLM Non-Salient (0)
**Example 1**:
- **Question**: On what date does the Catholic Church celebrate the Feast of the Assumption? | What do Eastern Catholics call the Feast of the Assumption? | On what date of the Julian calendar do Eastern Catholics celebrate the Domition of the Theotokos? | How many Great Feasts does the Eastern Orthodox Church celebrate? | Do Protestants celebrate Marian feasts or not?
- **Sentence**: *"Protestants do not celebrate this, or any other Marian feasts."*
- **LLM Reasoning**: Judgment: No

Reasoning: The given sentence "Protestants do not celebrate this, or any other Marian feasts" does not provide the specific date on which the Catholic Church celebrates the Feast of the Assumption, nor does it give details about how Eastern Catholics refer to the Feast of

**Example 2**:
- **Question**: Being exposed to what type of pollution has been theorized to increase aggression? | The decrease in crime in New York is sometimes attributed to the decline of what street drug?
- **Sentence**: *"Others cite the end of the crack epidemic and demographic changes, including from immigration."*
- **LLM Reasoning**: Judgment: No

The given sentence does not directly address either of the two questions posed in the passage. It discusses various theories about factors contributing to the decrease in crime in New York but does not provide specific information about the types of pollution or drugs involved.

**Example 3**:
- **Question**: What is used to figure the relative strengh of gravity? | Who made the first to measure value of the Newton Universal Gravitation Constant? | When was the first measurement of the value of the Newton Universal Gravitation Constant? | Who figured out that his law of gravity had to be universal?
- **Sentence**: *"Succinctly stated, Newton's Law of Gravitation states that the force on a spherical object of mass  due to the gravitational pull of mass  is"*
- **LLM Reasoning**: Judgment: No

The sentence does not directly answer any of the questions about what is used to figure the relative strength of gravity, who made the first measurement, when the measurement was made, or who figured out that his law of gravity had to be universal. It instead provides additional details about Newton

**Example 4**:
- **Question**: What factor forced the French to retreat to Metz? | At Metz, what was the approximate number of Prussian troops? | Who formed the new French Army of Chalons? | Along with the Army of the Meuse, what other army did Moltke march north with? | How many casualties did Moltke suffer at Beaumont?
- **Sentence**: *"After a sharp fight in which they lost 5,000 men and 40 cannons, the French withdrew toward Sedan."*
- **LLM Reasoning**: Judgment: No

The sentence describes a battle outcome but does not directly state why the French retreated to Metz or provide specific numbers about the Prussian forces at Metz. It focuses more on the aftermath of the battle rather than explaining the initial reason for their retreat.

**Example 5**:
- **Question**: Who discovered X-Rays? | Who won the first Nobel Prize in Physics? | In what year, was the first Nobel Prize in Physics won? | Where is the Walhalla temple located? | What two people were important to the creation of modern physics?
- **Sentence**: *"The work of David Hilbert and Max Planck was crucial to the foundation of modern physics, which Werner Heisenberg and Erwin Schrödinger developed further."*
- **LLM Reasoning**: Judgment: No

Reasoning: The given sentence does not provide any direct answers to the specific questions asked about who discovered X-rays, who won the first Nobel Prize in Physics, when this award was given, where the Walhalla temple is located, or the importance of David Hilbert,

### Category B: Silver Non-Salient (0) but LLM Salient (1)
**Example 1**:
- **Question**: Which metro area is largest in terms of population? | What was it's total population back in 1900?
- **Sentence**: *"The population of the city of A Coruña in 1900 was 43,971."*
- **LLM Reasoning**: Judgment: Yes

Reasoning: The sentence directly states the population of A Coruña in 1900, which answers both parts of the question about the largest metro area and its population size.

**Example 2**:
- **Question**: How much did Spectre make in its first week? | Which movie previously had the highest earnings for its first seven days? | What movie did Spectre unseat as having the highest earnings for IMAX screenings in the UK? | How much money did Spectre gross on it's opening day in the UK? | What film previously held the UK record for highest first week opening gross? | Spectre passed which movie to become the highest grossing IMAX release in the UK? | Spectre passed which movie to have the largest per screen opening average?
- **Sentence**: *"In the United Kingdom, the film grossed £4.1 million ($6.4 million) from its Monday preview screenings."*
- **LLM Reasoning**: Judgment: Yes

The sentence "In the United Kingdom, the film grossed £4.1 million ($6.4 million) from its Monday preview screenings" directly states the amount that Spectre made in its first week, making it relevant to answering the questions about the film's performance in

**Example 3**:
- **Question**: How many actors have played Doctor Who? | Who followed Matt Smith in the role of Doctor Who? | Which episode was Matt Smith's final one as Doctor Who? | When does Doctor Who transition to a new body? | In addition to a new body, what else changes about the Doctor?
- **Sentence**: *"Each actor's portrayal differs, but they are all intended to be aspects of the same character and form part of the same storyline."*
- **LLM Reasoning**: Judgment: Yes

This sentence directly states that there are twelve actors who have played the Doctor, providing essential information needed to answer the question about how many actors have played Doctor Who.

**Example 4**:
- **Question**: Whose idea was 'tabula rasa'? | What religion was Al Farabi? | What does al-'aql al-hayulani mean? | What does al-'aql al-fa'il mean? | What is crucial for understanding?
- **Sentence**: *"For Avicenna (Ibn Sina), for example, the tabula rasa is a pure potentiality that is actualized through education, and knowledge is attained through "empirical familiarity with objects in this world from which one abstracts universal concepts" developed through a "syllogistic method of reasoning in which observations lead to propositional statements which when compounded lead to further abstract concepts"."*
- **LLM Reasoning**: Judgment: Yes

The sentence directly addresses Avicenna's concept of tabula rasa and explains its development through education and empirical observation, providing essential information relevant to answering the question about whose idea it was.

**Example 5**:
- **Question**: The UMC prohibits the celebration of what type of unions? | In what year was the trial of Rev. Jimmy Creech? | In what year was same-sex marriage legalized nationwide?
- **Sentence**: *"In April of 2016, a United Methodist bishop made a bold statement by performing a same-sex marriage in church as a clear sign of his advocacy for change."*
- **LLM Reasoning**: Judgment: Yes

Reasoning: The sentence directly states that "a United Methodist bishop made a bold statement by performing a same-sex marriage in church as a clear sign of his advocacy for change," which answers both parts of the question about what type of unions the UMC prohibits and when same-sex

