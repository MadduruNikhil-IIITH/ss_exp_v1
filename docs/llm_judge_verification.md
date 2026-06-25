# LLM-as-a-Judge Dataset Verification Report

This report validates the exact-index silver annotations of the SQuAD sentence salience dataset against zero-shot predictions from a local **`Qwen/Qwen2.5-1.5B-Instruct`** model.

## 1. Agreement Metrics
- **Sample Size**: 100 sentences (balanced: 50 salient, 50 non-salient)
- **Agreement Rate (Accuracy)**: `0.8200`
- **Cohen's Kappa Score**: `0.6400` (measures agreement above chance)
- **Silver Label Quality** (treating LLM Judge as ground truth):
  - **Precision**: `0.7400`
  - **Recall**: `0.8810`
  - **F1 Score**: `0.8043`

### Confusion Matrix
| | LLM Salient (1) | LLM Non-Salient (0) |
| --- | --- | --- |
| **Silver Salient (1)** | **TP: 37** (Agree) | **FP: 13** (Silver=1, LLM=0) |
| **Silver Non-Salient (0)** | **FN: 5** (Silver=0, LLM=1) | **TN: 45** (Agree) |

## 2. Qualitative Error Analysis

### Category A: Silver Salient (1) but LLM Non-Salient (0)
> [!NOTE]
> These are cases where the sentence physically intersects the SQuAD annotated answer offset, but the LLM believes it is not sufficient or contextually relevant to answer the question. This can happen if an answer span overlaps sentence boundaries slightly or if the sentence contains the answer keyword but lacks the semantic context.

**Example 1**:
- **Question**: Beyonce's father worked as a sales manager for what company?
- **Sentence**: *"Beyoncé Giselle Knowles was born in Houston, Texas, to Celestine Ann "Tina" Knowles (née Beyincé), a hairdresser and salon owner, and Mathew Knowles, a Xerox sales manager."*
- **LLM Reasoning**: Judgment: No

The given sentence does not provide any information about Beyoncé's father working at a specific company. It only states that Mathew Knowles, Beyoncé's father, was a Xerox sales manager. This information is tangential to answering the question about his job at a particular

**Example 2**:
- **Question**: What is the name of the main library at Notre Dame?
- **Sentence**: *"The main building is the 14-story Theodore M. Hesburgh Library, completed in 1963, which is the third building to house the main collection of books."*
- **LLM Reasoning**: Judgment: No

Reasoning: The given sentence does not provide any information about the name of the main library at Notre Dame. It only describes the location and construction details of the Theodore M. Hesburgh Library.

**Example 3**:
- **Question**: How many picks did Aqib Talib have?
- **Sentence**: *"Cornerbacks Aqib Talib (three interceptions) and Chris Harris, Jr. (two interceptions) were the other two Pro Bowl selections from the defense."*
- **LLM Reasoning**: Judgment: No

Reasoning: The given sentence does not provide any information about how many picks Aqib Talib had; instead, it lists his interceptions as part of the Pro Bowl selections.

**Example 4**:
- **Question**: How many touchdowns did Ronnie Hillman make?
- **Sentence**: *"Veteran receiver Demaryius Thomas led the team with 105 receptions for 1,304 yards and six touchdowns, while Emmanuel Sanders caught 76 passes for 1,135 yards and six scores, while adding another 106 yards returning punts."*
- **LLM Reasoning**: Judgment: No

The given sentence does not provide the number of touchdowns that Ronnie Hillman made; instead, it lists other players' statistics.

**Example 5**:
- **Question**: Where is the headquarters of the Congregation of the Holy Cross?
- **Sentence**: *"The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome)."*
- **LLM Reasoning**: Judgment: No

Reasoning: The given sentence does not provide any direct information about the headquarters of the Congregation of the Holy Cross; instead, it states that the university is the "major seat" but does not specify where this is located.

### Category B: Silver Non-Salient (0) but LLM Salient (1)
> [!NOTE]
> These represent cases where the sentence does NOT contain the exact answer character span, but the LLM judges it as salient. This typically occurs when a sentence contains crucial background context necessary to understand the answer, or when it contains a paraphrase of the answer that SQuAD annotators did not explicitly select.

**Example 1**:
- **Question**: What type of mascot do the Notre Dame sport teams have?
- **Sentence**: *"The leprechaun was featured on the cover of Time in November 1964 and gained national exposure."*
- **LLM Reasoning**: Judgment: Yes

Reasoning: The sentence directly states that "The leprechaun was featured on the cover of Time in November 1964 and gained national exposure," providing essential information about the mascot of Notre Dame sport teams.

**Example 2**:
- **Question**: What when conference do the Notre Dame fencing teams take part in?
- **Sentence**: *"The Fighting Irish previously competed in the Horizon League from 1982-83 to 1985-86, and again from 1987-88 to 1994-95, and then in the Big East Conference through 2012–13."*
- **LLM Reasoning**: Judgment: Yes

This sentence directly answers the question by specifying that the Notre Dame fencing teams took part in the Big East Conference from 1987-88 to 1994-95.

**Example 3**:
- **Question**: What year was Super Bowl 50?
- **Sentence**: *"As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."*
- **LLM Reasoning**: Judgment: Yes

The sentence directly states that Super Bowl 50 took place in 2016, providing the necessary information to answer the question about the year of Super Bowl 50.

**Example 4**:
- **Question**: When did John Jenkins become the president of Notre Dame?
- **Sentence**: *"Jenkins took over the position from Malloy on July 1, 2005."*
- **LLM Reasoning**: Judgment: Yes

The sentence "Jenkins took over the position from Malloy on July 1, 2005" directly answers when John Jenkins became the president of Notre Dame, providing the necessary information to answer the question.

**Example 5**:
- **Question**: What foot was injured on Manning that sidelined him in week 10?
- **Sentence**: *"Under Kubiak, the Broncos planned to install a run-oriented offense with zone blocking to blend in with quarterback Peyton Manning's shotgun passing skills, but struggled with numerous changes and injuries to the offensive line, as well as Manning having his worst statistical season since his rookie year with the Indianapolis Colts in 1998, due to a plantar fasciitis injury in his heel that he had suffered since the summer, and the simple fact that Manning was getting old, as he turned 39 in the 2015 off-season."*
- **LLM Reasoning**: Judgment: Yes

The sentence directly states that Manning had a plantar fasciitis injury in his heel that sidelined him in week 10. This information is crucial to answering the question about which foot was injured.
