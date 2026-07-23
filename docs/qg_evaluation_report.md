# Downstream QG & QA Agent Comparative Verification Report (Stage 5)

Generated at: 2026-07-22 21:08:50

### Summary Metrics
| Saliency Selection Method | Saliency Top-1 Accuracy | QA-Acceptability Rate (QA-AR) |
| :--- | :---: | :---: |
| **Oracle (Ground Truth)** | 1.0000 | 0.9300 |
| **LGSM (Trained Sequence Model)** | 0.9450 | 0.9250 |
| **LLM Judge (Zero-shot)** | 0.9050 | 0.9350 |

### Detailed Sample Generation Outputs
#### Sample 1 (ID: ctx_validation_0)
* **Oracle target sentence**: "This fragmentation was decisive in Khwarezmia's defeats, as it allowed the Mongols, although exhausted from the long journey, to immediately set about defeating small fractions of the Khwarzemi forces instead of facing a unified defense."
  * Generated Question: *"How did the fragmentation of the Shah's army contribute to Khwarezmia's defeat?"*
  * QA Agent Predicted Answer: "allowing the Mongols, although exhausted from the long journey, to immediately set about defeating small fractions of the Khwarzemi forces instead of facing a unified defense"
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "The Shah's army was split by diverse internecine feuds and by the Shah's decision to divide his army into small groups concentrated in various cities."
  * Generated Question: *"What caused the Khwarzemi forces' defeat at the hands of the Mongols?"*
  * QA Agent Predicted Answer: "fragmentation"
  * Verified: ❌ No

* **LLM Judge Predicted sentence**: "The Shah's army was split by diverse internecine feuds and by the Shah's decision to divide his army into small groups concentrated in various cities."
  * Generated Question: *"What caused the Khwarzemi forces' defeat at the hands of the Mongols?"*
  * QA Agent Predicted Answer: "fragmentation"
  * Verified: ❌ No
--------------------------------------------------------------------------------

#### Sample 2 (ID: ctx_validation_1)
* **Oracle target sentence**: "While the Treaties and Regulations will have direct effect (if clear, unconditional and immediate), Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Generated Question: *"What does the target sentence state about Directives?"*
  * QA Agent Predicted Answer: "Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "While the Treaties and Regulations will have direct effect (if clear, unconditional and immediate), Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Generated Question: *"What does the Target Sentence state about the relationship between Directives and citizen's ability to sue?"*
  * QA Agent Predicted Answer: "Directives do not generally give citizens standing to sue other citizens."
  * Verified: ✅ Yes

* **LLM Judge Predicted sentence**: "While the Treaties and Regulations will have direct effect (if clear, unconditional and immediate), Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Generated Question: *"What does the Target Sentence state about Directives?"*
  * QA Agent Predicted Answer: "Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Verified: ✅ Yes
--------------------------------------------------------------------------------

#### Sample 3 (ID: ctx_validation_10)
* **Oracle target sentence**: "Edward John Noble, the owner of Life Savers candy, drugstore chain Rexall and New York City radio station WMCA, purchased the network for $8 million."
  * Generated Question: *"Who bought the network for $8 million?"*
  * QA Agent Predicted Answer: "Edward John Noble"
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "Edward John Noble, the owner of Life Savers candy, drugstore chain Rexall and New York City radio station WMCA, purchased the network for $8 million."
  * Generated Question: *"Who purchased the network for $8 million?"*
  * QA Agent Predicted Answer: "Edward John Noble"
  * Verified: ✅ Yes

* **LLM Judge Predicted sentence**: "Edward John Noble, the owner of Life Savers candy, drugstore chain Rexall and New York City radio station WMCA, purchased the network for $8 million."
  * Generated Question: *"Who purchased the network for $8 million?"*
  * QA Agent Predicted Answer: "Edward John Noble"
  * Verified: ✅ Yes
--------------------------------------------------------------------------------

#### Sample 4 (ID: ctx_validation_100)
* **Oracle target sentence**: "Another important monument, the statue of Little Insurgent located at the ramparts of the Old Town, commemorates the children who served as messengers and frontline troops in the Warsaw Uprising, while the impressive Warsaw Uprising Monument by Wincenty Kućma was erected in memory of the largest insurrection of World War II."
  * Generated Question: *"What other significant monument exists in Warsaw that honors the contributions of children during the Warsaw Uprising?"*
  * QA Agent Predicted Answer: "The statue of Little Insurgent located at the ramparts of the Old Town."
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "Another important monument, the statue of Little Insurgent located at the ramparts of the Old Town, commemorates the children who served as messengers and frontline troops in the Warsaw Uprising, while the impressive Warsaw Uprising Monument by Wincenty Kućma was erected in memory of the largest insurrection of World War II."
  * Generated Question: *"What other significant monument exists besides the Warsaw Citadel?"*
  * QA Agent Predicted Answer: "The statue of Little Insurgent located at the ramparts of the Old Town"
  * Verified: ✅ Yes

* **LLM Judge Predicted sentence**: "The Warsaw Citadel, an impressive 19th-century fortification built after the defeat of the November Uprising, was a place of martyr for the Poles."
  * Generated Question: *"What historical event did the Warsaw Citadel commemorate?"*
  * QA Agent Predicted Answer: "the defeat of the November Uprising"
  * Verified: ✅ Yes
--------------------------------------------------------------------------------

#### Sample 5 (ID: ctx_validation_101)
* **Oracle target sentence**: "A construction project is a complex net of contracts and other legal obligations, each of which all parties must carefully consider."
  * Generated Question: *"What does the context passage state about construction projects?"*
  * QA Agent Predicted Answer: "Construction projects involve careful consideration of multiple contracts and legal obligations, with emphasis on ensuring clarity and smooth execution despite time constraints."
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "A construction project is a complex net of contracts and other legal obligations, each of which all parties must carefully consider."
  * Generated Question: *"What does the context passage state about construction projects?"*
  * QA Agent Predicted Answer: "Construction projects involve careful consideration of multiple contracts and legal obligations, with emphasis on ensuring clarity and smooth execution despite time constraints."
  * Verified: ✅ Yes

* **LLM Judge Predicted sentence**: "A construction project is a complex net of contracts and other legal obligations, each of which all parties must carefully consider."
  * Generated Question: *"What does the context passage state about construction projects?"*
  * QA Agent Predicted Answer: "Construction projects involve careful consideration of multiple contracts and legal obligations, with emphasis on ensuring clarity and smooth execution despite time constraints."
  * Verified: ✅ Yes
--------------------------------------------------------------------------------

