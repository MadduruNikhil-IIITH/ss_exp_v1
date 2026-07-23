# Downstream QG & QA Agent Comparative Verification Report (Stage 5)

Generated at: 2026-07-22 13:16:16

### Summary Metrics
| Saliency Selection Method | Saliency Top-1 Accuracy | QA-Acceptability Rate (QA-AR) |
| :--- | :---: | :---: |
| **Oracle (Ground Truth)** | 1.0000 | 1.0000 |
| **LGSM (Trained Sequence Model)** | 0.7000 | 0.9000 |
| **LLM Judge (Zero-shot)** | 0.7000 | 0.8000 |

### Detailed Sample Generation Outputs
#### Sample 1 (ID: ctx_validation_0)
* **Oracle target sentence**: "This fragmentation was decisive in Khwarezmia's defeats, as it allowed the Mongols, although exhausted from the long journey, to immediately set about defeating small fractions of the Khwarzemi forces instead of facing a unified defense."
  * Generated Question: *"How did the fragmentation of the Shah's army contribute to Khwarezmia's defeat?"*
  * QA Agent Predicted Answer: "allowing the Mongols, although exhausted from the long journey, to immediately set about defeating small fractions of the Khwarzemi forces instead of facing a unified defense"
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "The Shah's army was split by diverse internecine feuds and by the Shah's decision to divide his army into small groups concentrated in various cities."
  * Generated Question: *"What caused the Khwarzemi forces' defeat?"*
  * QA Agent Predicted Answer: "fragmentation"
  * Verified: ❌ No

* **LLM Judge Predicted sentence**: "The Shah's army was split by diverse internecine feuds and by the Shah's decision to divide his army into small groups concentrated in various cities."
  * Generated Question: *"What factors contributed to Khwarezmia's defeats according to the passage?"*
  * QA Agent Predicted Answer: "diverse internecine feuds and the Shah's decision to divide his army into small groups"
  * Verified: ✅ Yes
--------------------------------------------------------------------------------

#### Sample 2 (ID: ctx_validation_1)
* **Oracle target sentence**: "While the Treaties and Regulations will have direct effect (if clear, unconditional and immediate), Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Generated Question: *"What does the Target Sentence state about Directives?"*
  * QA Agent Predicted Answer: "Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "While the Treaties and Regulations will have direct effect (if clear, unconditional and immediate), Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Generated Question: *"What does the target sentence state about Directives?"*
  * QA Agent Predicted Answer: "Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Verified: ✅ Yes

* **LLM Judge Predicted sentence**: "While the Treaties and Regulations will have direct effect (if clear, unconditional and immediate), Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Generated Question: *"What does the Target Sentence state about Directives?"*
  * QA Agent Predicted Answer: "Directives do not generally give citizens (as opposed to the member state) standing to sue other citizens."
  * Verified: ✅ Yes
--------------------------------------------------------------------------------

#### Sample 3 (ID: ctx_validation_2)
* **Oracle target sentence**: "Chrysler ended production of their full-sized luxury sedans at the end of the 1981 model year, moving instead to a full front-wheel drive lineup for 1982 (except for the M-body Dodge Diplomat/Plymouth Gran Fury and Chrysler New Yorker Fifth Avenue sedans)."
  * Generated Question: *"What happened to Chrysler's full-sized luxury sedans by the end of the 1981 model year?"*
  * QA Agent Predicted Answer: "Chrysler ended production of their full-sized luxury sedans at the end of the 1981 model year, moving instead to a full front-wheel drive lineup for 1982 ("
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "Chrysler ended production of their full-sized luxury sedans at the end of the 1981 model year, moving instead to a full front-wheel drive lineup for 1982 (except for the M-body Dodge Diplomat/Plymouth Gran Fury and Chrysler New Yorker Fifth Avenue sedans)."
  * Generated Question: *"What happened to Chrysler's full-sized luxury sedans by the end of 1981?"*
  * QA Agent Predicted Answer: "Chrysler ended production of their full-sized luxury sedans at the end of the 1981 model year, moving instead to a full front-wheel drive lineup for 1982 ("
  * Verified: ✅ Yes

* **LLM Judge Predicted sentence**: "Federal safety standards, such as NHTSA Federal Motor Vehicle Safety Standard 215 (pertaining to safety bumpers), and compacts like the 1974 Mustang I were a prelude to the DOT "downsize" revision of vehicle categories."
  * Generated Question: *"What factors led to the decline of large car sizes in the United States during the late 1970s?"*
  * QA Agent Predicted Answer: "The decline of large car sizes in the United States during the late 1970s was due to federal safety standards, compact vehicles, and the economic impact of fuel shortages and rising oil prices"
  * Verified: ❌ No
--------------------------------------------------------------------------------

#### Sample 4 (ID: ctx_validation_3)
* **Oracle target sentence**: "He was a man who knew that the good laws and rules lived longer than fancy palaces," Elbegdorj said in his speech on the 850th anniversary of Chinggis Khaan's birth."
  * Generated Question: *"What did President Tsakhiagiin Elbegdorj emphasize about Genghis Khan's understanding of law?"*
  * QA Agent Predicted Answer: "He was a man who deeply realized that the justice begins and consolidates with the equality of law, and not with the distinctions between people. He was a man who knew that the good laws and rules"
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "Genghis Khan is regarded as one of the prominent leaders in Mongolia's history."
  * Generated Question: *"Who is considered one of the prominent leaders in Mongolia's history?"*
  * QA Agent Predicted Answer: "Genghis Khan"
  * Verified: ✅ Yes

* **LLM Judge Predicted sentence**: "Genghis Khan is regarded as one of the prominent leaders in Mongolia's history."
  * Generated Question: *"Who is considered one of the prominent leaders in Mongolia's history?"*
  * QA Agent Predicted Answer: "Genghis Khan"
  * Verified: ✅ Yes
--------------------------------------------------------------------------------

#### Sample 5 (ID: ctx_validation_4)
* **Oracle target sentence**: "The engulfed alga was broken down, leaving only its chloroplast, and sometimes its cell membrane and nucleus, forming a chloroplast with three or four membranes—the two cyanobacterial membranes, sometimes the eaten alga's cell membrane, and the phagosomal vacuole from the host's cell membrane."
  * Generated Question: *"What happened to the engulfed alga after being digested?"*
  * QA Agent Predicted Answer: "The engulfed alga was broken down, leaving only its chloroplast, and sometimes its cell membrane and nucleus, forming a chloroplast with three or four membranes."
  * Verified: ✅ Yes

* **LGSM Predicted sentence**: "The engulfed alga was broken down, leaving only its chloroplast, and sometimes its cell membrane and nucleus, forming a chloroplast with three or four membranes—the two cyanobacterial membranes, sometimes the eaten alga's cell membrane, and the phagosomal vacuole from the host's cell membrane."
  * Generated Question: *"What happens after the engulfed alga is broken down?"*
  * QA Agent Predicted Answer: "Only its chloroplast remains, along with its cell membrane and nucleus, forming a chloroplast with three or four membranes."
  * Verified: ✅ Yes

* **LLM Judge Predicted sentence**: "While primary chloroplasts have a double membrane from their cyanobacterial ancestor, secondary chloroplasts have additional membranes outside of the original two, as a result of the secondary endosymbiotic event, when a nonphotosynthetic eukaryote engulfed a chloroplast-containing alga but failed to digest it—much like the cyanobacterium at the beginning of this story."
  * Generated Question: *"What explains why secondary chloroplasts have more than two membranes compared to primary chloroplasts?"*
  * QA Agent Predicted Answer: "The engulfed alga was broken down, leaving only its chloroplast, and sometimes its cell membrane and nucleus, forming a chloroplast with three or four membranes."
  * Verified: ❌ No
--------------------------------------------------------------------------------

