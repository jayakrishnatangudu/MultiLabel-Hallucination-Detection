# Phase 1 Dataset Acquisition, Preprocessing & Quality Report

- **Raw Manifest Timestamp (UTC)**: `2026-11-12T00:04:32.997589+00:00`
- **Preprocessing Config SHA-256**: `801a1f92bdf2e7c1d49141a275873131a7a19dfe967278e0dcd474ec1239be0f`
- **Split Random Seed**: `42`
- **Total Raw Records**: `115,530`
- **Total Cleaned Records**: `99,461`
- **Total Unique Question/Document Groups**: `86,933`

## 1. Dataset-Wise Attrition & Length Statistics

| Dataset | Domain | Context | Raw Rows | Clean Rows | Dropped | Retention | Unique Groups | Median Q/A/C Words |
|---|---|---|---:|---:|---:|---:|---:|---|
| `medquad` | medical | False | 16,407 | 16,233 | 174 | 98.9% | 14,154 | 8.0 / 137.0 / — |
| `legal_qa_v1` | legal | False | 3,742 | 3,718 | 24 | 99.4% | 3,699 | 64.0 / 137.0 / — |
| `fiqa` | finance | False | 6,500 | 6,454 | 46 | 99.3% | 6,454 | 10.0 / 155.0 / — |
| `pubmedqa` | medical | True | 62,249 | 62,227 | 22 | 100.0% | 62,118 | 13.0 / 39.0 / 197.0 |
| `cuad_qa` | legal | True | 26,632 | 10,829 | 15,803 | 40.7% | 508 | 38.0 / 38.0 / 8888.0 |

### Attrition & Missingness Details

| Dataset | Missing Raw Q / A / C | CUAD Unanswerable | Empty Q/A/C | Too Short | Too Long | Exact Tuple Dups | Multi-Answer Same (Q,C) Groups |
|---|---|---:|---:|---:|---:|---:|---:|
| `medquad` | 0 / 0 / 0 | 0 | 0 | 1 | 125 | 44 | 1,426 (2,035 extra rows) |
| `legal_qa_v1` | 0 / 0 / 0 | 0 | 0 | 5 | 18 | 1 | 9 (9 extra rows) |
| `fiqa` | 0 / 11 / 0 | 0 | 11 | 30 | 5 | 0 | 0 (0 extra rows) |
| `pubmedqa` | 0 / 0 / 0 | 0 | 0 | 13 | 0 | 9 | 0 (0 extra rows) |
| `cuad_qa` | 0 / 14208 / 0 | 14,208 | 0 | 1,576 | 0 | 19 | 1,957 (4,638 extra rows) |

## 2. Near-Duplicate Clustering Audit

- **`medquad`**: evaluated `14,198` unique questions -> merged `44` near-duplicate pairs into `42` multi-member clusters (max cluster size `3`), while rejecting `18` template-similar pairs via rare-token guard and `148` via `qtype` mismatch.
  - *Sample merged pairs*:
    - `1.0` (rare Jaccard `1.0`): "What is (are) Parkinson's Disease ?" <-> "What is (are) Parkinson disease ?"
    - `1.0` (rare Jaccard `1.0`): "What are the treatments for Parkinson's Disease ?" <-> "What are the treatments for Parkinson disease ?"
    - `1.0` (rare Jaccard `1.0`): "What is (are) Sjgren's Syndrome ?" <-> "What is (are) Sjgren syndrome ?"
    - `1.0` (rare Jaccard `1.0`): "What are the treatments for Sjgren's Syndrome ?" <-> "What are the treatments for Sjgren syndrome ?"
    - `1.0` (rare Jaccard `1.0`): "What is (are) Friedreich's Ataxia ?" <-> "What is (are) Friedreich ataxia ?"
  - *Sample borderline unmerged pairs (verified distinct entities/topics)*:
    - `0.7855` (rare Jaccard `0.625`, below_cosine_threshold): "what are public health agencies doing to prevent or control botulism?" != "what are public health agencies doing to prevent or control yersiniosis for Yersinia ?"
    - `0.8216` (rare Jaccard `0.6667`, below_cosine_threshold): "What is (are) Chronic Fatigue Syndrome (CFS) ?" != "What is (are) Chronic Fatigue Syndrome ?"
    - `0.8461` (rare Jaccard `0.6667`, below_cosine_threshold): "How to diagnose Chronic Fatigue Syndrome (CFS) ?" != "How to diagnose Chronic fatigue syndrome ?"
    - `0.8317` (rare Jaccard `0.6667`, below_cosine_threshold): "What are the treatments for Chronic Fatigue Syndrome (CFS) ?" != "What are the treatments for Chronic fatigue syndrome ?"
    - `0.7808` (rare Jaccard `0.3333`, below_cosine_threshold): "what research (or clinical trials) is being done for Cerebral Palsy ?" != "what research (or clinical trials) is being done for Cerebral Atrophy ?"
- **`legal_qa_v1`**: evaluated `3,709` unique questions -> merged `10` near-duplicate pairs into `10` multi-member clusters (max cluster size `2`), while rejecting `0` template-similar pairs via rare-token guard and `0` via `qtype` mismatch.
  - *Sample merged pairs*:
    - `0.9433` (rare Jaccard `0.9412`): "Central District Court of California - specific question. Serving complaint/summons - federal case. Central district court of California. Case was efiled in EDSS system for in Pro Per plaintiffs. Documents will appear in PACER. Can copies of complaint/summons for serving be downloaded from PACER? What is conformed copy of documents? When they are needed?" <-> "Serving complaint/summons - federal case. Central district court of California.. Serving complaint/summons - federal case. Central district court of California. Case was efiled in EDSS system for in Pro Per plaintiffs. Documents will appear in PACER. Can copies of complaint/summons for serving be downloaded from PACER? What is conformed copy of documents? When they are needed?"
    - `0.9618` (rare Jaccard `0.9333`): "Do I have to continue paying a credit line loan if the bank no longer has the original signed documents?. I took out a credit line on a business over 20 years ago, that business is now closed. The bank at some point converted the account into my personal name and they have raised the interest rate 3 times what the original rate was. They cannot produce the original signed documents which I requested to explain the rate increases. Thank you." <-> "Do I have to continue paying a credit line loan if the bank no longer has the original signed documents?. I took out a credit line on a business over 20 years ago, that business is now closed. The bank at some point converted the account into my personal name and they have raised the interest rate 3 times what the original rate was. They cannot produce the original signed documents which I requested to explain the rate increases. The loan originated in California. Thank you."
    - `0.9326` (rare Jaccard `0.9643`): "What has to be specific legal basis to confront Defendant's motion to strike based on failure to join a party (rule 19). EMTALA complaint against hospital has additional claim for CMIA violation. Plaintiff medical information was disclosed to hospice (with non-existent terminal illness). What has to be specific legal basis for Plaintiff to confront Defendant's motion to strike based on failure to join a party (hospice) - rule 19? Apparently legal action against hospice would not be limited only to CMIA violation, adding hospice as defendant at this point would deprive Plaintiff's rights. Can Cmia violation claim be stricken from complaint without impact to original EMTALA claim?" <-> "What has to be specific legal basis? - please see question below. EMTALA complaint against hospital has additional claim for CMIA violation. Plaintiff medical information was disclosed to hospice (with non-existent terminal illness). What has to be specific legal basis for Plaintiff to confront Defendant's motion to strike based on failure to join a party (hospice) - rule 19? Apparently legal action against hospice would not be limited only to CMIA violation, adding hospice as defendant at this point would deprive Plaintiff's rights. Can Cmia violation claim be stricken from complaint without impact to original EMTALA claim?"
    - `0.9263` (rare Jaccard `0.92`): "Adding claim/defendant to federal complaint - question.. Adding claim/defendant to federal complaint - question. EMTALA complaint against hospital was just filed with federal court, with additional claim for CMIA violation (under supplemental jurisdiction) There is basis to add false claims act claim for fraudulent referral to hospice with non-existent terminal illness, adding hospice as defendant. Should plaintiff wait for scheduling conference? At what point of lawsuit plaintiff can ask court to grant leave to amend complaint?" <-> "Adding claim to complaint, adding defendant - federal case question. Central District Court.. Adding claim/defendant to federal complaint - question. EMTALA complaint against hospital was just filed with federal court, with additional claim for CMIA violation (under supplemental jurisdiction) There is basis to add false claims act claim for fraudulent referral to hospice with non-existent terminal illness, adding hospice as defendant. Should plaintiff wait for scheduling conference? At what point of lawsuit plaintiff can ask court to grant leave to amend complaint?"
    - `0.9873` (rare Jaccard `0.898`): "The Skullgirls team has taken a digital product that many people purchased and altered it in very significant ways.. The Skullgirls team has taken a digital product that many people purchased and altered it in very significant ways. They've cut out entire pages from the art book, removed and re-announced voice lines, and drawn over concept art to make it less offensive. The current Skullgirls team, while many of them have ties to the very beginning of the game, is not the same team it was at the beginning. They're missing some of the most important contributors to the Skullgirls brand. They're also literally a completely different company than the original Lab Zero, who is now censoring Lab Zero's products. if these were physical products. Imagine if you bought a physical art book that you really loved. Ten years later, when the company you bought that art book from is overtaken by another company, that new company comes into your house and demands that you give them the pages of the art book that they deemed offensive. simiar situated similar facts & loss" <-> "The Skullgirls team has taken a digital product that many people purchased and altered it in very significant ways.. The Skullgirls team has taken a digital product that many people purchased and altered it in very significant ways. They've cut out entire pages from the art book, removed and re-announced voice lines, and drawn over concept art to make it less offensive. The current Skullgirls team, while many of them have ties to the very beginning of the game, is not the same team it was at the beginning. They're missing some of the most important contributors to the Skullgirls brand. They're also literally a completely different company than the original Lab Zero, who is now censoring Lab Zero's products. if these were physical products. Imagine if you bought a physical art book that you really loved. Ten years later, when the company you bought that art book from is overtaken by another company, that new company comes into your house and demands that you give them the pages of the art book that they deemed offensive."
  - *Sample borderline unmerged pairs (verified distinct entities/topics)*:
    - `0.8028` (rare Jaccard `0.7083`, below_cosine_threshold): "Why are the lower courts operating under a maritime law when we are on land how did they have jurisdiction on land. Why is it California judicial system violates the rights under the constitution of due process when the Constitution states anything that's adverse to the Constitution is null and void? And how is it that the police department is allowed to invade your privacy punish you without proper due process rights. And the court judge and the district attorney when you question them about having jurisdiction over the subject matter they tell me that the county has jurisdiction over the living man from my understanding from reading the Constitution that no corporation has jurisdiction over Man." != "Why are the lower courts operating under a maritime law when we are on land how did they have jurisdiction on land. Why is it California judicial system violates the rights under the constitution of due process when the Constitution states anything that's adverse to the Constitution is null and void? And how is it that the police department is allowed to invade your privacy punish you without proper due process rights"
    - `0.8011` (rare Jaccard `0.75`, below_cosine_threshold): "I’ve slapped my bf before. He has evidence. Then he attacked me twice very brutal attacks. Who gets into trouble?. I’ve slapped him 3 times in the past. But yesterday he brutally attacked me, choked me slapped me dragged me by my hair and threw a door on me. Who gets into trouble?" != "I’ve slapped him 3 times in the past. But yesterday he brutally attacked me, choked me slapped me dragged me by my hair. I’ve slapped him 3 times in the past. But yesterday he brutally attacked me, choked me slapped me dragged me by my hair and threw a door on me. Who gets into trouble?"
    - `0.8634` (rare Jaccard `0.875`, below_cosine_threshold): "Does California false claims act cover medicare violations?. Does California false claims act cover medicare violations? (Or limited to California Medi-cal violations?" != "Does California Healthcare false claims act limited to California Medi-cal violations?. Does California false claims act cover medicare violations? (Or limited to California Medi-cal violations?)"
    - `0.893` (rare Jaccard `0.8636`, below_cosine_threshold): "Plaintiff's response to Defendant's demurrer/motion to strike.. Options are: opposition or amended complaint. But if Plaintiff needs to obtain more information to amend complaint, short of time to do it before hearing: how Plaintiff can inform court about it? Can amended complaint include statement that defendant's discovery responses expected?" != "In California superior court. Plaintiff's preparation for hearing on demurrer/motion to strike.. Options are: opposition or amended complaint. But if Plaintiff needs to obtain more information to amend complaint, short of time to do it before hearing: how Plaintiff can inform court about it? Can amended complaint include statement that defendant's discovery responses expected?"
    - `0.8897` (rare Jaccard `0.8182`, below_cosine_threshold): "Questions regarding legal career. I'm currently a Political Science student at UCLA. I am considering going to law school and I would like to talk to currently practicing lawyers to gain some perspective regarding law school and careers in law. Would appreciate if anyone would be willing to chat :)" != "Questions regarding legal career. I'm currently a Political Science student at UCLA. I am considering going to law school and I would like to talk to currently practicing lawyers to gain some perspective regarding law school and careers in law so I can make an informed decision as to whether or not to attend law school. Would appreciate if anyone would be willing to chat :)"
- **`fiqa`**: evaluated `6,454` unique questions -> merged `0` near-duplicate pairs into `0` multi-member clusters (max cluster size `1`), while rejecting `0` template-similar pairs via rare-token guard and `0` via `qtype` mismatch.
  - *Sample borderline unmerged pairs (verified distinct entities/topics)*:
    - `0.7983` (rare Jaccard `0.6`, below_cosine_threshold): "Does the CRA reprieve those who have to commute for work?" != "Does the IRS reprieve those who have to commute for work?"
    - `0.8461` (rare Jaccard `0.75`, below_cosine_threshold): "Where to start with personal finance?" != "Where to start with personal finance in Canada?"
    - `0.8652` (rare Jaccard `0.6667`, below_cosine_threshold): "How do I get into investing?" != "How do I get into investing in stocks?"
    - `0.7847` (rare Jaccard `1.0`, below_cosine_threshold): "What are my investment options in Australia?" != "Investment options in Australia"
- **`pubmedqa`**: Document-grounded dataset grouped primarily by source_document_id to avoid merging template questions across distinct contracts/abstracts. (unique documents: `62227`).
- **`cuad_qa`**: Document-grounded dataset grouped primarily by source_document_id to avoid merging template questions across distinct contracts/abstracts. (unique documents: `508`).

## 3. Source Licensing, Attribution & Redistribution Status

| Dataset | Version (Commit SHA) | HF Card Tag | Verified Upstream License | Status | Redistribute? | Required Attribution |
|---|---|---|---|---|---|---|
| `medquad` | `5b0961fbaa6d` | `None` | `CC-BY-4.0` | `upstream_caution` | `True` | Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. BMC Bioinformatics, 20, 1-23. |
| `legal_qa_v1` | `6280beb74faf` | `None` | `unresolved` | `unresolved` | `False` | dzunggg/legal-qa-v1 (Hugging Face community dataset; upstream legal Q&A portal provenance unverified). |
| `fiqa` | `dbbbdef53188` | `apache-2.0` | `mixed_upstream_cc_by_sa_and_forum_tos` | `upstream_caution` | `False` | Maia, M., Handschuh, S., Freitas, A., Davis, B., McDermott, R., Zarrouk, M., & Balahur, A. (2018). WWW'18 Open Challenge: Financial Opinion Mining and Question Answering. |
| `pubmedqa` | `9001f2853fb8` | `mit` | `MIT` | `verified_open` | `True` | Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., & Lu, X. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP-IJCNLP 2019. |
| `cuad_qa` | `67faa0e6023b` | `cc-by-4.0` | `CC-BY-4.0` | `upstream_caution` | `True` | Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. NeurIPS 2021. |

## 4. Question/Document-Grouped Data Split Distribution

- **Leakage Check**: `{'unique_groups_checked': 86933, 'unique_documents_checked': 62735, 'unique_context_free_questions_checked': 24361, 'leaked_groups': 0, 'leaked_documents': 0, 'leaked_context_free_questions': 0}`

| Dataset / Subset | Train (80%) | Val (10%) | Test Synthetic Pool (8%) | Test Natural Pool (2%) |
|---|---:|---:|---:|---:|
| `medquad/train` | 13,010 | 1,638 | 1,260 | 325 |
| `legal_qa_v1/train` | 2,974 | 372 | 298 | 74 |
| `fiqa/train` | 4,369 | 546 | 437 | 109 |
| `fiqa/test` | 794 | 99 | 80 | 20 |
| `pubmedqa/pqa_labeled` | 800 | 100 | 80 | 20 |
| `pubmedqa/pqa_unlabeled` | 48,978 | 6,120 | 4,905 | 1,224 |
| `cuad_qa/train` | 7,733 | 919 | 783 | 236 |
| `cuad_qa/test` | 970 | 92 | 79 | 17 |
| **Total Records** | **79,628** | **9,886** | **7,922** | **2,025** |
| **Total Groups** | **69,546** | **8,694** | **6,955** | **1,738** |
