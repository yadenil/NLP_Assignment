# Addis Ababa University
## College of Technology & Built Environment
### School of Information Technology and Engineering
### Course: Natural Language Processing (NLP) — Assignment 1
**Instructor:** Fantahun B. (PhD)  
**Corpus:** General-Purpose Amharic Corpus (GPAC)  
**Reference Paper:** *Combating Fake News in “Low-Resource” Languages: Amharic Fake News Detection Accompanied by Resource Crafting* ([DOI: 10.3390/info12010020](https://doi.org/10.3390/info12010020))

---

## 📖 Executive Summary & Table of Contents
This repository contains the complete, production-grade implementation for all four tasks (Questions 1 through 4) of NLP Assignment 1.

1. **[Question #1: N-gram Language Model (6%)](#question-1-n-gram-language-model)**
   - 1.1 Corpus Statistics & Amharic Preprocessing
   - 1.2 N-gram Construction (n = 1, 2, 3, 4)
   - 1.3 Probabilities of N-grams & Top 10 Rankings
   - 1.4 Bigram Conditional Probability $P(w_i | w_{i-1})$
   - 1.5 Stopword Removal & Recomputed Frequencies
   - 1.6 Word Cloud Visualizations (n=1 to 4 Before & After Stopwords)
   - 1.7 Sentence Probability: *"ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።"*
   - 1.8 Random Sentence Generation & Linguistic Analysis
2. **[Question #2: Intrinsic Evaluation Method (3%)](#question-2-intrinsic-evaluation-perplexity)**
   - Train/Test Split (80% / 20%)
   - Smoothing Techniques: Laplace (Add-1), Add-k (Lidstone), and Jelinek-Mercer Linear Interpolation
   - Cross-Entropy and Perplexity (PPL) Benchmarks
   - Visualizations & Empirical Analysis
3. **[Question #3: Extrinsic Evaluation Method (5%)](#question-3-extrinsic-evaluation-downstream-nlp-tasks)**
   - Downstream Task 1: Next-Word Prediction / Cloze Benchmark (Top-1, Top-3, Top-5 Accuracy, MRR, Latency)
   - Downstream Task 2: Sentence Fluency & Acceptability Discrimination
   - Performance Trade-offs & Model Comparisons
4. **[Question #4: Predictive Text and Next-Word Suggestion System (6%)](#question-4-amharic-predictive-text-system)**
   - Real-time Suggestion Architecture & Katz Backoff
   - PDF Benchmark Verification: `"ኢትዮጵያ ውስጥ የቡና"`
   - Interactive CLI Application
   - Interactive Modern Streamlit Web Application
5. **[How to Run the Code](#how-to-run-the-code)**

---

## 📁 Repository Structure
```tree
NLP assignment/
├── amharic_preprocessor.py      # Preprocessing, normalization, tokenization, abbreviation handling, stopword filter
├── ngram_model.py              # Flexible N-gram LM class with MLE, Add-k, Laplace, Interpolation, & Sampling
├── q1_ngram_language_model.py  # Question 1: Corpus statistics, n-grams, top-10, word clouds, sentence probability
├── q2_intrinsic_evaluation.py  # Question 2: Perplexity evaluation across n=1..4 and smoothing algorithms
├── q3_extrinsic_evaluation.py  # Question 3: Next-word prediction accuracy (Top-1/3/5), MRR, & fluency discrimination
├── q4_predictive_system.py     # Question 4: Next-word suggestion engine with CLI interactive composer
├── app_streamlit.py            # Question 4: Interactive Web UI application built with Streamlit
├── run_all.py                  # Master runner executing Questions 1 to 4 end-to-end
├── wordclouds/                 # Generated 8 high-resolution Amharic word clouds
│   ├── wordcloud_n1_before_stopwords.png
│   ├── wordcloud_n2_before_stopwords.png
│   ├── wordcloud_n3_before_stopwords.png
│   ├── wordcloud_n4_before_stopwords.png
│   ├── wordcloud_n1_after_stopwords.png
│   ├── wordcloud_n2_after_stopwords.png
│   ├── wordcloud_n3_after_stopwords.png
│   └── wordcloud_n4_after_stopwords.png
├── plots/                      # Evaluation charts and comparative plots
│   ├── q2_perplexity_comparison.png
│   ├── q2_smoothing_methods_log_ppl.png
│   ├── q3_next_word_accuracy.png
│   └── q3_mrr_and_fluency.png
├── models/                     # Serialized precomputed n-gram models for instant startup
│   └── ngram_cache.pkl
└── results/                    # Machine-readable JSON logs of all metrics
    ├── q1_results.json
    ├── q2_results.json
    └── q3_results.json
```

---

## Question #1: N-gram Language Model

### 1.1 Corpus Statistics and Relevant Preprocessing
#### Amharic Text Preprocessing Pipeline:
1. **Sentence Segmentation:** Formulated according to instructor specifications:
   $$\text{SENT\_SPLIT} = \text{re.compile}(r'(?:፡፡)+|።+|[!?]+')$$
   Extracts complete, semantically standalone Amharic sentences while discarding micro-fragments.
2. **Abbreviation Handling (Multi-token unification):**
   Amharic abbreviations often contain slashes, dots, or spaces that standard tokenizers break into nonsensical single letters. Our pipeline canonicalizes and protects them as atomic tokens:
   - `ዶ ር` / `ዶ/ር` / `ዶ.ር` $\rightarrow$ `ዶ/ር` (Doctor)
   - `ዓ ም` / `ኣ ም` / `ዓ.ም` $\rightarrow$ `ኣ.ም` (Year / ዓመተ ምሕረት)
   - `እ ኤ አ` / `እ.ኤ.አ` $\rightarrow$ `እ.ኤ.አ` (Gregorian Calendar / AD)
   - `ት ቤት` / `ት/ቤት` $\rightarrow$ `ት/ቤት` (School)
   - `ወ ሮ` / `ወ/ሮ` $\rightarrow$ `ወ/ሮ` (Mrs.)
   - `መ ቤት` / `መ/ቤት` $\rightarrow$ `መ/ቤት` (Office)
   - `ጠ ሚ` / `ጠ/ሚ` $\rightarrow$ `ጠ/ሚ` (Prime Minister)
3. **Homophone Normalization:** Standardizes variant Ge'ez graphemes representing identical phonemes:
   - $h$-phonemes: ሐ, ኀ, ኃ, ኻ $\rightarrow$ canonical ሀ (across all 7 vowel orders)
   - $s$-phonemes: ሠ $\rightarrow$ canonical ሰ (across all 7 vowel orders)
   - $a$-phonemes: ዐ $\rightarrow$ canonical አ (across all 7 vowel orders)
   - $ts$-phonemes: ፀ $\rightarrow$ canonical ጸ (across all 7 vowel orders)
4. **Punctuation & Word-Separator Normalization:**
   Replaces Ethiopic word separator (`፡`) with whitespace; removes non-Ethiopic punctuation while preserving valid internal abbreviation symbols.

#### Empirical Corpus Statistics Table (GPAC Sample):
| Metric | Value |
| :--- | :--- |
| **Total Sampled Sentences** | 59,999 |
| **Total Word Tokens** | 778,445 |
| **Vocabulary Size (Unique Words)** | 152,827 |
| **Raw Character Count** | 3,979,553 |
| **Cleaned Character Count** | 3,154,687 |
| **Average Sentence Length (Words)** | 12.97 tokens |
| **Average Sentence Length (Characters)** | 52.58 characters |
| **Type-Token Ratio (TTR / Lexical Diversity)** | 0.1963 |
| **Hapax Legomena (Words occurring exactly once)** | 95,661 (62.59% of vocabulary) |

---

### 1.2 N-gram Construction (n = 1, 2, 3, 4)
Extracted pure word n-grams without artificial `<s>` sequence tags:
- **Sample Sentence:** `['አንባቢያን', 'ኢትዮጵያ', 'በተደጋጋሚ', 'ጥሪው', 'ደርሷት', 'ልትታደመው', 'ያልቻለችው', 'የአለም', 'የእግር', 'ኳስ', 'ዋ', 'ለ19ኛ', 'ጊዜ', 'በደቡብ', 'አፍሪካ', 'ሲጠጣ', 'በሩቅ', 'እያየች', 'አንጀቷ', 'ባረረ', 'ልክ', 'በአመቱ', 'በለስ', 'ቀናትና', 'ሌላ', 'ዋ', 'ልትታደም', 'ሁለት', 'ልጆቿን', 'ወደ', 'ደቡብ', 'አፍሪካ', 'ላከች']`
- **Sample Unigrams (n=1):** `('አንባቢያን',)`, `('ኢትዮጵያ',)`, `('በተደጋጋሚ',)`, `('ጥሪው',)`
- **Sample Bigrams (n=2):** `('አንባቢያን', 'ኢትዮጵያ')`, `('ኢትዮጵያ', 'በተደጋጋሚ')`, `('በተደጋጋሚ', 'ጥሪው')`
- **Sample Trigrams (n=3):** `('አንባቢያን', 'ኢትዮጵያ', 'በተደጋጋሚ')`, `('ኢትዮጵያ', 'በተደጋጋሚ', 'ጥሪው')`
- **Sample 4-grams (n=4):** `('አንባቢያን', 'ኢትዮጵያ', 'በተደጋጋሚ', 'ጥሪው')`, `('ኢትዮጵያ', 'በተደጋጋሚ', 'ጥሪው', 'ደርሷት')`

---

### 1.3 Probabilities of N-grams & Top 10 Most Likely
Probabilities are calculated strictly by aggregate count frequency across all n-grams:
- **Unigram Joint Probability:** $P(w) = \frac{\text{Count}(w)}{N}$
- **N-gram Joint Probability:** $P(w_1, \dots, w_n) = \frac{\text{Count}(w_1, \dots, w_n)}{N_n}$
- **Conditional Probability:** $P(w_n | w_1, \dots, w_{n-1}) = \frac{\text{Count}(w_1, \dots, w_n)}{\text{Count}(w_1, \dots, w_{n-1})}$

#### Top 10 Most Frequent N-grams:
| Rank | Unigrams ($n=1$) | Count | Bigrams ($n=2$) | Count | Trigrams ($n=3$) | Count | 4-grams ($n=4$) | Count |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| 1 | **ነው** | 13,615 | **ነገር ግን** | 504 | **ሀሁ በስድስት ወር** | 20 | **በብርሃንና ሰላም ማተሚያ ቤት** | 11 |
| 2 | **ላይ** | 6,716 | **ማለት ነው** | 450 | **ከእለታት አንድ ቀን** | 16 | **አዲስ አበባ ስፖርት ማህበር** | 11 |
| 3 | **ወደ** | 3,805 | **ብቻ ነው** | 398 | **ሀሁ ወይም ፐፑ** | 15 | **ሚኒስር የእናቶች ጤና አማካሪ** | 10 |
| 4 | **ግን** | 3,616 | **ብቻ ሳይሆን** | 386 | **በኢትዮጵያ ኦርቶዶክስ ተዋህዶ** | 14 | **በሀገር ፍቅር ትያትር ቤት** | 10 |
| 5 | **ነበር** | 3,432 | **ሚሊዮን ዶላር** | 256 | **በኢትዮጵያ ፕሪሚየር ሊግ** | 14 | **የአለም የኢኮኖሚ ቀውስና የነዳጅ** | 10 |
| 6 | **ውስጥ** | 3,394 | **አዲስ አበባ** | 250 | **በአዲስ አበባ ከተማ** | 14 | **የኢትዮጵያ የጽንስና ማህጸን ሀኪሞች** | 10 |
| 7 | **እና** | 3,276 | **ምን ያህል** | 220 | **የህዝብ ተወካዮች ምክር** | 14 | **የተባበሩት መንግስታት የልማት ፕሮግራም** | 10 |
| 8 | **ጋር** | 2,890 | **በአዲስ አበባ** | 171 | **የኢፌዲሪ የህዝብ ተወካዮች** | 14 | **የካቲት 12 ሆስፒታል ሜዲካል** | 10 |
| 9 | **ነገር** | 2,669 | **በሌላ በኩል** | 166 | **የኢትዮጵያ አየር መንገድ** | 14 | **ጋር ሆኖ ኢህአዴግን ወጋ** | 10 |
| 10 | **ጊዜ** | 2,284 | **አንድ ቀን** | 162 | **የተባበሩት መንግስታት ድርጅት** | 14 | **አንትሂልስ ኦፍ ዚ ሳቫና** | 9 |

---

### 1.4 Bigram Conditional Probability $P(w_i | w_{i-1})$
Formula:
$$P(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$$

| Previous Word ($w_{i-1}$) | Current Word ($w_i$) | Bigram Count $\text{Count}(w_{i-1}, w_i)$ | Unigram Count $\text{Count}(w_{i-1})$ | Conditional Probability $P(w_i \| w_{i-1})$ |
| :--- | :--- | :---: | :---: | :---: |
| **ኢትዮጵያ** | **ውስጥ** | 71 | 1,446 | **0.049101** |
| **ኢትዮጵያ** | **በአፍሪካ** | 1 | 1,446 | **0.000692** |
| **አዲስ** | **አበባ** | 250 | 863 | **0.289687** |
| **የቡና** | **ምርት** | 0 | 43 | **0.000000** |
| **የቡና** | **ቤት** | 8 | 43 | **0.186047** |
| **የትምህርት** | **ጥራት** | 4 | 128 | **0.031250** |
| **የህዝብ** | **ተወካዮች** | 3 | 190 | **0.015789** |
| **ጠቅላይ** | **ሚኒስትር** | 79 | 171 | **0.461988** |
| **የተባበሩት** | **መንግስታት** | 34 | 45 | **0.755556** |

---

### 1.5 Stopword Removal & Recomputed Frequencies
Filtered out 120 common Amharic functional words (copulas, prepositions, conjunctions). Top content n-grams emerge:

#### Top 10 N-grams After Stopword Removal:
| Rank | Filtered Unigrams ($n=1$) | Count | Filtered Bigrams ($n=2$) | Count | Filtered Trigrams ($n=3$) | Count | Filtered 4-grams ($n=4$) | Count |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| 1 | **ሰው** | 1,973 | **ሚሊዮን ዶላር** | 256 | **ሀሁ ስድስት ወር** | 20 | **ብርሃንና ሰላም ማተሚያ ቤት** | 11 |
| 2 | **አገር** | 1,605 | **አዲስ አበባ** | 250 | **ፕ ት ግርማ** | 19 | **አዲስ አበባ ስፖርት ማህበር** | 11 |
| 3 | **ኢትዮጵያ** | 1,446 | **ቢሊዮን ብር** | 148 | **ሀሁ ፐፑ** | 15 | **ሚኒስር የእናቶች ጤና አማካሪ** | 10 |
| 4 | **ቤት** | 1,328 | **ሚሊዮን ብር** | 147 | **ኢትዮጵያ ፕሪሚየር ሊግ** | 14 | **ሀገር ፍቅር ትያትር ቤት** | 10 |
| 5 | **መንግስት** | 1,288 | **ህዝብ ተወካዮች** | 127 | **ኢፌዲሪ ህዝብ ተወካዮች** | 14 | **የአለም ኢኮኖሚ ቀውስና ነዳጅ** | 10 |
| 6 | **ህዝብ** | 1,213 | **ኢፌዲሪ ህዝብ** | 117 | **ኢትዮጵያ አየር መንገድ** | 14 | **ኢትዮጵያ የጽንስና ማህጸን ሀኪሞች** | 10 |
| 7 | **ስራ** | 1,173 | **ተወካዮች ምክር** | 115 | **የካቲት 12 ሆስፒታል** | 13 | **የተባበሩት መንግስታት ልማት ፕሮግራም** | 10 |
| 8 | **ነገሮች** | 1,061 | **አቶ መለስ** | 113 | **ህዝብ ተወካዮች ምክር** | 12 | **የካቲት 12 ሆስፒታል ሜዲካል** | 10 |
| 9 | **ነገር** | 993 | **ጠቅላይ ሚኒስትር** | 108 | **ተወካዮች ምክር ቤት** | 12 | **ሆኖ ኢህአዴግን ወጋ** | 10 |
| 10 | **ከተማ** | 973 | **ዶ/ር አብይ** | 98 | **ኢትዮጵያ ኦርቶዶክስ ተዋህዶ** | 12 | **አንትሂልስ ኦፍ ዚ ሳቫና** | 9 |

---

### 1.6 Word Cloud Visualizations
Word clouds for $n = 1, 2, 3, 4$ were generated and saved using `NotoSansEthiopic-Regular.ttf` in the `wordclouds/` directory:
- `wordclouds/wordcloud_n1_before_stopwords.png`
- `wordclouds/wordcloud_n2_before_stopwords.png`
- `wordclouds/wordcloud_n3_before_stopwords.png`
- `wordclouds/wordcloud_n4_before_stopwords.png`
- `wordclouds/wordcloud_n1_after_stopwords.png`
- `wordclouds/wordcloud_n2_after_stopwords.png`
- `wordclouds/wordcloud_n3_after_stopwords.png`
- `wordclouds/wordcloud_n4_after_stopwords.png`

---

### 1.7 Sentence Probability: *"ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።"*
Evaluated using log-space formulation with Add-k smoothing ($k=0.1$) to prevent zero-probability underflow:
$$\log_2 P(S) = \sum_{i=n}^M \log_2 P(w_i | w_{i-n+1}^{i-1})$$

| Sentence Description | Model | $\log_2 P(S)$ [Smoothed] | $P(S)$ [Smoothed] | $P(S)$ [Raw MLE] |
| :--- | :--- | :---: | :---: | :---: |
| **"ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።"** *(Target)* | Unigram ($n=1$) | -74.0061 | $5.27 \times 10^{-23}$ | 0.0000 |
| | Bigram ($n=2$) | -68.9339 | $1.77 \times 10^{-21}$ | 0.0000 |
| | Trigram ($n=3$) | -51.6646 | $2.80 \times 10^{-16}$ | 0.0000 |
| | 4-gram ($n=4$) | -34.4431 | $4.28 \times 10^{-11}$ | 0.0000 |
| **"አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።"** *(Fact)* | 4-gram ($n=4$) | -42.8124 | $1.36 \times 10^{-13}$ | 0.0000 |
| **"ድንጋይ ሰማይ ውሃ ሩጫ በላ።"** *(Nonsense)* | 4-gram ($n=4$) | -99.9576 | $8.12 \times 10^{-31}$ | 0.0000 |

> **Key Finding:** Raw MLE yields $0.0000$ because the high-order transition `"ቀንድ ትገኛለች"` was never observed consecutively in this training partition. Smoothed probability rewards the grammatical sentence with $10^{20}\times$ higher probability than the ungrammatical word salad.

---

### 1.8 Random Sentence Generation & Linguistic Progression
- **Unigram ($n=1$):**
  > *"ፍሬዎቹን በምን ከሌላው ሊክደው ባህር የቀራት መጀመርና 6ኛው ጠርቀም ማደያ ኣለም ፈንታ ህዝባቸውን።"*  
  **Linguistic Explanation:** Bag-of-words random sampling. Totally devoid of syntax, morphological agreement, or semantic sense.
- **Bigram ($n=2$):**
  > *"ልክ እኮ ነው ብዬ ነው የተረፍኩት።"*  
  **Linguistic Explanation:** Adjacent pairs are coherent collocations, but long-range structure drifts rapidly.
- **Trigram ($n=3$):**
  > *"ኳስ ዋ ለ19ኛ ጊዜ በደቡብ አፍሪካ የተቀረጸ ትእይንት ተካቶበታል።"*  
  **Linguistic Explanation:** Coherent clausal structure, preserving Subject-Object-Verb (SOV) agreement over 3-word windows.
- **4-gram ($n=4$):**
  > *"ግለሰቦች በፈታኝ ሁኔታ ውስጥ በማለፍ ብቃታቸውን የሚያስመሰክሩበት መሆኑን ሰምተናል።"*  
  **Linguistic Explanation:** Flawless grammatical Amharic syntax, but shows high corpus memorization due to combinatorial sparsity.

---

## Question #2: Intrinsic Evaluation (Perplexity)

Intrinsic evaluation measures how well the probability distribution predicts unseen held-out text.
$$\text{Cross-Entropy: } H(W) = -\frac{1}{N} \sum_{i=1}^N \log_2 P(w_i | w_{i-n+1}^{i-1})$$
$$\text{Perplexity: } PPL(W) = 2^{H(W)}$$

### Benchmark Results Table (Held-Out Test Set):
| Model | Smoothing Algorithm | Cross-Entropy (bits/token) | Perplexity ($PPL$) |
| :--- | :--- | :---: | :---: |
| **Unigram ($n=1$)** | Laplace (Add-1) | 14.4425 | **22,265.75** |
| **Unigram ($n=1$)** | Add-k ($k=0.1$) | 14.7574 | **27,697.09** |
| **Unigram ($n=1$)** | Add-k ($k=0.01$) | 15.1909 | **37,404.49** |
| **Bigram ($n=2$)** | Laplace (Add-1) | 16.3783 | **85,185.95** |
| **Bigram ($n=2$)** | Add-k ($k=0.1$) | 15.6087 | **49,968.17** |
| **Bigram ($n=2$)** | Add-k ($k=0.01$) | 14.8822 | **30,198.43** |
| **Trigram ($n=3$)** | Laplace (Add-1) | 16.8784 | **120,473.34** |
| **Trigram ($n=3$)** | Add-k ($k=0.1$) | 16.6316 | **101,535.65** |
| **Trigram ($n=3$)** | Add-k ($k=0.01$) | 16.3361 | **82,728.84** |
| **4-gram ($n=4$)** | Laplace (Add-1) | 16.9221 | **124,178.25** |
| **4-gram ($n=4$)** | Add-k ($k=0.1$) | 16.7468 | **109,975.27** |
| **4-gram ($n=4$)** | Add-k ($k=0.01$) | 16.5344 | **94,916.42** |
| **Interpolated** | Jelinek-Mercer | 17.9576 | **254,555.66** |

### Analysis:
1. **The Large Vocabulary Dilemma:** With $|V| \approx 132,000$, Laplace (Add-1) assigns an enormous amount of probability mass to zero counts ($1 \times 132,000$), severely penalizing true observations.
2. **Add-k Superiority:** Reducing $k$ to $0.01$ dramatically curbs the probability redistribution penalty, lowering bigram perplexity from $85,185$ down to $30,198$.

---

## Question #3: Extrinsic Evaluation (Downstream NLP Tasks)

Evaluates the models on practical downstream NLP applications.

### Task 1: Next-Word Prediction & Ranking (Cloze Benchmark)
1,500 held-out prediction contexts were evaluated:
| Model | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Mean Reciprocal Rank (MRR) | Latency (ms/query) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unigram ($n=1$)** | 1.07% | 1.93% | 2.47% | 0.0186 | **0.006 ms** |
| **Bigram ($n=2$)** | 7.33% | 11.93% | 14.67% | 0.1035 | **0.093 ms** |
| **Trigram ($n=3$)** | 8.73% | 10.27% | 10.87% | 0.0984 | **0.010 ms** |
| **4-gram ($n=4$)** | 6.07% | 6.93% | 7.47% | 0.0684 | **0.009 ms** |
| **Backoff Model** | **11.80%** | **15.53%** | **17.87%** | **0.1419** | **0.051 ms** |

### Task 2: Sentence Fluency Discrimination
Assesses whether $P(S_{\text{real}}) > P(S_{\text{corrupt}})$ when a random middle word is corrupted:
- **Unigram ($n=1$):** 76.27% accuracy
- **Bigram ($n=2$):** 56.14% accuracy
- **Trigram ($n=3$):** 12.71% accuracy (due to unseen corrupted bigram/trigram contexts in unbacked model)

> **Conclusion:** The Hierarchical Backoff Model achieves the highest downstream performance (17.87% Top-5 accuracy and 0.1419 MRR) because it combines high-order specificity with low-order coverage.

---

## Question #4: Amharic Predictive Text System

### System Architecture
The application implements all required capabilities:
1. Accepts user Amharic input string.
2. Extracts final 1 or 2 words.
3. Retrieves candidates using Katz Backoff (Trigram $\rightarrow$ Bigram $\rightarrow$ Unigram).
4. Calculates candidate conditional probability distribution.
5. Returns the Top-5 suggestions.
6. Allows interactive selection (by index 1-5) to compose sentences iteratively.

### PDF Assignment Benchmark Verification:
**Input:** `"ኢትዮጵያ ውስጥ የቡና"`
```
Top 5 Next-Word Suggestions Generated by the Language Model:
Rank   | Candidate Word     | Estimated Probability  | Confidence
-----------------------------------------------------------------
  #1   | መገኛ                | 0.3841                 |  38.4% ███████
  #2   | ቤት                 | 0.3206                 |  32.1% ██████
  #3   | ላይ                 | 0.0984                 |   9.8% █
  #4   | መረጃ                | 0.0984                 |   9.8% █
  #5   | ጀበና                | 0.0984                 |   9.8% █
```
**Interactive Composition Demonstration:**
- Step 0: `"ኢትዮጵያ ውስጥ የቡና"`
- Step 1: Select `መገኛ` $\rightarrow$ `"ኢትዮጵያ ውስጥ የቡና መገኛ"`
- Step 2: Select `መሆኗን` $\rightarrow$ `"ኢትዮጵያ ውስጥ የቡና መገኛ መሆኗን"`
- Step 3: Select `ነው` $\rightarrow$ `"ኢትዮጵያ ውስጥ የቡና መገኛ መሆኗን ነው"`

---

## 🚀 How to Run the Code

### 1. Run Master Pipeline (Questions 1 to 4 Sequentially)
```bash
python3 "/home/genesis/NLP assignment/run_all.py"
```

### 2. Run Individual Questions
```bash
# Question 1: N-gram Modeling, Statistics, Wordclouds, Probabilities
python3 "/home/genesis/NLP assignment/q1_ngram_language_model.py" --sentences 60000

# Question 2: Intrinsic Evaluation (Perplexity)
python3 "/home/genesis/NLP assignment/q2_intrinsic_evaluation.py" --sentences 60000

# Question 3: Extrinsic Evaluation (Next-Word Downstream Benchmark)
python3 "/home/genesis/NLP assignment/q3_extrinsic_evaluation.py" --sentences 60000

# Question 4: Predictive System Verification
python3 "/home/genesis/NLP assignment/q4_predictive_system.py"

# Question 4: Interactive CLI Mode
python3 "/home/genesis/NLP assignment/q4_predictive_system.py" --interactive
```

### 3. Launch Interactive Streamlit Web App
```bash
streamlit run "/home/genesis/NLP assignment/app_streamlit.py"
```
