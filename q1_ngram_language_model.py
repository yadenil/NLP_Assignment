"""
Question #1: N-gram Language Modeling on Amharic Corpus (GPAC)
==============================================================
Addis Ababa University | College of Technology & Built Environment
Course: Natural Language Processing (NLP)

This script performs all requirements for Question 1:
  1.1. Compute corpus statistics and relevant Amharic preprocessing.
  1.2. Create n-grams for n = 1, 2, 3, 4 with sample prints.
  1.3. Calculate n-gram probabilities and find top 10 most likely n-grams for all n.
  1.4. Calculate conditional probability of a word given the previous word using bigrams.
  1.5. Remove common stopwords and recompute frequencies for n = 1, 2, 3, 4.
  1.6. Create and save word clouds for n-grams (n = 1-4) before and after stopword removal.
  1.7. Compute sentence probabilities for "ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።" and additional sentences.
  1.8. Generate random sentences using n-grams and explain what happens as n increases.
"""

import os
import sys
import json
import warnings
import argparse
from collections import Counter
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Suppress harmless font glyph fallback warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Import our custom Amharic preprocessor and N-gram model
from amharic_preprocessor import (
    SENT_SPLIT,
    normalize_amharic,
    tokenize_amharic_sentence,
    remove_stopwords,
    stream_sentences_from_corpus,
    AMHARIC_STOPWORDS
)
from ngram_model import NGramLanguageModel

# Ensure directories exist
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
WORDCLOUDS_DIR = os.path.join(OUTPUT_DIR, "wordclouds")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(WORDCLOUDS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ETHIOPIC_FONT = "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf"
DEFAULT_CORPUS_PATH = "/home/genesis/Downloads/GPAC.txt"


def format_ngram(ngram: Tuple[str, ...]) -> str:
    """Formats an n-gram tuple as a space-separated readable string."""
    return " ".join(ngram)


def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extracts pure word n-grams from sentence tokens without <s> or </s> sequence padding."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def run_question_1(corpus_path: str = DEFAULT_CORPUS_PATH, max_sentences: int = 100000):
    print("=" * 80)
    print("       ADDIS ABABA UNIVERSITY - NLP ASSIGNMENT: QUESTION #1")
    print("                 N-GRAM LANGUAGE MODELING (GPAC)")
    print("=" * 80)
    print(f"[*] Corpus Path: {corpus_path}")
    print(f"[*] Target Sample Sentences: {max_sentences:,}\n")

    # -----------------------------------------------------------------------
    # 1.1 Compute Corpus Statistics and Relevant Preprocessing
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1.1 CORPUS STATISTICS AND RELEVANT PREPROCESSING")
    print("=" * 80)
    
    raw_sentences = []
    tokenized_sentences = []
    filtered_sentences = []  # Sentences with stopwords removed
    
    total_raw_chars = 0
    total_cleaned_chars = 0
    
    print("[*] Streaming and segmenting sentences using SENT_SPLIT = re.compile(r'(?:፡፡)+|።+|[!?]+')...")
    for s_raw in stream_sentences_from_corpus(corpus_path, max_sentences=max_sentences):
        total_raw_chars += len(s_raw)
        raw_sentences.append(s_raw)
        
        # Tokenize and normalize
        tokens = tokenize_amharic_sentence(s_raw)
        if tokens:
            tokenized_sentences.append(tokens)
            total_cleaned_chars += sum(len(t) for t in tokens)
            
            # Stopword filtered version
            sw_filtered = remove_stopwords(tokens)
            if sw_filtered:
                filtered_sentences.append(sw_filtered)

    total_sentences = len(tokenized_sentences)
    all_tokens = [token for sent in tokenized_sentences for token in sent]
    total_words = len(all_tokens)
    vocab_counter = Counter(all_tokens)
    vocab_size = len(vocab_counter)
    
    avg_sentence_len_words = total_words / total_sentences if total_sentences else 0
    avg_sentence_len_chars = total_cleaned_chars / total_sentences if total_sentences else 0
    type_token_ratio = (vocab_size / total_words) if total_words else 0
    hapax_legomena = sum(1 for _, count in vocab_counter.items() if count == 1)
    hapax_pct = (hapax_legomena / vocab_size * 100) if vocab_size else 0

    stats = {
        "Total Sampled Sentences": total_sentences,
        "Total Words (Tokens)": total_words,
        "Vocabulary Size (Unique Types)": vocab_size,
        "Raw Character Count": total_raw_chars,
        "Cleaned Character Count": total_cleaned_chars,
        "Average Sentence Length (Words)": round(avg_sentence_len_words, 2),
        "Average Sentence Length (Chars)": round(avg_sentence_len_chars, 2),
        "Type-Token Ratio (TTR / Lexical Diversity)": round(type_token_ratio, 4),
        "Hapax Legomena (Single-occurrence words)": hapax_legomena,
        "Hapax Legomena Percentage": f"{hapax_pct:.2f}%"
    }

    print("\n--- Corpus Statistics Table ---")
    for k, v in stats.items():
        if isinstance(v, int):
            print(f"  • {k:<45}: {v:,}")
        else:
            print(f"  • {k:<45}: {v}")

    # -----------------------------------------------------------------------
    # 1.2 Create N-grams for n = 1, 2, 3, 4 (Sample Prints)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1.2 CREATE N-GRAMS FOR n = 1, 2, 3, 4 (SAMPLE PRINTS)")
    print("=" * 80)
    
    sample_sentence = tokenized_sentences[0] if tokenized_sentences else ["ኢትዮጵያ", "በአፍሪካ", "ቀንድ", "ትገኛለች"]
    print(f"Sample Sentence Tokens: {sample_sentence}")
    
    for n in [1, 2, 3, 4]:
        label = {1: "Unigrams (n=1)", 2: "Bigrams (n=2)", 3: "Trigrams (n=3)", 4: "4-grams (n=4)"}[n]
        ngrams = extract_ngrams(sample_sentence, n)
        print(f"\n[{label}] (Total extracted: {len(ngrams)}):")
        for i, ng in enumerate(ngrams[:6], 1):
            print(f"   {i}. {format_ngram(ng)}")
        if len(ngrams) > 6:
            print(f"   ... and {len(ngrams) - 6} more")

    # Train models for n = 1, 2, 3, 4 without <s> padding tags
    print("\n[*] Training N-Gram Language Models for n = 1, 2, 3, 4 (without <s> sequence tags)...")
    models: Dict[int, NGramLanguageModel] = {}
    for n in [1, 2, 3, 4]:
        model = NGramLanguageModel(n=n, use_padding=False)
        model.train(tokenized_sentences)
        models[n] = model
        print(f"  • {n}-gram model trained: {len(model.n_gram_counts):,} unique {n}-grams counted.")

    # -----------------------------------------------------------------------
    # 1.3 Calculate Probabilities of N-grams & Top 10 Most Likely
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1.3 TOP 10 MOST LIKELY N-GRAMS AND PROBABILITIES (n = 1, 2, 3, 4)")
    print("=" * 80)
    
    top_ngrams_summary = {}
    for n in [1, 2, 3, 4]:
        model = models[n]
        total_n = model.total_ngrams
        # Sort strictly by aggregate frequency count descending (with deterministic tie-breaker)
        sorted_ngrams = sorted(model.n_gram_counts.items(), key=lambda item: (-item[1], item[0]))
        top_10 = sorted_ngrams[:10]
        
        label = {1: "Unigrams (n=1)", 2: "Bigrams (n=2)", 3: "Trigrams (n=3)", 4: "4-grams (n=4)"}[n]
        print(f"\n--- Top 10 Most Likely {label} (Total Occurrences: {total_n:,}) ---")
        print(f"{'Rank':<5} | {'N-Gram':<38} | {'Frequency':<10} | {'Joint P(w)':<14} | {'Conditional P(w|ctx)':<20}")
        print("-" * 95)
        
        ngram_list = []
        for rank, (ng, count) in enumerate(top_10, 1):
            joint_prob = count / total_n if total_n else 0.0
            cond_prob = model.mle_probability(ng[-1], ng[:-1])
            ng_str = format_ngram(ng)
            print(f"{rank:<5} | {ng_str:<38} | {count:<10,} | {joint_prob:<14.6f} | {cond_prob:<20.6f}")
            ngram_list.append({
                "rank": rank,
                "ngram": ng_str,
                "count": count,
                "joint_probability": joint_prob,
                "conditional_probability": cond_prob
            })
        top_ngrams_summary[f"top_{n}grams"] = ngram_list

    # -----------------------------------------------------------------------
    # 1.4 Calculate Bigram Conditional Probability P(w_i | w_{i-1})
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1.4 CONDITIONAL PROBABILITY OF A WORD GIVEN THE PREVIOUS WORD P(w_i | w_{i-1})")
    print("=" * 80)
    print("Formula: P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})\n")

    bigram_model = models[2]

    def get_conditional_prob(w_prev: str, w_curr: str) -> Tuple[int, int, float]:
        """Returns (bigram_count, unigram_count, conditional_probability)."""
        c_prev = bigram_model.unigram_counts.get(w_prev, 0)
        c_bi = bigram_model.n_gram_counts.get((w_prev, w_curr), 0)
        p = (c_bi / c_prev) if c_prev > 0 else 0.0
        return c_bi, c_prev, p

    sample_word_pairs = [
        ("ኢትዮጵያ", "ውስጥ"),
        ("ኢትዮጵያ", "በአፍሪካ"),
        ("አዲስ", "አበባ"),
        ("የቡና", "ምርት"),
        ("የቡና", "ቤት"),
        ("የትምህርት", "ጥራት"),
        ("የህዝብ", "ተወካዮች"),
        ("ጠቅላይ", "ሚኒስትር"),
        ("የተባበሩት", "መንግስታት"),
    ]

    print(f"{'w_{i-1}':<15} | {'w_i':<15} | {'Count(w_{i-1}, w_i)':<20} | {'Count(w_{i-1})':<15} | {'P(w_i | w_{i-1})':<18}")
    print("-" * 90)
    for w_prev, w_curr in sample_word_pairs:
        c_bi, c_prev, p = get_conditional_prob(w_prev, w_curr)
        print(f"{w_prev:<15} | {w_curr:<15} | {c_bi:<20,} | {c_prev:<15,} | {p:<18.6f}")

    # -----------------------------------------------------------------------
    # 1.5 Remove Common Stopwords and Recompute N-gram Frequencies (n=1,2,3,4)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1.5 REMOVE COMMON STOPWORDS AND RECOMPUTE N-GRAM FREQUENCIES (n = 1, 2, 3, 4)")
    print("=" * 80)
    print(f"[*] Total Stopwords in Filter List: {len(AMHARIC_STOPWORDS)}")
    print(f"[*] Sentences after stopword filtering: {len(filtered_sentences):,}")

    filtered_models: Dict[int, NGramLanguageModel] = {}
    for n in [1, 2, 3, 4]:
        f_model = NGramLanguageModel(n=n, use_padding=False)
        f_model.train(filtered_sentences)
        filtered_models[n] = f_model

    top_filtered_summary = {}
    for n in [1, 2, 3, 4]:
        f_model = filtered_models[n]
        total_fn = f_model.total_ngrams
        # Sort strictly by aggregate frequency count descending (with deterministic tie-breaker)
        sorted_filtered = sorted(f_model.n_gram_counts.items(), key=lambda item: (-item[1], item[0]))
        top_10 = sorted_filtered[:10]
        
        label = {1: "Unigrams (n=1)", 2: "Bigrams (n=2)", 3: "Trigrams (n=3)", 4: "4-grams (n=4)"}[n]
        print(f"\n--- Top 10 Most Likely {label} (AFTER Stopword Removal) ---")
        print(f"{'Rank':<5} | {'N-Gram':<38} | {'Frequency':<10} | {'Joint P(w)':<14}")
        print("-" * 75)
        
        f_list = []
        for rank, (ng, count) in enumerate(top_10, 1):
            p = count / total_fn if total_fn else 0.0
            ng_str = format_ngram(ng)
            print(f"{rank:<5} | {ng_str:<38} | {count:<10,} | {p:<14.6f}")
            f_list.append({"rank": rank, "ngram": ng_str, "count": count, "probability": p})
        top_filtered_summary[f"filtered_top_{n}grams"] = f_list

    # -----------------------------------------------------------------------
    # 1.6 Create Word Clouds Before and After Stopword Removal
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1.6 CREATE WORD CLOUDS FOR N-GRAMS (n = 1-4) BEFORE AND AFTER STOPWORD REMOVAL")
    print("=" * 80)
    
    def generate_and_save_wordcloud(freq_dict: Dict[str, int], title: str, save_path: str):
        """Generates a high-resolution Amharic wordcloud image and saves it to disk."""
        if not freq_dict:
            return
        
        # Filter special boundary tokens from display
        clean_freq = {
            k: v for k, v in freq_dict.items()
            if NGramLanguageModel.START_TOKEN not in k and NGramLanguageModel.END_TOKEN not in k
        }
        if not clean_freq:
            clean_freq = freq_dict

        wc = WordCloud(
            font_path=ETHIOPIC_FONT,
            width=1000,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=120,
            random_state=42
        )
        wc.generate_from_frequencies(clean_freq)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=16, pad=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [✓] Saved: {os.path.basename(save_path)}")

    print("[*] Generating 8 Word Cloud Visualizations...")
    # 1. Before Stopwords
    for n in [1, 2, 3, 4]:
        raw_freq = {format_ngram(k): v for k, v in models[n].n_gram_counts.items()}
        title = f"GPAC Amharic {n}-grams (BEFORE Stopword Removal)"
        path = os.path.join(WORDCLOUDS_DIR, f"wordcloud_n{n}_before_stopwords.png")
        generate_and_save_wordcloud(raw_freq, title, path)

    # 2. After Stopwords
    for n in [1, 2, 3, 4]:
        filtered_freq = {format_ngram(k): v for k, v in filtered_models[n].n_gram_counts.items()}
        title = f"GPAC Amharic {n}-grams (AFTER Stopword Removal)"
        path = os.path.join(WORDCLOUDS_DIR, f"wordcloud_n{n}_after_stopwords.png")
        generate_and_save_wordcloud(filtered_freq, title, path)

    # -----------------------------------------------------------------------
    # 1.7 Probability of Sentence: "ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።" + Extra Sentences
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1.7 SENTENCE PROBABILITY COMPUTATION")
    print("=" * 80)
    
    test_sentences = [
        ("Target Assignment Sentence", "ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።"),
        ("Common Fact Sentence", "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።"),
        ("Educational Sentence", "ተማሪዎች በትምህርት ቤት ውስጥ ዕውቀት ይገበያሉ።"),
        ("Ungrammatical / Word Salad", "ድንጋይ ሰማይ ውሃ ሩጫ በላ።")
    ]

    sentence_prob_results = []
    for sent_desc, raw_s in test_sentences:
        tokens = tokenize_amharic_sentence(raw_s)
        print(f"\nSentence Description: {sent_desc}")
        print(f"Original Text: {raw_s}")
        print(f"Processed Tokens: {tokens}")
        print(f"{'Model':<15} | {'Log2 P(S) [Smoothed]':<22} | {'P(S) [Smoothed]':<20} | {'P(S) [Raw MLE]':<15}")
        print("-" * 78)
        
        entry = {"description": sent_desc, "sentence": raw_s, "tokens": tokens, "models": {}}
        for n in [1, 2, 3, 4]:
            model = models[n]
            # Smoothed probability (Add-k smoothing with k=0.1 avoids zero underflow)
            log2_prob_smooth = model.log_sentence_probability(tokens, smoothing="add_k", alpha=0.1)
            prob_smooth = model.sentence_probability(tokens, smoothing="add_k", alpha=0.1)
            # Unsmoothed raw MLE
            prob_mle = model.sentence_probability(tokens, smoothing="mle")
            
            label = f"{n}-gram (n={n})"
            print(f"{label:<15} | {log2_prob_smooth:<22.4f} | {prob_smooth:<20.4e} | {prob_mle:<15.4e}")
            entry["models"][label] = {
                "log2_probability_smooth": log2_prob_smooth,
                "probability_smooth": prob_smooth,
                "probability_mle": prob_mle
            }
        sentence_prob_results.append(entry)

    # -----------------------------------------------------------------------
    # 1.8 Random Sentence Generation using N-grams & Linguistic Analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1.8 RANDOM SENTENCE GENERATION (n = 1, 2, 3, 4) AND ANALYSIS")
    print("=" * 80)
    
    generated_sentences_dict = {}
    for n in [1, 2, 3, 4]:
        label = {1: "Unigram (n=1)", 2: "Bigram (n=2)", 3: "Trigram (n=3)", 4: "4-gram (n=4)"}[n]
        print(f"\n[{label} Generated Sentences]:")
        gens = []
        for i in range(3):
            # Sample with temperature=0.85
            gen_tokens = models[n].generate_sentence(max_length=15, temperature=0.85, seed=42 + i * 17 + n * 99)
            gen_str = " ".join(gen_tokens) + "።"
            print(f"  Sentence {i+1}: {gen_str}")
            gens.append(gen_str)
        generated_sentences_dict[label] = gens

    analysis_text = """
================================================================================
                    LINGUISTIC & COMPUTATIONAL ANALYSIS
              WHAT HAPPENS AS n INCREASES IN N-GRAM MODELS?
================================================================================
1. n = 1 (Unigram Model):
   - The unigram model ignores word order entirely; it treats text as an exchangeable
     "bag-of-words" and generates words proportionally to their overall frequency.
   - Result: Utter lack of syntax or semantic coherence ("word salad"). Verbs, nouns,
     and prepositions appear in arbitrary order with zero grammatical agreement.

2. n = 2 (Bigram Model):
   - Bigrams condition each word solely on the immediately preceding token P(w_i | w_{i-1}).
   - Result: Local 2-word collocations are preserved (e.g. valid noun-preposition or
     verb pairs), but the sentence quickly drifts in topic and lacks global sentence
     structure, subject-verb concord, or tense consistency.

3. n = 3 (Trigram Model):
   - Trigrams condition each word on two preceding tokens P(w_i | w_{i-1}, w_{i-2}).
   - Result: Noticeable syntactic leap. Amharic phrases adhere well to standard
     SOV (Subject-Object-Verb) word order, noun phrase agreements, and logical
     semantic clauses over short spans.

4. n = 4 (4-gram Model):
   - 4-grams condition each word on three preceding tokens P(w_i | w_{i-1}, w_{i-2}, w_{i-3}).
   - Result: Sentences are grammatically fluent and idiomatically natural Amharic.
     However, due to combinatorial data sparsity, the model often reproduces exact
     verbatim sub-clauses from the training corpus ("memorization" vs generalization).
================================================================================
"""
    print(analysis_text)

    # Save summary results to JSON and TXT
    summary_data = {
        "corpus_statistics": stats,
        "top_ngrams": top_ngrams_summary,
        "top_filtered_ngrams": top_filtered_summary,
        "sentence_probabilities": sentence_prob_results,
        "generated_sentences": generated_sentences_dict,
        "analysis": analysis_text
    }
    json_path = os.path.join(RESULTS_DIR, "q1_results.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary_data, jf, ensure_ascii=False, indent=2)
        
    txt_path = os.path.join(RESULTS_DIR, "question1_results.txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write("=" * 80 + "\n")
        tf.write("       ADDIS ABABA UNIVERSITY - NLP ASSIGNMENT: QUESTION #1 RESULTS\n")
        tf.write("                 N-GRAM LANGUAGE MODELING (GPAC)\n")
        tf.write("=" * 80 + "\n\n")
        tf.write("1.1 CORPUS STATISTICS\n" + "-" * 40 + "\n")
        for k, v in stats.items():
            tf.write(f"  • {k:<45}: {v}\n")
        tf.write("\n1.3 TOP 10 N-GRAMS (BEFORE STOPWORDS)\n" + "-" * 40 + "\n")
        for n_key, n_list in top_ngrams_summary.items():
            tf.write(f"\n--- {n_key} ---\n")
            for item in n_list:
                tf.write(f"  #{item['rank']:<2} | {item['ngram']:<30} | Count: {item['count']:<6} | Joint P: {item['joint_probability']:.6f} | Cond P: {item['conditional_probability']:.6f}\n")
        tf.write("\n1.5 TOP 10 N-GRAMS (AFTER STOPWORD REMOVAL)\n" + "-" * 40 + "\n")
        for n_key, n_list in top_filtered_summary.items():
            tf.write(f"\n--- {n_key} ---\n")
            for item in n_list:
                tf.write(f"  #{item['rank']:<2} | {item['ngram']:<30} | Count: {item['count']:<6} | P: {item['probability']:.6f}\n")
        tf.write("\n1.7 SENTENCE PROBABILITIES\n" + "-" * 40 + "\n")
        for sent in sentence_prob_results:
            tf.write(f"\nSentence: \"{sent['sentence']}\" ({sent['description']})\n")
            for m_name, m_vals in sent['models'].items():
                tf.write(f"  • {m_name:<15}: Log2 P = {m_vals['log2_probability_smooth']:<10.4f} | Smoothed P = {m_vals['probability_smooth']:.4e} | Raw MLE P = {m_vals['probability_mle']:.4e}\n")
        tf.write("\n1.8 GENERATED SENTENCES\n" + "-" * 40 + "\n")
        for m_name, sents in generated_sentences_dict.items():
            tf.write(f"\n{m_name}:\n")
            for i, s in enumerate(sents, 1):
                tf.write(f"  {i}. {s}\n")
        tf.write("\n" + analysis_text + "\n")

    print(f"\n[✓] All Question 1 results saved to:\n  • {json_path}\n  • {txt_path}")
    print("[✓] Question 1 complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Question 1: Amharic N-gram Language Modeling")
    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS_PATH, help="Path to GPAC.txt")
    parser.add_argument("--sentences", type=int, default=100000, help="Number of sentences to process")
    args = parser.parse_args()
    
    run_question_1(corpus_path=args.corpus, max_sentences=args.sentences)
