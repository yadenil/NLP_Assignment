"""
Master Execution Script for NLP Assignment (Tasks 1 - 4)
=======================================================
Addis Ababa University | College of Technology & Built Environment
Department of Information Technology and Engineering
Instructor: Fantahun B. (PhD)
Course: Natural Language Processing (NLP)

This master script runs all 4 assignment questions in sequence:
  #1 N-gram Language Modeling (GPAC Statistics, N-grams, Probabilities, Stopwords, Wordclouds, Generation)
  #2 Intrinsic Evaluation (Perplexity on Held-out Test Set across n=1..4 with Smoothing)
  #3 Extrinsic Evaluation (Next-Word Downstream Prediction & Sentence Fluency Discrimination)
  #4 Practical Predictive Text & Next-Word Suggestion System
"""

import os
import sys
import time
import argparse

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from q1_ngram_language_model import run_question_1
from q2_intrinsic_evaluation import run_question_2
from q3_extrinsic_evaluation import run_question_3
from q4_predictive_system import run_question_4_verification

DEFAULT_CORPUS_PATH = "/home/genesis/Downloads/GPAC.txt"


def main():
    parser = argparse.ArgumentParser(description="Run Full NLP Assignment (Questions 1 - 4)")
    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS_PATH, help="Path to GPAC.txt")
    parser.add_argument("--sentences", type=int, default=80000, help="Number of sentences to sample for training")
    args = parser.parse_args()

    overall_start = time.time()
    
    print("\n" + "#" * 80)
    print("      ADDIS ABABA UNIVERSITY - SCHOOL OF INFORMATION TECHNOLOGY & ENGINEERING")
    print("                        NLP ASSIGNMENT 1 (TASKS #1 - #4)")
    print("                     Corpus: GPAC (General-Purpose Amharic Corpus)")
    print("#" * 80 + "\n")

    # Question 1
    t0 = time.time()
    print("\n>>> EXECUTING QUESTION #1: N-GRAM LANGUAGE MODELING (6%)...\n")
    run_question_1(corpus_path=args.corpus, max_sentences=args.sentences)
    print(f"[✓] Question #1 completed in {time.time() - t0:.2f} seconds.")

    # Question 2
    t1 = time.time()
    print("\n>>> EXECUTING QUESTION #2: INTRINSIC EVALUATION (3%)...\n")
    run_question_2(corpus_path=args.corpus, max_sentences=int(args.sentences * 0.75))
    print(f"[✓] Question #2 completed in {time.time() - t1:.2f} seconds.")

    # Question 3
    t2 = time.time()
    print("\n>>> EXECUTING QUESTION #3: EXTRINSIC EVALUATION (5%)...\n")
    run_question_3(corpus_path=args.corpus, max_sentences=int(args.sentences * 0.75))
    print(f"[✓] Question #3 completed in {time.time() - t2:.2f} seconds.")

    # Question 4
    t3 = time.time()
    print("\n>>> EXECUTING QUESTION #4: AMHARIC PREDICTIVE TEXT SYSTEM (6%)...\n")
    run_question_4_verification()
    print(f"[✓] Question #4 completed in {time.time() - t3:.2f} seconds.")

    total_time = time.time() - overall_start
    print("\n" + "=" * 80)
    print(f"🎉 ALL 4 NLP ASSIGNMENT QUESTIONS EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 80)
    print(f"Generated artifacts:")
    print(f"  • Word Clouds:  {os.path.join(CURRENT_DIR, 'wordclouds')}")
    print(f"  • Eval Plots:   {os.path.join(CURRENT_DIR, 'plots')}")
    print(f"  • JSON Logs:    {os.path.join(CURRENT_DIR, 'results')}")
    print(f"  • Cached Model: {os.path.join(CURRENT_DIR, 'models')}")
    print("\nTo launch the interactive Web App for Question 4:")
    print(f"  streamlit run \"{os.path.join(CURRENT_DIR, 'app_streamlit.py')}\"")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
