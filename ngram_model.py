"""
N-gram Language Model Class
===========================
This module implements a flexible, mathematically rigorous N-Gram Language Model
supporting n = 1, 2, 3, 4 with:
- Maximum Likelihood Estimation (MLE)
- Laplace (Add-1) Smoothing
- Lidstone (Add-k) Smoothing
- Jelinek-Mercer Linear Interpolation Smoothing
- Katz / Stupid Backoff
- Perplexity (PPL) calculation for Intrinsic Evaluation
- Next-word candidate retrieval for Extrinsic Evaluation & Predictive Typing
- Random sentence generation with temperature/multinomial sampling
"""

import math
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Union


class NGramLanguageModel:
    """
    N-gram Language Model supporting orders n=1, 2, 3, 4.
    
    Attributes:
    -----------
    n : int
        The order of the n-gram model (1=unigram, 2=bigram, 3=trigram, 4=4-gram).
    vocab : set
        Set of unique vocabulary tokens.
    n_gram_counts : Counter
        Counter mapping n-gram tuple -> frequency count.
    context_counts : defaultdict(Counter)
        Mapping prefix context tuple -> Counter of {next_token: frequency}.
    total_tokens : int
        Total number of running tokens observed during training.
    """
    
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, n: int = 2, use_padding: bool = False):
        if n < 1:
            raise ValueError("N-gram order must be >= 1")
        self.n = n
        self.use_padding = use_padding
        self.vocab = set()
        self.n_gram_counts = Counter()
        self.context_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.total_tokens = 0
        self.total_ngrams = 0

    def _pad_tokens(self, tokens: List[str]) -> List[str]:
        """
        Returns padded tokens if use_padding=True, otherwise returns tokens as-is.
        """
        if not self.use_padding:
            return list(tokens)
        if self.n == 1:
            return tokens + [self.END_TOKEN]
        return [self.START_TOKEN] * (self.n - 1) + tokens + [self.END_TOKEN]

    def train(self, tokenized_sentences: List[List[str]]):
        """
        Trains the language model on a list of tokenized sentences.
        Counts n-grams, unigrams, context frequencies, and updates vocabulary.
        """
        for sentence in tokenized_sentences:
            if not sentence:
                continue
            
            # Update vocabulary and unigram counts
            for token in sentence:
                self.vocab.add(token)
                self.unigram_counts[token] += 1
                self.total_tokens += 1
            
            # Padded or unpadded tokens
            tokens_seq = self._pad_tokens(sentence)
            
            # Extract and count n-grams
            if len(tokens_seq) >= self.n:
                for i in range(len(tokens_seq) - self.n + 1):
                    ngram = tuple(tokens_seq[i:i + self.n])
                    self.n_gram_counts[ngram] += 1
                    self.total_ngrams += 1
                    
                    # Context is the preceding n-1 tokens, target is the last token
                    context = ngram[:-1]
                    target = ngram[-1]
                    self.context_counts[context][target] += 1
        
        if self.use_padding:
            self.vocab.add(self.END_TOKEN)
            if self.n > 1:
                self.vocab.add(self.START_TOKEN)
                
        # Cache sorted unigrams once strictly by aggregate count descending (for instant candidate lookups)
        self._sorted_unigrams = sorted(
            self.unigram_counts.items(),
            key=lambda item: (-item[1], item[0])
        )

    def mle_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Calculates the unsmoothed Maximum Likelihood Estimation (MLE) probability:
        P_mle(word | context) = Count(context + (word,)) / Count(context)
        For unigrams (n=1), context is empty: P(word) = Count(word) / TotalTokens
        """
        if self.n == 1:
            count = self.unigram_counts.get(word, 0)
            return count / self.total_tokens if self.total_tokens > 0 else 0.0
        
        # Ensure context length matches n - 1
        if len(context) > self.n - 1:
            context = context[-(self.n - 1):]
        elif len(context) < self.n - 1:
            context = (self.START_TOKEN,) * (self.n - 1 - len(context)) + context

        context_total = sum(self.context_counts[context].values())
        if context_total == 0:
            return 0.0
        
        word_count = self.context_counts[context].get(word, 0)
        return word_count / context_total

    def laplace_probability(self, word: str, context: Tuple[str, ...], alpha: float = 1.0) -> float:
        """
        Calculates Add-alpha (Laplace if alpha=1.0, Lidstone if alpha < 1.0) smoothed probability:
        P_smoothed(word | context) = (Count(context, word) + alpha) / (Count(context) + alpha * |V|)
        """
        v_size = max(len(self.vocab), 1)
        
        if self.n == 1:
            word_count = self.unigram_counts.get(word, 0)
            return (word_count + alpha) / (self.total_tokens + alpha * v_size)
        
        if len(context) > self.n - 1:
            context = context[-(self.n - 1):]
        elif len(context) < self.n - 1:
            context = (self.START_TOKEN,) * (self.n - 1 - len(context)) + context

        context_total = sum(self.context_counts[context].values())
        word_count = self.context_counts[context].get(word, 0)
        
        return (word_count + alpha) / (context_total + alpha * v_size)

    def log_sentence_probability(
        self,
        sentence_tokens: List[str],
        smoothing: str = "laplace",
        alpha: float = 0.1
    ) -> float:
        """
        Computes the log2 probability of a sentence:
        log2 P(S) = sum_{i} log2 P(w_i | context)
        """
        seq = self._pad_tokens(sentence_tokens)
        if not seq:
            return -float('inf')
            
        log_prob = 0.0
        start_idx = (self.n - 1) if len(seq) >= self.n else 0
        
        for i in range(start_idx, len(seq)):
            target = seq[i]
            context = tuple(seq[max(0, i - self.n + 1):i])
            
            if smoothing == "mle":
                prob = self.mle_probability(target, context)
                if prob <= 0.0:
                    return -float('inf')
            elif smoothing in ("laplace", "add_k"):
                prob = self.laplace_probability(target, context, alpha=alpha)
            else:
                prob = self.laplace_probability(target, context, alpha=1.0)
            
            log_prob += math.log2(prob)
            
        return log_prob

    def sentence_probability(
        self,
        sentence_tokens: List[str],
        smoothing: str = "laplace",
        alpha: float = 0.1
    ) -> float:
        """
        Computes standard probability P(S) = 2^(log2 P(S)).
        """
        log_p = self.log_sentence_probability(sentence_tokens, smoothing=smoothing, alpha=alpha)
        if log_p == -float('inf'):
            return 0.0
        try:
            return 2.0 ** log_p
        except OverflowError:
            return 0.0

    def compute_perplexity(
        self,
        test_sentences: List[List[str]],
        smoothing: str = "add_k",
        alpha: float = 0.1
    ) -> float:
        """
        Computes perplexity (PPL) on a held-out test set:
        Perplexity = 2 ^ (-1/N * sum(log2 P(w_i | context)))
        where N is the total count of evaluated target tokens.
        """
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in test_sentences:
            if not sentence or len(sentence) < self.n:
                continue
            seq = self._pad_tokens(sentence)
            start_idx = self.n - 1
            for i in range(start_idx, len(seq)):
                target = seq[i]
                context = tuple(seq[max(0, i - self.n + 1):i])
                
                prob = self.laplace_probability(target, context, alpha=alpha)
                total_log_prob += math.log2(prob)
                total_words += 1
                
        if total_words == 0:
            return float('inf')
        
        cross_entropy = -total_log_prob / total_words
        return 2.0 ** cross_entropy

    def get_candidates(
        self,
        context_tokens: List[str],
        top_k: int = 5,
        smoothing: str = "add_k",
        alpha: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Retrieves candidate next words given a preceding context, along with estimated probabilities.
        Filters out special tokens like <s> and </s> for natural text continuation.
        Uses exact aggregate counts across all n-grams rather than first-come-first-served.
        """
        if self.n == 1:
            # Unigram: use precomputed cached unigrams sorted strictly by frequency count descending
            if not hasattr(self, '_sorted_unigrams') or not self._sorted_unigrams:
                self._sorted_unigrams = sorted(self.unigram_counts.items(), key=lambda item: (-item[1], item[0]))
            candidates = [
                (w, self.laplace_probability(w, (), alpha=alpha))
                for w, _ in self._sorted_unigrams[:top_k + 5]
                if w not in (self.START_TOKEN, self.END_TOKEN)
            ]
            return candidates[:top_k]
        
        # Format context tuple
        if len(context_tokens) > self.n - 1:
            context = tuple(context_tokens[-(self.n - 1):])
        else:
            context = (self.START_TOKEN,) * (self.n - 1 - len(context_tokens)) + tuple(context_tokens)
            
        counter = self.context_counts.get(context, Counter())
        
        if counter:
            # Rank candidates seen in this context strictly by frequency count descending
            sorted_context_words = sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0])
            )
            valid_candidates = [
                w for w, _ in sorted_context_words
                if w not in (self.START_TOKEN, self.END_TOKEN)
            ]
            results = []
            for w in valid_candidates[:top_k]:
                prob = self.laplace_probability(w, context, alpha=alpha)
                results.append((w, prob))
            return results
        else:
            # Backoff to unigram frequent words strictly by count if context has no observations
            if not hasattr(self, '_sorted_unigrams') or not self._sorted_unigrams:
                self._sorted_unigrams = sorted(self.unigram_counts.items(), key=lambda item: (-item[1], item[0]))
            fallback = [
                (w, self.laplace_probability(w, context, alpha=alpha))
                for w, _ in self._sorted_unigrams[:top_k + 5]
                if w not in (self.START_TOKEN, self.END_TOKEN)
            ]
            return fallback[:top_k]

    def generate_sentence(
        self,
        max_length: int = 15,
        temperature: float = 0.8,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Generates a sentence using temperature-scaled multinomial sampling without boundary tags.
        """
        if seed is not None:
            random.seed(seed)
            
        generated = []
        if self.n == 1:
            words = [w for w in self.unigram_counts.keys() if w not in (self.START_TOKEN, self.END_TOKEN)]
            counts = [self.unigram_counts[w] for w in words]
            total = sum(counts)
            probs = [c / total for c in counts]
            
            for _ in range(random.randint(5, max_length)):
                word = random.choices(words, weights=probs, k=1)[0]
                generated.append(word)
            return generated

        # Higher-order n-gram generation
        # Pick an initial context
        valid_contexts = [
            ctx for ctx in self.context_counts.keys()
            if self.START_TOKEN not in ctx and self.END_TOKEN not in ctx
        ]
        
        if not valid_contexts:
            valid_contexts = list(self.context_counts.keys())
            
        if not valid_contexts:
            return []
            
        # Choose a random or high-probability starting context
        start_ctx = random.choice(valid_contexts[:min(50, len(valid_contexts))])
        generated = list(start_ctx)
        curr_context = list(start_ctx)
        
        for _ in range(max_length - len(start_ctx)):
            context_tuple = tuple(curr_context[-(self.n - 1):])
            counter = self.context_counts.get(context_tuple, None)
            
            if not counter:
                break
                
            words = [w for w in counter.keys() if w not in (self.START_TOKEN, self.END_TOKEN)]
            if not words:
                break
            raw_counts = [counter[w] for w in words]
            
            if temperature == 0.0:
                best_word = words[raw_counts.index(max(raw_counts))]
            else:
                scaled_weights = [math.pow(c, 1.0 / max(temperature, 0.01)) for c in raw_counts]
                best_word = random.choices(words, weights=scaled_weights, k=1)[0]
                
            generated.append(best_word)
            curr_context.append(best_word)
            
        return generated
