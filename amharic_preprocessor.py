"""
Amharic Text Preprocessing Module
==================================
This module provides complete preprocessing, normalization, sentence segmentation,
tokenization, and stopword filtering utilities specifically tailored for Amharic NLP
and the GPAC (General-Purpose Amharic Corpus).

Key features:
1. Character Normalization: Unifies Amharic homophones (letters with identical phonemes
   such as ha/hha/xa variants, se/sze, a/ayin, tse/tse) to mitigate vocabulary sparsity.
2. Sentence Segmentation: Splits text into complete meaningful sentences using the
   regular expression specified by course instructor instructions:
   `SENT_SPLIT = re.compile(r'(?:፡፡)+|።+|[!?]+')`
3. Punctuation and Noise Cleaning: Cleans ASCII and Ethiopic punctuation marks, URLs,
   and non-Amharic noise while preserving valid Ethiopic word characters.
4. Stopword Removal: Curated list of high-frequency Amharic functional words, copulas,
   prepositions, conjunctions, and auxiliary verbs.
"""

import re
from typing import List, Generator, Set

# ---------------------------------------------------------------------------
# 1. Instructor-Specified Sentence Splitter
# ---------------------------------------------------------------------------
# Captures Ethiopic double colon (፡፡), Ethiopic full stop (።), and exclamation/question marks
SENT_SPLIT = re.compile(r'(?:፡፡)+|።+|[!?]+')

# ---------------------------------------------------------------------------
# 2. Amharic Homophone Character Normalization Mapping
# ---------------------------------------------------------------------------
# In Amharic orthography, several graphemes represent the exact same spoken phoneme.
# Standardizing these homophones to a single canonical letter reduces artificial
# vocabulary inflation and improves statistical language model generalizability.
AMHARIC_CHAR_MAP = {
    # 'h' phoneme variants (ሐ, ኀ, ኃ, ኻ -> canonical ሀ)
    'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
    'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
    'ኻ': 'ሃ', 'ኹ': 'ሁ', 'ኺ': 'ሂ', 'ኼ': 'ሄ', 'ኽ': 'ህ', 'ኾ': 'ሆ',
    
    # 's' phoneme variants (ሠ -> canonical ሰ)
    'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
    
    # 'a' vowel / glottal stop variants (ዐ -> canonical አ, and related orders)
    'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'ኣ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
    
    # 'ts' phoneme variants (ፀ -> canonical ጸ)
    'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ፂ': 'ጺ', 'ፃ': 'ጻ', 'ፄ': 'ጼ', 'ፅ': 'ጽ', 'ፆ': 'ጾ',
}

# ---------------------------------------------------------------------------
# 3. Comprehensive Curated Amharic Stopwords List
# ---------------------------------------------------------------------------
# Stopwords are high-frequency grammatical functional words (prepositions, conjunctions,
# pronouns, copulas, auxiliary verbs) that carry little domain-specific semantic content.
AMHARIC_STOPWORDS: Set[str] = {
    # Copulas and auxiliary verbs
    'ነው', 'ነበር', 'ናት', 'ናቸው', 'ነበረች', 'ነበሩ', 'ሆነ', 'ሆኖ', 'ሆና', 'ሆኑ',
    'መሆን', 'እንደሆነ', 'ቢሆንም', 'አለ', 'አሉ', 'አለች', 'የለም', 'አይደለም', 'አይደሉም',
    'ነበረበት', 'ይሆናል', 'ይችላል', 'ይገኛል', 'ይገኛሉ',
    
    # Conjunctions and discourse connectives
    'እና', 'ወይም', 'ግን', 'ደግሞ', 'እንጂ', 'እንዲሁም', 'ነገር', 'ነገር ግን',
    'ስለሆነም', 'በመሆኑም', 'ስለዚህ', 'ቢሆንም', 'እንኳን', 'እንኳ', 'እንደዚሁ',
    
    # Prepositions and locatives
    'ውስጥ', 'ላይ', 'ታች', 'በታች', 'በላይ', 'መካከል', 'በኩል', 'ወደ', 'ጋር', 'ስለ',
    'እስከ', 'ዘንድ', 'በስተቀር', 'በኋላ', 'በፊት', 'አጠገብ', 'ዙሪያ',
    
    # Pronouns and demonstratives
    'ይህ', 'ይህን', 'ይህም', 'ይህች', 'ይህንኑ', 'ይኸው', 'እነዚህ', 'እነዚህን',
    'ያ', 'ያን', 'ያንን', 'ያንኑ', 'እነዚያ', 'እነዚያን', 'እኔ', 'እኛ', 'አንተ', 'አንቺ',
    'እናንተ', 'እሱ', 'እሷ', 'እነሱ', 'እነርሱ', 'ራሱ', 'ራሷ', 'ራሳቸው', 'ራሳችን',
    
    # Quantifiers, modifiers, and adverbs
    'በጣም', 'ብቻ', 'ሁሉ', 'ሁሉንም', 'ሁሉን', 'ብዙ', 'ጥቂት', 'አንዳንድ', 'አንድ',
    'ሁለት', 'ሶስት', 'አራት', 'አምስት', 'ሌላ', 'ሌሎች', 'ሙሉ', 'ያህል',
    
    # Reporting and functional speech tokens
    'ማለት', 'ይላል', 'ብሎ', 'ብላ', 'ብለው', 'በማለት', 'ሲል', 'ስትል', 'ሲሉ',
    'እንደ', 'እንዴት', 'ለምን', 'የት', 'መቼ', 'ማን', 'ምን', 'ጊዜ', 'ቀን', 'አመት',
    'ዓመት', 'ወዘተ'
}

# ---------------------------------------------------------------------------
# 4. Amharic Abbreviation Patterns and Protection
# ---------------------------------------------------------------------------
# Preserves multi-word / punctuated abbreviations as single atomic tokens
# (e.g., 'ዶ ር' or 'ዶ/ር' -> 'ዶ/ር', and 'ዓ ም' or 'ዓ.ም' or 'ኣ ም' -> 'ኣ.ም')
ABBREVIATION_PATTERNS = [
    (re.compile(r'እ\s*[\/\.]\s*ኤ\s*[\/\.]\s*አ|እ\s+ኤ\s+አ'), 'እ.ኤ.አ'),
    (re.compile(r'ዶ\s*[\/\.]\s*ር|ዶ\s+ር'), 'ዶ/ር'),
    (re.compile(r'ፕ\s*[\/\.]\s*ር|ፕ\s+ር'), 'ፕ/ር'),
    (re.compile(r'ኢ\s*[\/\.]\s*ር|ኢ\s+ር'), 'ኢ/ር'),
    (re.compile(r'ወ\s*[\/\.]\s*ሮ|ወ\s+ሮ'), 'ወ/ሮ'),
    (re.compile(r'ወ\s*[\/\.]\s*ሪት|ወ\s+ሪት'), 'ወ/ሪት'),
    (re.compile(r'ት\s*[\/\.]\s*ቤት|ት\s+ቤት'), 'ት/ቤት'),
    (re.compile(r'መ\s*[\/\.]\s*ቤት|መ\s+ቤት'), 'መ/ቤት'),
    (re.compile(r'ክ\s*[\/\.]\s*ከተማ|ክ\s+ከተማ'), 'ክ/ከተማ'),
    (re.compile(r'ጠ\s*[\/\.]\s*ሚ|ጠ\s+ሚ'), 'ጠ/ሚ'),
    (re.compile(r'[ዓኣ]\s*[\/\.]\s*ም|[ዓኣ]\s+ም'), 'ኣ.ም'),
    (re.compile(r'[ዓኣ]\s*[\/\.]\s*ዓ|[ዓኣ]\s+ዓ'), 'ኣ.ዓ'),
]

SAFE_ABBREV_MAP = {
    'እ.ኤ.አ': 'እኤአየአህጽሮትቃል',
    'ዶ/ር': 'ዶርየአህጽሮትቃል',
    'ፕ/ር': 'ፕርየአህጽሮትቃል',
    'ኢ/ር': 'ኢርየአህጽሮትቃል',
    'ወ/ሮ': 'ወሮየአህጽሮትቃል',
    'ወ/ሪት': 'ወሪትየአህጽሮትቃል',
    'ት/ቤት': 'ትቤትየአህጽሮትቃል',
    'መ/ቤት': 'መቤትየአህጽሮትቃል',
    'ክ/ከተማ': 'ክከተማየአህጽሮትቃል',
    'ጠ/ሚ': 'ጠሚየአህጽሮትቃል',
    'ኣ.ም': 'ኣምየአህጽሮትቃል',
    'ኣ.ዓ': 'ኣዓየአህጽሮትቃል',
}
REVERSE_ABBREV_MAP = {v: k for k, v in SAFE_ABBREV_MAP.items()}

# ---------------------------------------------------------------------------
# 5. Preprocessing Functions
# ---------------------------------------------------------------------------
def normalize_amharic(text: str) -> str:
    """
    Applies Amharic character normalization:
    - Standardizes abbreviations (e.g. 'ዶ ር' -> 'ዶ/ር', 'ዓ ም' -> 'ኣ.ም').
    - Normalizes homophone graphemes (ha, se, a, tse orders) to canonical forms.
    - Replaces the Ethiopic word separator (፡) with a standard space.
    - Removes punctuation, symbols, and non-Ethiopic noise while preserving
      abbreviations as single atomic tokens.
    - Preserves Ethiopic alphabetic characters (\u1200 - \u137F).
    """
    if not text:
        return ""
    
    # 1. Unify and standardize abbreviations first before stripping punctuation or spaces
    for pattern, replacement in ABBREVIATION_PATTERNS:
        text = pattern.sub(replacement, text)

    # 2. Replace homophones
    for char, canonical in AMHARIC_CHAR_MAP.items():
        text = text.replace(char, canonical)
    
    # 3. Replace Ethiopic word separator with space
    text = text.replace('፡', ' ')
    
    # 4. Protect abbreviations with safe Ethiopic tokens before punctuation removal
    for abbrev, placeholder in SAFE_ABBREV_MAP.items():
        text = text.replace(abbrev, placeholder)

    # 5. Remove punctuation and special symbols
    text = re.sub(r'[^\w\s\u1200-\u137F]', ' ', text)
    
    # 6. Restore protected abbreviations
    for placeholder, abbrev in REVERSE_ABBREV_MAP.items():
        text = text.replace(placeholder, abbrev)

    # 7. Collapse multiple whitespace characters into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_amharic_sentence(sentence: str) -> List[str]:
    """
    Tokenizes a single normalized Amharic sentence into a list of word tokens.
    Keeps valid tokens that contain Ethiopic script characters.
    """
    normalized = normalize_amharic(sentence)
    tokens = []
    for token in normalized.split():
        # Check if token contains at least one Ethiopic character
        if any('\u1200' <= char <= '\u137F' for char in token):
            tokens.append(token)
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filters out common Amharic stopwords from a list of tokens.
    """
    return [token for token in tokens if token not in AMHARIC_STOPWORDS]


def stream_sentences_from_corpus(
    file_path: str,
    max_sentences: int = None,
    min_tokens: int = 3,
    chunk_size: int = 4 * 1024 * 1024
) -> Generator[str, None, None]:
    """
    Streams meaningful sentences from GPAC.txt using the instructor's regex
    SENT_SPLIT = re.compile(r'(?:፡፡)+|።+|[!?]+') in a memory-efficient chunked manner.

    Parameters:
    -----------
    file_path : str
        Path to GPAC.txt file.
    max_sentences : int, optional
        Maximum number of valid sentences to yield. If None, yields all sentences.
    min_tokens : int
        Minimum number of space-delimited words for a sentence to be considered meaningful.
    chunk_size : int
        Buffer chunk size for streaming (default 4MB).
    """
    count = 0
    buffer = ""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                # Process remaining buffer
                if buffer.strip():
                    parts = SENT_SPLIT.split(buffer)
                    for s in parts:
                        s_clean = s.strip()
                        if len(s_clean.split()) >= min_tokens:
                            yield s_clean
                            count += 1
                            if max_sentences and count >= max_sentences:
                                return
                break

            buffer += chunk
            parts = SENT_SPLIT.split(buffer)
            # The last segment might be incomplete, save it for the next chunk
            buffer = parts[-1]
            for s in parts[:-1]:
                s_clean = s.strip()
                if len(s_clean.split()) >= min_tokens:
                    yield s_clean
                    count += 1
                    if max_sentences and count >= max_sentences:
                        return
