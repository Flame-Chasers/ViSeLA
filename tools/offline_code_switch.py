#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline code-switching augmentation for captions (multilingual).

What it does:
    - Applies code-switching augmentation to captions in any supported language
    - Randomly translates a fraction of the words into the other languages (excluding the source)
    - Generates a configurable number of augmented variants per caption
    - Supports resuming from a progress file, a translation cache, and quality filtering

Usage:
    # English captions
    python tools/offline_code_switch.py \
        --lang en \
        --input data/CUHK/train_reid_en.json \
        --output data/CUHK/train_reid_en_cs.json \
        --device cuda:0

    # Chinese captions
    python tools/offline_code_switch.py \
        --lang ch \
        --input data/CUHK/train_reid_ch.json \
        --output data/CUHK/train_reid_ch_cs.json \
        --device cuda:0

Dependencies:
    pip install spacy transformers

    # spaCy models for every supported language
    python -m spacy download en_core_web_sm  # English
    python -m spacy download zh_core_web_sm  # Chinese
    python -m spacy download de_core_news_sm # German
    python -m spacy download fr_core_news_sm # French

    Only the model for the source language is required at run time, but installing
    all of them is recommended for multilingual processing.
"""

import argparse
import json
import os
import random
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from functools import lru_cache
from tqdm import tqdm

import spacy
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported languages (aligned with the project config)
SUPPORTED_LANGUAGES = ["en", "ch", "de", "fr"]

# Language code mapping (user input -> M2M100 code)
M2M_LANGUAGE_CODE_MAP = {
    "en": "en",
    "ch": "zh",  # Chinese uses "zh" in M2M100
    "zh": "zh",
    "de": "de",
    "fr": "fr",
}

# spaCy model per language
SPACY_MODELS: Dict[str, str] = {
    "en": "en_core_web_sm",
    "ch": "zh_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
}


def normalize_language_code(lang: str) -> str:
    """Map a user-facing language code to the ISO code used by M2M100."""
    return M2M_LANGUAGE_CODE_MAP.get(lang.lower(), lang.lower())


def build_target_langs(source_lang: str) -> List[str]:
    """
    Return the target languages, i.e. every supported language except the source
    (normalized to M2M100 codes).

    Args:
        source_lang: source language (en/ch/de/fr)

    Returns:
        Target language list, e.g. source_lang=en -> ["zh", "de", "fr"]
    """
    normalized_source = normalize_language_code(source_lang)

    # Normalize every candidate, drop the source language, deduplicate
    normalized_targets = sorted({
        normalize_language_code(candidate)
        for candidate in SUPPORTED_LANGUAGES
        if normalize_language_code(candidate) != normalized_source
    })

    return normalized_targets

# Domain vocabulary for person descriptions (clothing, colors, accessories, ...), per language.
# Currently English only; other languages can be added later.
DOMAIN_VOCAB: Dict[str, Set[str]] = {
    "en": {
        # Clothing
        "jacket", "coat", "hoodie", "shirt", "t-shirt", "sweater", "vest", "blazer",
        "dress", "skirt", "pants", "jeans", "shorts", "trousers", "leggings",
        # Footwear
        "shoes", "sneakers", "boots", "sandals", "heels", "slippers",
        # Accessories
        "backpack", "bag", "handbag", "purse", "hat", "cap", "scarf", "glasses",
        "sunglasses", "watch", "belt", "gloves", "umbrella",
        # Colors
        "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
        "brown", "pink", "purple", "orange", "beige", "navy", "khaki",
        # Patterns / materials
        "striped", "plaid", "checkered", "floral", "solid", "patterned",
        "denim", "leather", "cotton", "wool",
        # Adjectives
        "long", "short", "tight", "loose", "dark", "light", "bright",
        # Body parts / attributes
        "hair", "beard", "mustache", "bald", "tall", "slim", "heavy",
    },
    # Domain vocabularies for other languages can be added here
    # "ch": {...},
    # "de": {...},
    # "fr": {...},
}

# Custom stopwords that complement the spaCy built-in lists:
# function words, particles, articles, pronouns, demonstratives, ...
CUSTOM_STOPWORDS: Dict[str, Set[str]] = {
    "ch": {
        # Function words
        "的", "了", "是", "在", "和", "与", "及", "而", "将", "被", "把", "给", "从", "向", "到",
        # Pronouns
        "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
        "自己", "大家", "别人", "谁",
        # Demonstratives
        "这", "那", "哪", "这个", "那个", "哪个", "这些", "那些", "哪些",
        "这里", "那里", "哪里", "这儿", "那儿", "哪儿",
        # Measure words
        "个", "位", "条", "只", "件", "些",
    },
    "de": {"der", "die", "das", "ein", "eine", "dem", "den", "des", "und", "oder", "zu", "im", "am", "vom"},
    "fr": {"le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "au", "aux", "dans", "sur"},
}

# POS whitelist (Universal POS tags).
# The Chinese model has no NER, so PROPN is excluded for Chinese.
ALLOWED_POS_TAGS: Dict[str, Set[str]] = {
    "en": {"NOUN", "ADJ", "VERB", "PROPN"},  # English: PROPN allowed (NER available)
    "ch": {"NOUN", "ADJ", "VERB"},            # Chinese: PROPN excluded (no NER)
    "de": {"NOUN", "ADJ", "VERB", "PROPN"},  # German: PROPN allowed
    "fr": {"NOUN", "ADJ", "VERB", "PROPN"},  # French: PROPN allowed
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Offline code-switching augmentation for captions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input / output
    parser.add_argument("--input", required=True, help="input JSON file")
    parser.add_argument("--output", required=True, help="output JSON file")
    parser.add_argument("--progress", default=None, help="progress file for resuming (JSONL)")
    parser.add_argument("--cache", default=None, help="translation cache file (JSON)")

    # Model
    parser.add_argument("--model", default="facebook/m2m100_418M", help="translation model")
    parser.add_argument("--device", default="cuda:0", help="compute device")

    # Augmentation
    parser.add_argument("--num-variants", type=int, default=2, help="augmented variants per caption")
    parser.add_argument("--min-ratio", type=float, default=0.20, help="minimum replacement ratio")
    parser.add_argument("--max-ratio", type=float, default=0.30, help="maximum replacement ratio")
    parser.add_argument("--max-consecutive", type=int, default=2, help="maximum number of consecutive replaced words")

    # Quality filtering
    parser.add_argument("--max-len-ratio", type=float, default=1.6, help="maximum translation/original length ratio")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="translation batch size")

    # Source language
    parser.add_argument(
        "--lang",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="source language of the input captions (excluded from the target languages)"
    )

    return parser.parse_args()


class TranslationCache:
    """Translation cache with an in-memory dict and optional on-disk persistence."""

    def __init__(self, cache_path: Optional[str] = None, max_size: int = 50000):
        self.cache_path = cache_path
        self.max_size = max_size
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        """Load the cache from disk."""
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached translations")
            except Exception as e:
                logger.warning(f"Failed to load translation cache: {e}")

    def save_cache(self):
        """Save the cache to disk."""
        if self.cache_path:
            try:
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(self.cache)} cached translations")
            except Exception as e:
                logger.warning(f"Failed to save translation cache: {e}")

    def get(self, key: str) -> Optional[str]:
        """Return the cached translation, if any."""
        return self.cache.get(key)

    def set(self, key: str, value: str):
        """Store a translation."""
        if len(self.cache) >= self.max_size:
            # Simple eviction: drop the oldest 10% of entries
            keys_to_remove = list(self.cache.keys())[:self.max_size // 10]
            for k in keys_to_remove:
                del self.cache[k]
        self.cache[key] = value

    def make_key(self, word: str, source_lang: str, target_lang: str) -> str:
        """Build a cache key that includes the source and target languages."""
        return f"{word}||{source_lang}||{target_lang}"


class CodeSwitchAugmentor:
    """Multilingual code-switching augmentor."""

    def __init__(
        self,
        model_name: str,
        device: str,
        cache: TranslationCache,
        source_lang: str = "en",  # source language
        min_ratio: float = 0.15,
        max_ratio: float = 0.30,
        max_consecutive: int = 2,
        max_len_ratio: float = 1.6,
    ):
        self.device = device
        self.cache = cache

        # Normalize the source language code ("zh" is stored as "ch" to match SPACY_MODELS)
        if source_lang.lower() == "zh":
            self.source_lang = "ch"
        else:
            self.source_lang = source_lang.lower()

        # Validate the source language
        if self.source_lang not in SPACY_MODELS:
            raise ValueError(
                f"Unsupported source language: {source_lang}. "
                f"Supported languages: {', '.join(SPACY_MODELS.keys())}"
            )

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_consecutive = max_consecutive
        self.max_len_ratio = max_len_ratio

        # spaCy handles tokenization and POS filtering for every language
        spacy_model = SPACY_MODELS[self.source_lang]

        try:
            logger.info(f"Loading spaCy model: {spacy_model}")
            self.nlp = spacy.load(spacy_model, disable=["parser", "textcat"])
            logger.info(f"spaCy model loaded: {spacy_model}")
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{spacy_model}' not found.\n"
                f"Install it with:\n"
                f"  python -m spacy download {spacy_model}\n\n"
                f"Install commands for all languages:\n"
                f"  python -m spacy download en_core_web_sm\n"
                f"  python -m spacy download zh_core_web_sm\n"
                f"  python -m spacy download de_core_news_sm\n"
                f"  python -m spacy download fr_core_news_sm"
            ) from e

        # Translation model
        logger.info(f"Loading translation model: {model_name}")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info("Translation model loaded")

    def _get_dynamic_ratio(self, token_count: int) -> float:
        """Replacement ratio as a function of the token count."""
        if token_count <= 8:
            return self.min_ratio
        elif token_count <= 15:
            # Linear interpolation
            return self.min_ratio + (self.max_ratio - self.min_ratio) * (token_count - 8) / 7
        else:
            return self.max_ratio

    def _should_replace(self, token) -> bool:
        """Decide whether a token is eligible for replacement."""
        # Stopwords (built-in + custom)
        custom_stopwords = CUSTOM_STOPWORDS.get(self.source_lang, set())
        if token.is_stop or token.lower_ in custom_stopwords:
            return False

        # Punctuation and numbers
        if token.is_punct or token.like_num:
            return False

        # Short tokens (minimum length 1 for Chinese, 3 otherwise)
        min_len = 1 if self.source_lang in ("ch", "zh") else 3
        if len(token.text) < min_len:
            return False

        # Named entities (when the model supports NER)
        if token.ent_type_:
            return False

        # Language-specific POS whitelist
        allowed_pos = ALLOWED_POS_TAGS.get(self.source_lang, {"NOUN", "ADJ", "VERB"})
        if token.pos_ not in allowed_pos:
            return False

        return True

    def _select_candidates(self, doc, ratio: float) -> List[Tuple[int, str]]:
        """
        Select the words to replace.
        Returns: [(token_index, word), ...]
        """
        # Domain vocabulary for this language (falls back to English)
        domain_vocab = DOMAIN_VOCAB.get(self.source_lang, DOMAIN_VOCAB.get("en", set()))

        # Collect replaceable tokens
        candidates = []
        for token in doc:
            if self._should_replace(token):
                # Up-weight domain vocabulary
                weight = 2.0 if token.lemma_.lower() in domain_vocab else 1.0
                candidates.append((token.i, token.text, weight))

        if not candidates:
            return []

        # Number of words to replace
        num_replace = max(1, int(len(doc) * ratio))

        # Weighted random ordering
        random.shuffle(candidates)
        candidates.sort(key=lambda x: -x[2] * random.random())

        # Pick candidates while limiting consecutive replacements
        selected = []
        consecutive = 0
        last_idx = -10

        for idx, word, _ in candidates:
            if len(selected) >= num_replace:
                break

            # Consecutive-run check
            if idx == last_idx + 1:
                consecutive += 1
                if consecutive > self.max_consecutive:
                    continue
            else:
                consecutive = 1

            selected.append((idx, word))
            last_idx = idx

        return selected

    def _translate_word(self, word: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a single word.

        Args:
            word: word to translate
            source_lang: source language
            target_lang: target language

        Returns:
            Translated word
        """
        normalized_source = normalize_language_code(source_lang)
        normalized_target = normalize_language_code(target_lang)

        # Cache lookup
        cache_key = self.cache.make_key(word, normalized_source, normalized_target)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            self.tokenizer.src_lang = normalized_source
            inputs = self.tokenizer(word, return_tensors="pt").to(self.device)
            forced_bos_id = self.tokenizer.get_lang_id(normalized_target)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_id,
                    max_new_tokens=10,
                    num_beams=1,  # greedy decoding for speed
                )

            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Cache the result
            self.cache.set(cache_key, translation)
            return translation

        except Exception as e:
            logger.debug(f"Translation failed for '{word}' ({source_lang} -> {target_lang}): {e}")
            return word

    def _filter_translation(self, original: str, translation: str) -> bool:
        """
        Quality filter. Returns True when the translation is acceptable.
        """
        # Empty translation
        if not translation:
            return False

        # Identical to the original
        if translation.lower() == original.lower():
            return False

        # Length filter (CJK originals get a looser ratio)
        has_cjk = any('\u4e00' <= char <= '\u9fff' for char in original)
        if has_cjk:
            # Chinese words are short (1-2 characters) and translations tend to be longer
            max_ratio = 4.0
        else:
            max_ratio = self.max_len_ratio

        if len(translation) > len(original) * max_ratio:
            return False

        # Reject garbled output: allow Latin letters, CJK, Cyrillic, whitespace, hyphen, apostrophe
        valid_pattern = r'^[A-Za-z\u4e00-\u9fff\u00C0-\u024F\u0400-\u052F\s\-\']+$'
        if not re.match(valid_pattern, translation):
            return False

        return True

    def augment(self, text: str, target_langs: List[str]) -> str:
        """
        Apply code-switching augmentation with a mix of target languages.

        spaCy provides tokenization, POS filtering and NER so that only suitable
        content words are replaced and the sentence structure is preserved.

        Note: the source language is fixed at construction time (self.source_lang).

        Args:
            text: original text
            target_langs: target languages (M2M100 codes, e.g. ["zh", "de", "fr"])

        Returns:
            Augmented text
        """
        if not self.nlp:
            raise RuntimeError("spaCy model is not initialized; cannot process text")

        # Normalize target languages, drop the source language, deduplicate
        normalized_source = normalize_language_code(self.source_lang)
        normalized_targets = sorted({
            normalize_language_code(lang)
            for lang in target_langs
            if normalize_language_code(lang) != normalized_source
        })

        if not normalized_targets:
            logger.warning(f"No valid target languages (source language: {self.source_lang})")
            return text

        # Tokenize and POS-tag with spaCy
        doc = self.nlp(text)
        ratio = self._get_dynamic_ratio(len(doc))
        candidates = self._select_candidates(doc, ratio)

        if not candidates:
            return text

        # Translate each candidate into a randomly chosen target language
        translations: Dict[int, str] = {}
        for idx, word in candidates:
            target_lang = random.choice(normalized_targets)
            trans = self._translate_word(word, self.source_lang, target_lang)
            if self._filter_translation(word, trans):
                translations[idx] = trans

        # Rebuild the text (text_with_ws keeps the original whitespace)
        result_parts: List[str] = []
        for token in doc:
            if token.i in translations:
                # Replace with the translation, keeping the trailing whitespace
                result_parts.append(translations[token.i])
                result_parts.append(token.whitespace_)
            else:
                # Keep the original token (with trailing whitespace)
                result_parts.append(token.text_with_ws)

        return "".join(result_parts).strip()


def load_progress(progress_path: Optional[str]) -> Set[str]:
    """Load the set of already processed ids."""
    done = set()
    if progress_path and os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        done.add(str(rec.get("id", "")))
                    except:
                        continue
            logger.info(f"Loaded {len(done)} processed ids from the progress file")
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}")
    return done


def append_progress(progress_path: Optional[str], item_id: Any):
    """Append one processed id to the progress file."""
    if not progress_path:
        return
    try:
        with open(progress_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"id": item_id}) + '\n')
    except Exception as e:
        logger.warning(f"Failed to write progress: {e}")


def main():
    args = parse_args()

    # Seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Target languages (everything except the source)
    source_lang = args.lang
    target_langs = build_target_langs(source_lang)

    logger.info(f"Source language: {source_lang}")
    logger.info(f"Target languages: {target_langs}")

    # Input data
    logger.info(f"Loading input: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"{len(data)} items")

    # Progress
    done_ids = load_progress(args.progress)

    # Existing output (merged when resuming)
    existing_results = {}
    if args.output and os.path.exists(args.output) and done_ids:
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            for item in old_data:
                existing_results[str(item.get("id", ""))] = item
            logger.info(f"Loaded {len(existing_results)} previously processed results")
        except Exception as e:
            logger.warning(f"Failed to load existing output: {e}")

    # Cache
    cache = TranslationCache(args.cache)

    # Augmentor
    augmentor = CodeSwitchAugmentor(
        model_name=args.model,
        device=args.device,
        cache=cache,
        source_lang=source_lang,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        max_consecutive=args.max_consecutive,
        max_len_ratio=args.max_len_ratio,
    )

    # Process
    results = []
    skipped = 0

    for item in tqdm(data, desc="Processing"):
        item_id = str(item.get("id", ""))

        # Reuse the stored result for already processed items
        if item_id in done_ids:
            if item_id in existing_results:
                results.append(existing_results[item_id])
            skipped += 1
            continue

        # Copy the original item
        result_item = item.copy()

        # Augment every caption
        captions = item.get("captions", [])
        captions_cs = []

        for caption in captions:
            variants = []
            for _ in range(args.num_variants):
                # The source language was fixed when the augmentor was created
                augmented = augmentor.augment(caption, target_langs)
                variants.append(augmented)
            captions_cs.append(variants)

        result_item["captions_cs"] = captions_cs
        results.append(result_item)

        # Record progress
        append_progress(args.progress, item_id)

    # Save results
    logger.info(f"Saving results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save the translation cache
    cache.save_cache()

    # Summary
    logger.info(f"Done: {len(results) - skipped} new, {skipped} reused")
    logger.info(f"Translation cache: {len(cache.cache)} entries")


if __name__ == "__main__":
    main()
