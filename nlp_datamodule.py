import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List, Union
from collections import Counter
from tqdm import tqdm
from utils import (
    normalize_repeating_chars,
    normalize_repeating_words,
    normalize_hyphenated_words,
    normalize_quotation_marks,
    normalize_unicode,
    normalize_whitespace,
    replace_currency_symbols,
    replace_emojis,
    replace_numbers,
    replace_emails,
    replace_hashtags,
    replace_phone_numbers,
    replace_urls,
    replace_user_handles,
    # remove_accents,
    remove_punctuation,
    remove_stopwords,
    normalize_tolowercase,
)


class NLPDataModule(pl.LightningDataModule):
    """
    In each text classification pipeline there are the following steps
    1. Data preprocessing:
        - normalization
        - substitution
        - spellchecking
        - tokenization
        - cleaning
        - word_collocation
    2. Numericalization:
        - compute word frequencies
        - create vocabulary
    4. Create dataloaders
    """

    def __init__(self, vocab=None, *args, **kwargs):
        self.vocab = vocab
        pass

    def normalization(self, text: Union[List[str], str]) -> Union[List[str], str]:
        """
        Raw text often contains all kinds of oddities such as emoji's,
        numbers, currency symbols, inconsistent punctuation, and could even be
        in a text encoding not suited for operating on. Normalization of text
        aims to remove/transform these oddities as a first step to reduce noise
        in the data.

        1. Replace emojis with "xx_emoji"
        1. Replace currency symbols (i.e. \\$) with "xx_currency"
        1. Replace numbers with ""xx_number"
        1. Adjusts unicode encoding to NFC
        1. Concatenates words split across a line by a hyphen
            (i.e. hyphen-ated to hyphenated)
        1. Adjusts all quotation marks to use the same symbols
            (i.e. \\` \\' and \\" all get changed to \\")
        1. Removes extra whitespace and line breaks
        """
        text = replace_currency_symbols(text)
        text = replace_emojis(text)
        text = replace_numbers(text)
        text = replace_emails(text)
        text = replace_hashtags(text)
        text = replace_phone_numbers(text)
        text = replace_urls(text)
        text = replace_user_handles(text)
        text = normalize_hyphenated_words(text)
        text = normalize_unicode(text)
        text = normalize_repeating_chars(text)
        text = normalize_repeating_words(text)
        text = normalize_whitespace(text)
        text = normalize_tolowercase(text)
        return text

    def substitution(self, text: Union[List[str], str]) -> Union[List[str], str]:
        """
        Sometimes standard preprocessing steps are not enough to prepare the text
        and one needs to apply a custom substitution rule based on regular expressions.
        For example, imagine a survey related to a postal company, where a concept of
        “track and trace” is used a lot. With the default tokenization settings further
        in the workflow, this expression would be split into two different sentences
        (on “and”). In order to avoid this, we want to replace “track and trace” with
        “track_and_trace”.
        """
        pass

    def spellchecking(self, text: Union[List[str], str]) -> Union[List[str], str]:
        """
        As we can imagine, miss-spelled words are ubiquitous in textual data.
        Ultimately, words with typos/incorrect spelling would be recognized
        as different words. Therefore, we want to make sure we transform
        misspelled words into the correct spelling.
        The basic principles behind the spell-checker:
        1. Scan sentences to identify individual words
        1. Compare these words to a large dictionary of correctly spelled words
        1. Calculate a 'distance' between the word spelling and the dictionary of
            correctly spelled words
        1. The dictionary word with closest distance to the verbatim word is the
            suggestion for correction
        """
        pass

    def tokenization(self, text: str) -> List[str]:
        """
        Tokenization is the process of tokenizing or splitting a string of text into a
        list of tokens. One can think of a token as a part of the original verbatim - a
        sentence is a token/part of a paragraph, a sentence part is a token/part of a
        sentence and a word is a token/part of a sentence part.
        In practice therefore, the tokenizer essentially:
        1. Breaks raw text into paragraphs
        1. Breaks paragraphs into sentences
        1. Breaks sentences into sentence parts.
        1. Breaks sentence parts into individual words
        """
        return text.split()

    def cleaning(self, text: Union[List[str], str]) -> Union[List[str], str]:
        """
        At this point, we still have many punctuations, stop words (e.g. "the", "a",
        "to be", "and", "but", etc.) and words expressed as a variation of the stem
        word. In this step one can stem orlemmatize words (i.e. word-stemming), remove
        punctuation and stop-words.
        """
        # NOTE: the order is important here
        text = remove_stopwords(text)
        text = normalize_quotation_marks(text)
        text = remove_punctuation(text)
        return text

    def word_collocation(self, text: Union[List[str], str]) -> Union[List[str], str]:
        """
        Certain concepts or entities consist of multiple words. Sometimes words can
        have different meanings depending on the phrase they are used in (e.g. "customer
        service" vs. "network service"). In addition, words could also be taken out of
        context if we do not consider collocations (i.e. "new" in "New York" or "black"
        in "Black Friday"). By identifying collocations (i.e. words that often appear
        next to each other), we can treat these word combinations as single tokens in
        order to account for their true meaning.
        """
        pass

    def numericalization(self, text: List[str], max_len: int, pad: int) -> List[int]:
        """
        This step can be done as a pre-processing step or be part of an end-to-end
        training procedure. In general, we need to translate words into numbers in
        order to operate on them. There are various methods to represent words
        mathematically: n_grams, tf-idf, word-embeddings (shallow or deep)
        """
        if self.vocab is None:
            raise ValueError("You need to provide a vocabulary to numericalize.")
        text = [self.word2index[word] if word in self.word2index else 0 for word in text[:max_len]]
        if max_len - len(text) > 0:
            padding = [pad] * (max_len - len(text))
            text.extend(padding)
        return text
        
    # NOTE: this might be better, but the important point is that build_vocab
    # should be a first-class citizen in NLPDataModule
    def build_vocab(self, text: List[List[str]]) -> None:
        self.vocab = Counter()
        for i in tqdm(text, desc="Building vocab"):
            self.vocab.update(i)
        self.word2index = {k: i+2 for i, k in enumerate(self.vocab.keys())}
        self.word2index.update({"<pad>": 0, "<unk>": 1})
        self.idx2word = {i: k for k, i in self.word2index.items()}
