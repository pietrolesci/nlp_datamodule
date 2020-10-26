from multipledispatch import dispatch

import re

from textacy.preprocessing.normalize import (
    normalize_hyphenated_words as _normalize_hyphenated_words,
    normalize_quotation_marks as _normalize_quotation_marks,
    normalize_unicode as _normalize_unicode,
    normalize_whitespace as _normalize_whitespace,
)
from textacy.preprocessing.replace import (
    replace_currency_symbols as _replace_currency_symbols,
    replace_emojis as _replace_emojis,
    replace_numbers as _replace_numbers,
    replace_emails as _replace_emails,
    replace_hashtags as _replace_hashtags,
    replace_phone_numbers as _replace_phone_numbers,
    replace_urls as _replace_urls,
    replace_user_handles as _replace_user_handles,
)
from textacy.preprocessing.remove import (
    remove_accents as _remove_accents,
    remove_punctuation as _remove_punctuation,
)
from nltk.corpus import stopwords
import stop_words

from typing import List


# fastai preprocessing tools
_re_rep = re.compile(r"(\S)(\1{2,})")
_re_wrep = re.compile(r"(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)")


@dispatch(str)
def normalize_repeating_chars(t: str) -> str:
    "Replace repetitions at the character level: cccc -> c"

    def _replace_rep(m):
        c, cc = m.groups()
        return f" {c} "

    return _re_rep.sub(_replace_rep, t)


@dispatch(list)
def normalize_repeating_chars(t: List[str]) -> List[str]:
    "Replace repetitions at the character level: cccc -> c"

    def _replace_rep(m):
        c, cc = m.groups()
        return f" {c} "

    return [_re_rep.sub(_replace_rep, i) for i in t]


@dispatch(str)
def normalize_repeating_words(t: str) -> str:
    "Replace word repetitions: word word word word -> word"

    def _replace_wrep(m):
        c, cc, e = m.groups()
        return f" {c} "

    return _re_wrep.sub(_replace_wrep, t)


@dispatch(list)
def normalize_repeating_words(t: List[str]) -> List[str]:
    "Replace word repetitions: word word word word -> word"

    def _replace_wrep(m):
        c, cc, e = m.groups()
        return f" {c} "

    return [_re_wrep.sub(_replace_wrep, i) for i in t]


# textacy preprocessing tools
EMOJI_CODE = " xxemoji "
CURRENCY_CODE = " xxcurrency "
NUMBER_CODE = " xxnumber "
EMAIL_CODE = " xxemail "
PHONENUMBER_CODE = " xxphonenumber "
HASHTAG_CODE = " xxhashtag "
URL_CODE = " xxurl "
USERHANDLER_CODE = " xxuserhandler "
UNICODE_FORM = " NFKC "


@dispatch(str)
def normalize_hyphenated_words(t: str) -> str:
    return _normalize_hyphenated_words(t)


@dispatch(list)
def normalize_hyphenated_words(t: List[str]) -> List[str]:
    return [_normalize_hyphenated_words(i) for i in t]


@dispatch(str)
def normalize_quotation_marks(t: str) -> str:
    return _normalize_quotation_marks(t)


@dispatch(list)
def normalize_quotation_marks(t: List[str]) -> List[str]:
    return [_normalize_quotation_marks(i) for i in t]


@dispatch(str)
def normalize_unicode(t: str) -> str:
    return _normalize_unicode(t, UNICODE_FORM)


@dispatch(list)
def normalize_unicode(t: List[str]) -> List[str]:
    return [_normalize_unicode(i, UNICODE_FORM) for i in t]


@dispatch(str)
def normalize_whitespace(t: str) -> str:
    return _normalize_whitespace(t)


@dispatch(list)
def normalize_whitespace(t: List[str]) -> List[str]:
    return [_normalize_whitespace(i) for i in t]


@dispatch(str)
def replace_currency_symbols(t: str) -> str:
    return _replace_currency_symbols(t, CURRENCY_CODE)


@dispatch(list)
def replace_currency_symbols(t: List[str]) -> List[str]:
    return [_replace_currency_symbols(i, CURRENCY_CODE) for i in t]


@dispatch(str)
def replace_emojis(t: str) -> str:
    return _replace_emojis(t, EMOJI_CODE)


@dispatch(list)
def replace_emojis(t: List[str]) -> List[str]:
    return [_replace_emojis(i, EMOJI_CODE) for i in t]


@dispatch(str)
def replace_numbers(t: str) -> str:
    return _replace_numbers(t, NUMBER_CODE)


@dispatch(list)
def replace_numbers(t: List[str]) -> List[str]:
    return [_replace_numbers(i, NUMBER_CODE) for i in t]


@dispatch(str)
def replace_emails(t: str) -> str:
    return _replace_emails(t, EMAIL_CODE)


@dispatch(list)
def replace_emails(t: List[str]) -> List[str]:
    return [_replace_emails(i, EMAIL_CODE) for i in t]


@dispatch(str)
def replace_hashtags(t: str) -> str:
    return _replace_hashtags(t, HASHTAG_CODE)


@dispatch(list)
def replace_hashtags(t: List[str]) -> List[str]:
    return [_replace_hashtags(i, HASHTAG_CODE) for i in t]


@dispatch(str)
def replace_phone_numbers(t: str) -> str:
    return _replace_phone_numbers(t, PHONENUMBER_CODE)


@dispatch(list)
def replace_phone_numbers(t: List[str]) -> List[str]:
    return [_replace_phone_numbers(i, PHONENUMBER_CODE) for i in t]


@dispatch(str)
def replace_urls(t: str) -> str:
    return _replace_urls(t, URL_CODE)


@dispatch(list)
def replace_urls(t: List[str]) -> List[str]:
    return [_replace_urls(i, URL_CODE) for i in t]


@dispatch(str)
def replace_user_handles(t: str) -> str:
    return _replace_user_handles(t, USERHANDLER_CODE)


@dispatch(list)
def replace_user_handles(t: List[str]) -> List[str]:
    return [_replace_user_handles(i, USERHANDLER_CODE) for i in t]


@dispatch(str)
def remove_accents(t: str) -> str:
    return _remove_accents(t)


@dispatch(list)
def remove_accents(t: List[str]) -> List[str]:
    return [_remove_accents(i) for i in t]


@dispatch(str)
def remove_punctuation(t: str) -> str:
    return _remove_punctuation(t)


@dispatch(list)
def remove_punctuation(t: List[str]) -> List[str]:
    return [_remove_punctuation(i) for i in t]


# nltk and stop_words preprocessing tools
STOPWORDS = stopwords.words("english")
STOPWORDS.append(stop_words.get_stop_words("english"))


@dispatch(str)
def remove_stopwords(t: str) -> str:
    return t.translate(str.maketrans("", "", STOPWORDS))


@dispatch(list)
def remove_stopwords(t: List[str]) -> List[str]:
    return [i.translate(str.maketrans("", "", STOPWORDS)) for i in t]


# custom functions
@dispatch(str)
def lowercase(t: str) -> str:
    return t.lower()


@dispatch(list)
def lowercase(t: List[str]) -> List[str]:
    return [i.lower() for i in t]


__all__ = [
    "normalize_repeating_chars",
    "normalize_repeating_words",
    "normalize_hyphenated_words",
    "normalize_quotation_marks",
    "normalize_unicode",
    "normalize_whitespace",
    "replace_currency_symbols",
    "replace_emojis",
    "replace_numbers",
    "replace_emails",
    "replace_hashtags",
    "replace_phone_numbers",
    "replace_urls",
    "replace_user_handles",
    "remove_accents",
    "remove_punctuation",
    "remove_stopwords",
    "lowercase",
]
