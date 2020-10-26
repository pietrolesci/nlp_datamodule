# Introduction
A first attempt to design a LightningDataModule for NLP.


# Structure
The `utils.py` file collects some of the most common preprocessing steps. Instead of reinventing the wheel, I tried to reuse existing functionalities provided in other libraries. In particular, for now I am leveraging [textacy](https://textacy.readthedocs.io/en/stable/api_reference/text_processing.html#textacy.preprocessing.remove.remove_punctuation) and [fastai](https://docs.fast.ai/text.core#spec_add_spaces). Since the functions must be able to work on `str` and `List[str]`, I used multiple dispatch to create different methods based on the input type. To accomplish this I use the [multiple-dispatch](https://multiple-dispatch.readthedocs.io/en/latest/design.html#types) library.

The `nlp_module.py` implements the main class. There are still a few design choices that needs to be made. In particular:

- How to implement numericalization supporting both dense (e.g., word vectors _à la_ Word2Vec, or something particular like [PRADO](https://ai.googleblog.com/2020/09/advancing-nlp-with-efficient-projection.html?m=1)) and sparse (e.g., tf-idf)
    - In both cases we need to build a vocabulary which needs a frequency count -> implementing these things is one of the most valuable contributions

- General API design: efficiency vs flexibility, plus make it compatible with the general Pytorch-Lightning approach


However, two things remain fixed:
- A tokenizer always accepts `str` and returns `List[str]`
- Numericalization always accepts `List[str]` and return `List[int]`
