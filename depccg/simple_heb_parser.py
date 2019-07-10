
import argparse
import sys
import logging
import json
from lxml import etree

from .parser import EnglishCCGParser, JapaneseCCGParser, HebrewCCGParser
from .printer import print_
from depccg.tokens import Token, english_annotator, japanese_annotator, annotate_XX
from .download import download, load_model_directory, SEMANTIC_TEMPLATES, CONFIGS
from .utils import read_partial_tree, read_weights
from .combinator import en_default_binary_rules, ja_default_binary_rules, he_default_binary_rules
from .combinator import remove_disfluency, headfirst_combinator

Parsers = {'en': EnglishCCGParser, 'ja': JapaneseCCGParser, 'he': HebrewCCGParser}
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)


test_sentences = ['this is a test sentence .']
tokenize = False

binary_rules = he_default_binary_rules
annotate_fun = annotate_XX
kwargs = dict(
    unary_penalty=0.1,
    nbest=1,
    binary_rules=binary_rules,
    possible_root_cats=None,
    pruning_size=50,
    beta=1e-05,
    use_beta=True,
    use_seen_rules=True,
    use_category_dict=True,
    max_length=250,
    max_steps=10000000,
    gpu=-1
)

model = load_model_directory('en')
config = CONFIGS['en']
parser = Parsers['he'].from_json(config, model, **kwargs)

doc = [l.strip() for l in test_sentences]
doc = [sentence for sentence in doc if len(sentence) > 0]
tagged_doc = annotate_fun([[word for word in sent.split(' ')] for sent in doc],
                          tokenize=tokenize)
if tokenize:
    tagged_doc, doc = tagged_doc

res = parser.parse_doc(doc,
                       probs=None,
                       tag_list=None,
                       batchsize=32)

semantic_templates = SEMANTIC_TEMPLATES.get('en')
print_(res, tagged_doc,
       format='auto',
       lang='en',
       semantic_templates=semantic_templates)