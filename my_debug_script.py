import json
import shutil
import sys
import os

from allennlp.commands import main

config_file = "depccg/models/my_allennlp/config/supertagger.jsonnet"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = '/tmp/serialization_dir_debug'

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

os.environ['vocab'] = '/Users/eagami/personal/development/hccg/data/heb_treebank/ccg/vocab'
os.environ['train_data'] = '/Users/eagami/personal/development/hccg/data/heb_treebank/ccg/ccg_10_3_19_train.auto'
os.environ['test_data'] = '/Users/eagami/personal/development/hccg/data/heb_treebank/ccg/ccg_10_3_19_dev.auto'
os.environ['encoder_type'] = 'lstm'
os.environ['token_embedding_type'] = 'char'

# Assemble the command into sys.argv
sys.argv = [
    'allennlp',
    'train',
    config_file,
    '-s', serialization_dir,
    '--include-package', 'depccg.models.my_allennlp',
    '-o', overrides
]



main()
