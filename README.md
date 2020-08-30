# DOCument SUMMARY
Python code auto summary an English doc in plain text.

How to use?

import spacy

nlp = spacy.load("en_core_web_lg")

summarize(nlp, file_name)

You may also need "python -m spacy download en_vectors_web_lg"
