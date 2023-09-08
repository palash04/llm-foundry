import os
from transformers import AutoTokenizer

# variable_value = os.environ.get("foo")
# if variable_value is not None:
#     print(variable_value)
#     print('I am from git')
# else:
#     print("Variable not found")

tokenizer = AutoTokenizer.from_pretrained('Palash123/ola_eng_indic', token='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')
print(tokenizer)
