import os
from transformers import AutoTokenizer

# variable_value = os.environ.get("foo")
# if variable_value is not None:
#     print(variable_value)
#     print('I am from git')
# else:
#     print("Variable not found")
token = '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'
tok = AutoTokenizer.from_pretrained('Palash123/ola_eng_indic', use_auth_token=token)
print(tok)
