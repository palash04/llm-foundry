import os

variable_value = os.environ.get("foo")
if variable_value is not None:
    print(variable_value)
    print('I am from git')
else:
    print("Variable not found")
