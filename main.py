from huggingface_hub import login
from optimum.quanto import *
from transformers import *

token_question=int(input('Do you want to enter your Hugging Face token?\n1. Yes\n2. No\n>> '))
while token_question not in [1,2]:
    print('Token authentication error! Select valid option')
    token_question=int(input('Do you want to enter your Hugging Face token?\n1. Yes\n2. No\n>> '))
    
if token_question==1: 
    login(token=input('Enter your token\n>> '))
elif token_question==2: 
    pass
d = input('Which model to quantize? (Example: meta-llama/llama3.2-1b)\n>> ')
weights_type = int(input(f'Select weight type (by number):\n1. int2\n2. int4\n3. int8\n4. float8\n>> '))
while weights_type not in [1,2,3,4]:
    print('Invalid weight type! Select number between 1-4')
    weights_type=int(input(f'Select weight type (by number):\n1. int2\n2. int4\n3. int8\n4. float8\n>> '))
if weights_type==1: weights_type=qint2
elif weights_type==2: weights_type=qint4
elif weights_type==3: weights_type=qint8
elif weights_type==4: weights_type=qfloat8
exclude_type = int(input(f'Which exclude type?\n1. lm_head\n>> '))
while exclude_type!=1:
    print('Invalid exclusion type! Only lm_head is currently in code.')
    exclude_type=int(input(f'Which exclude?\n1. lm_head\n>> '))
if exclude_type==1: exclude_type='lm_head'
save_path=input('Enter model save folder path (example: ./my_quantized_model)\n>> ')
model = AutoModelForCausalLM.from_pretrained(d)
qmodel = QuantizedModelForCausalLM.quantize(model,weights=weights_type,exclude=[exclude_type])
qmodel.save_pretrained(save_path)
print(f'Quantized model successfully saved in {save_path}')
