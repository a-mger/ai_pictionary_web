#Timer in python
#pip install playsound
import time

question = int(input('How many seconds to wait: '))
time.sleep(question)
print(f'that was {question} seconds')