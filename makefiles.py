

import os

with open('questions.txt', 'r') as f:
    lines = f.readlines()

lines = [line.strip().split(' - ')[0] for line in lines]

lines = [line.replace(' ', '_') for line in lines]
lines = [line.replace('/', '_') for line in lines]

for line in lines:
    #make a java file for every element in lines
    with open(f'java/{line}.java', 'w+') as f:
        f.write('')
    
                
                