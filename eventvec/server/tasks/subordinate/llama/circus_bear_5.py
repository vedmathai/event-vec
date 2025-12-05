# Will miss the play vs will attend the play. Does the order change?
# The order is not wrong though.
# Coversational implicature messes things up.
# Cancellations?
# Updated information.
# If you change the order of the if statements
# Combining more information.
# Same order jumble order
# Talking about the certainity of the concept as if it is a anaphora. So <proposition x>, I am not sure about proposition x.

# Structural
# Structural Level 1
# What are the chances of the proposition itself.
# What are the chances of the opposite verb.
# Compare two options

# Structural Level 2
# What are the chances of one step away.

# Structural Level 3
# What are the chances of multiple sentences
# backwards presentation
# random order
# Up the chain

# Structural Level 4
# Superfluous information
# Branching
# Branching backwards, random, up the chain
# Cycles

# Linguistic Levels
# Level 1
# Straight sentences
# Level 2
# Modals, if-thens, questions, nots, possible thats
# Level 3
# Conversational implicatures
# Level 4
# Cancellations and updated information
# Sarah may have ridden the bus even though she woke up late today.",
"Sarah said she hoped to wake up early today, but her alarm let her down.",
# Level 5
# Talking about the modality of a different topic
#"Sarah said that waking up early is crazy. But she is glad that she did today because she got to see the sunrise today.",
# Level 5
# Refering to ideas through anaphora and speaking directly about their probability



import random
from collections import defaultdict
import time

story1 = [
    "If Sarah wakes up late she will ride the bus.",
    "If Sarah wakes up early she will drive her car.",
    #"If Sarah wakes up early she goes for a walk.",
    #"If Sarah wakes up late she doesn't go on a walk.",
    "If Sarah rides the bus she will miss her meeting.",
    "If Sarah drives her car she will make her meeting.",
    "If Sarah sleeps late she will ride the bus.",
    "If Sarah sleeps late she will not eat breakfast.",
    "If Sarah doesn't eat breakfast she will miss her meeting.",
    "If Sarah misses her meeting she will be not be able to submit her project before the EOD deadline.",
    "If Sarah cannot submit her project before the EOD deadline she will be able to come home before the home-going traffic starts.",
    "If Sarah comes home before the home-going traffic starts she can go for yoga.",
    "If Sarah misses her meeting she will drink two cups of coffee.",
    "If Sarah makes her meeting she will skip coffee.",
    "If she drinks two cups of coffee she will not be able to take her dog for a walk."
]

story1 = story1[:4:1]
#print(story1)
#random.shuffle(story1)

story = ' '.join(story1)
#story = story3

cases = [
    
    #"'Sarah may have ridden the bus because she woke up early today.'",
    #"'I thought I saw Sarah on the bus today.'",
    #"'Sarah tells me she missed the bus today.'",
    #'I thought I saw Sarah on the bus today, but she tells me she missed it.'",
    #"'I am not sure if Sarah caught the bus today.'",
    #"'Sarah said that the bus was packed today and she stood the whole way.'",
    #"'Sarah said that the bus was packed today and she stood the whole way. But, the bus was empty.'",
    #"Sarah had told me that she thought that the Christmas tree looked beautiful today on her walk. But I would certainly doubt if it really did look that beautiful.",
    #"Sarah said that waking up early is crazy. But she is glad that she did today because she got to see the sunrise today.",
    "'Sarah said she has a back pain because her bus hit a bump in the morning and gave her a jolt.'",
    "'Sarah said she has a back pain because her car hit a bump in the morning and gave her a jolt.'",
    #"'Sarah may have ridden the bus even though she woke early today.'",
    #"'Sarah said her skirt was wet. It may have been because someone spilt something on the bus seat she was sitting on, but I am not sure, her skirt looked dry.'",
    
]

prompt1 = """
The following premise of if-else causal statements are all true that form our world-view. Reason over the task using only this world-view.
World-view: {}

I completely trust Lila. And take everything she says as truth unless she says so. I want to know {} given what Lila says.
Lila says, "{}"

{}? Say 'certain' if you are certain, 'uncertain' if you are uncertain or 'certainly not' if you are certain that the opposite is true. Give only your label, nothing else.


"""

prompt2 = """
{}

There are two alternative worlds in which Lila makes the following statements:
situation 1:
{}

situation 2:
{}

Hypothesis: {}
Answer with situation 1 or situation 2 or can't tell. Nothing else. Do not assume any logic that is not mentioned.

"""


from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4
from eventvec.server.tasks.entailment_classification.gpt_4.sambanova import sambanova
import csv

system_prompt = ""
cache = {}
data = []
with open('/home/lalady6977/oerc/projects/event-vec/eventvec/server/tasks/subordinate/llama/uncertainity_data.tsv', 'rt') as f:
    reader = csv.reader(f, delimiter=',')
    for di, d in enumerate(reader):
        if not 22 < di :
            continue
        print(len(d))
        if len(d[0]) == 0:
            continue
        data.append(d)

acc = defaultdict(list)
acc_line = defaultdict(lambda: defaultdict(list))
filtered_data = []
for di, d in enumerate(data):
    story = list(d[1:5])
    #random.shuffle(story)
    story = ' '.join(story[::1])
    if int(d[0]) % 2 == 1:
        ab = 'a'
        ba = 'b'
    else:
        continue
    filtered_data += [(story, d[5], d[7], d[9], d[13], ab, d[0])]
    filtered_data += [(story, d[5], d[7], d[10], d[14], ab, d[0])]
    #filtered_data += [(story, d[5], d[8], d[9], d[15], ba, d[0])]
    #filtered_data += [(story, d[5], d[8], d[10], d[16], ba, d[0])]

random.shuffle(filtered_data)
for d in filtered_data:
    story, typ, statement, question, correct_answer, ab, line = d
    for model in ['llama70']:
        prompt_filled = prompt1.format(story, question.strip('?'), statement, question.strip('?'))
        for prompti, prompt in enumerate([prompt_filled]):
            user_prompt = prompt
            if user_prompt not in cache:
                cache[user_prompt] = gpt_4(model, system_prompt, user_prompt)
            model_answer = cache[user_prompt]
            print(ab, statement, question)
            print('-->', model_answer, correct_answer)
            if model_answer.lower() == correct_answer.lower():
                acc[ab] += [1]
                acc_line[ab][line] += [1]
            else:
                acc[ab] += [0] 
                acc_line[ab][line] += [0]
            #print(f'Model: {model}', f'Prompt: {prompti+1}')
            #print(answer)
    time.sleep(10)
    for ab in acc:
        print(ab, sum(acc[ab])/len(acc[ab]))

    for ab in sorted(list(acc_line.keys())):
        for line in sorted(list(acc_line[ab].keys())):
            print(line, ab, sum(acc_line[ab][line])/len(acc_line[ab][line]))
