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
# Level 5
# Refering to ideas through anaphora and speaking directly about their probability



import random

story1 = [
    "If Sarah wakes up she will walk her dog.",
    "If Sarah walks her dog she will catch the bus.",
    "Everytime Sarah catches the bus she misses her meeting.",
    "If Sarah misses her meeting she will be shouted at by her boss.",
    "If Sarah is shouted at by her boss she will not focus at her meeting.",
    "If Sarah is shouted at by her boss she will cry.",
    "If Sarah cries she will call Dave and complain.",
    "If Sarah complains to Dave then he will be upset.",
    "If Dave is upset he will not focus at his meeting.",
    "Everytime Sarah meets her friends at the pub she pays and is late to go home.",
    "Everytime Sarah goes home late at night she catches the bus in the morning.",
]

story1 = story1[::-1]
random.shuffle(story1)

story = ' '.join(story1)
#story = story3

cases = [
    
    #"Lila said, 'I thought I saw Sarah run past my school in the morning. But, the more I think about it, I think it was someone else.'",
    #"Lila said, 'I think the play is today.' and Pam said, 'I think the game is tonight.'",
    #"Lila said, 'If all goes well the play will take place today. and Pam said, 'I don't think the game is tonight.",
    #"Lila said, 'The play is either today or tomorrow.' and Pam said, 'The game is either tonight or tomorrow night.'",
    #"Lila said that she believes Sarah's 10-minute jog took her past the train station.'",
    #"'I believe Sarah looked refreshed walking her dog in the morning.'",
    #"'I thought that Sarah looked nice last night at the pub.'"
    "Sarah's dog looked well-behaved today on his walk. But, I may be wrong, he could have been hungry.",
    "Sarah wanted to get a seat on the crowded bus today."
]

prompt1 = """
{}

I want to know if Dave will focus at his meeting. I ask Lila about Sarah and she says:
{}

Give a score on -3 to 3 for the probability that Dave will focus at his meeting? 3 is completely certain, 0 is completely uncertain. You can use decimals. Give only the score. Nothing else. Do not assume any logic that is not mentioned.


"""

prompt2 = """
{}

I want to know if Dave will be zoned out at his meeting. I ask Lila about Sarah and she says:
{}

Give a score on -3 to 3 for the probability that Dave will be zoned out at his meeting? . 3 is completely certain, 0 is completely uncertain. You can use decimals. Give only the score. Nothing else. Do not assume any logic that is not mentioned.

"""

prompt3 = """
{}

There are two alternative worlds in which Lila makes the following statements:
Alternative 1:
{}

Alternative 2:
{}

Hypothesis: In which alternative is there a higher likelihood that that Dave will focus at his meeting?
Answer with Alternative 1 or Alternative 2. Nothing else. Do not assume any logic that is not mentioned.

"""

prompt4 = """
{}

There are two alternative worlds in which Lila made the following statements:
Alternative 1:
{}

Alternative 2:
{}

Hypothesis: In which alternative is there a higher likelihood that Dave will be zoned out at his meeting?
Answer with Alternative 1 or Alternative 2. Nothing else. Do not assume any logic that is not mentioned.

"""

from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4
from eventvec.server.tasks.entailment_classification.gpt_4.sambanova import sambanova

system_prompt = ""
cache = {}
for model in ['gpt']:
    for casei in range(len(cases)):
        for casei2 in range(casei+1, len(cases)):
            print(cases[casei], '|', cases[casei2])
            prompt11_filled = prompt1.format(story, cases[casei])
            prompt12_filled = prompt1.format(story, cases[casei2])
            prompt21_filled = prompt2.format(story, cases[casei])
            prompt22_filled = prompt2.format(story, cases[casei2])
            prompt3_filled = prompt3.format(story, cases[casei], cases[casei2])
            prompt4_filled = prompt4.format(story, cases[casei], cases[casei2])
            for prompti, prompt in enumerate([prompt11_filled, prompt12_filled, prompt21_filled, prompt22_filled, prompt3_filled, prompt4_filled]):
                user_prompt = prompt
                if user_prompt not in cache:
                    cache[user_prompt] = sambanova(model, system_prompt, user_prompt)
                answer = cache[user_prompt]
                print('-'* 15)
                print(f'Model: {model}', f'Prompt: {prompti+1}')
                print(answer)



