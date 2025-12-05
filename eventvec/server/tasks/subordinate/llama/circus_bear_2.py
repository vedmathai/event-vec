# Will miss the play vs will attend the play. Does the order change?
# The order is not wrong though.
# Coversational implicature messes things up.
# Cancellations?
# Updated information.
# If you change the order of the if statements
# Combining more information.
# Same order jumble order

cases = [
    "Lila said, 'The play may be taking place today.',
    #"Lila said, 'I think the play is today.' and Pam said, 'I think the game is tonight.'",
    #"Lila said, 'If all goes well the play will take place today. and Pam said, 'I don't think the game is tonight.",
    #"Lila said, 'The play is either today or tomorrow.' and Pam said, 'The game is either tonight or tomorrow night.'",
    #"Lila said, 'My kid is playing the protagonist in today's play,' and Pam said, 'I hope the Raptors win tomorrow night.'",
]

prompt1 = """
If Jacob's friends are unhappy with him, then they will not call him for his birthday. He has dinner plans with his friends to watch the game. If he doesn't fly tonight then he can attend is daughter's play at her school in the evening. Jacob spoke to Lila and asked her what she knew about his daughter's play today evening. He also asked Pam when the game was. He has to choose whether he can fly to his meeting or not. He has a meeting in the morning and he has to fly tonight. If he flew tonight then he wouldn't be able to attend his daughter's play. However, if his daughter's play is in the evening then he won't be able to go to dinner with his friends and watch the game. If he can't go to meet them then he'll have to call them now and tell them he is not coming. If he does that they will be unhappy with him. If he meets them then they will be happy with him. However, if the game is not tonight then he can attend his daughter's play and then go to meet his friends later.


Lila made the following statments.
Statement:
{}

Give a score on -3 to 3 for the probability that Jacob's friends will call him for his birthday. 3 is certain, -3 is certain of the opposite and 0 is completely uncertain. You can use decimals. Give only the score. Nothing else.


"""

prompt2 = """

If Jacob's friends are unhappy with him, then they will not call him for his birthday. He has dinner plans with his friends to watch the game. If he doesn't fly tonight then he can attend is daughter's play at her school in the evening. Jacob spoke to Lila and asked her what she knew about his daughter's play today evening. He also asked Pam when the game was. He has to choose whether he can fly to his meeting or not. He has a meeting in the morning and he has to fly tonight. If he flew tonight then he wouldn't be able to attend his daughter's play. However, if his daughter's play is in the evening then he won't be able to go to dinner with his friends and watch the game. If he can't go to meet them then he'll have to call them now and tell them he is not coming. If he does that they will be unhappy with him. If he meets them then they will be happy with him. However, if the game is not tonight then he can attend his daughter's play and then go to meet his friends later.

Lila made the following statments.
Statement:
{}

Give a score on -3 to 3 for the probability that Jacob's friends will not call him for his birthday. 3 is certain, -3 is certain of the opposite and 0 is completely uncertain. You can use decimals. Give only the score. Nothing else.


"""

prompt3 = """
If Jacob's friends are unhappy with him, then they will not call him for his birthday. He has dinner plans with his friends to watch the game. If he doesn't fly tonight then he can attend is daughter's play at her school in the evening. Jacob spoke to Lila and asked her what she knew about his daughter's play today evening. He also asked Pam when the game was. He has to choose whether he can fly to his meeting or not. He has a meeting in the morning and he has to fly tonight. If he flew tonight then he wouldn't be able to attend his daughter's play. However, if his daughter's play is in the evening then he won't be able to go to dinner with his friends and watch the game. If he can't go to meet them then he'll have to call them now and tell them he is not coming. If he does that they will be unhappy with him. If he meets them then they will be happy with him. However, if the game is not tonight then he can attend his daughter's play and then go to meet his friends later.

There are two alternative worlds in which Lila and Pam made the following statements:
Alternative 1:
{}

Alternative 2:
{}

Hypothesis: In which alternative is there a higher likelihood that Jacob's friends will call him for his birthday?
Answer with Alternative 1 or Alternative 2. Nothing else.

"""

prompt4 = """
If Jacob's friends are unhappy with him, then they will not call him for his birthday. He has dinner plans with his friends to watch the game. If he doesn't fly tonight then he can attend is daughter's play at her school in the evening. Jacob spoke to Lila and asked her what she knew about his daughter's play today evening. He also asked Pam when the game was. He has to choose whether he can fly to his meeting or not. He has a meeting in the morning and he has to fly tonight. If he flew tonight then he wouldn't be able to attend his daughter's play. However, if his daughter's play is in the evening then he won't be able to go to dinner with his friends and watch the game. If he can't go to meet them then he'll have to call them now and tell them he is not coming. If he does that they will be unhappy with him. If he meets them then they will be happy with him. However, if the game is not tonight then he can attend his daughter's play and then go to meet his friends later.

There are two alternative worlds in which Lila and Pam made the following statements:
Alternative 1:
{}

Alternative 2:
{}

Hypothesis: In which alternative is there a higher likelihood that Jacob's friends will not call him for his birthday?
Answer with Alternative 1 or Alternative 2. Nothing else.

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
            prompt11_filled = prompt1.format(cases[casei])
            prompt12_filled = prompt1.format(cases[casei2])
            prompt21_filled = prompt2.format(cases[casei])
            prompt22_filled = prompt2.format(cases[casei2])
            prompt3_filled = prompt3.format(cases[casei], cases[casei2])
            prompt4_filled = prompt4.format(cases[casei], cases[casei2])
            for prompti, prompt in enumerate([prompt11_filled, prompt12_filled, prompt21_filled, prompt22_filled, prompt3_filled, prompt4_filled]):
                user_prompt = prompt
                if user_prompt not in cache:
                    cache[user_prompt] = sambanova(model, system_prompt, user_prompt)
                answer = cache[user_prompt]
                print('-'* 15)
                print(f'Model: {model}', f'Prompt: {prompti+1}')
                print(answer)

"""
Attribute a -3 to 3 credibility score to each of the above statements.
"""

"""
Use the Dempster-Shafer theory of evidence to combine the credibility scores.
"""

"""
Attribute a -3 to 3 credibility score to each of the above statements. Use the Bayes' theorm to combine the credibility scores.
"""

"""
Hypothesis: In which alternative is there a higher likelihood that there is a circus in town today?
Answer with Alternative 1 or Alternative 2. Nothing else.
"""