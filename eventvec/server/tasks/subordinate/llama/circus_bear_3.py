# Will miss the play vs will attend the play. Does the order change?
# The order is not wrong though.
# Coversational implicature messes things up.
# Cancellations?
# Updated information.
# If you change the order of the if statements
# Combining more information.
# Same order jumble order

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



import random


story1 = [
    "If Sarah skips her run, she will use that time to prepare slides for a presentation.",
    "If Sarah prepares the slides early, she will email them to her teammate, Jordan, for feedback.",
    "If Jordan receives the slides early, he will adjust his part of the presentation to match Sarah's updates.",
    "If Jordan adjusts his section, the presentation will have a more unified structure.",
    "If the presentation has a unified structure, the manager will approve it without requesting revisions.",
    "If the manager approves it early, the client meeting will be scheduled ahead of time.",
    "If the client meeting is scheduled early, the finance team will adjust their quarterly reporting timeline.",
    "If the finance team adjusts their timeline, the company will finalize its budget before the end of the month otherwise it will do it later.",
]

story2 = "If Sarah skips her run in the morning then the company will finalize its budget before the end of the month otherwise it will do it later."

story3 = [
    'If Sarah skips her run, she will use that time to prepare slides for a presentation.',
    'If Sarah prepares the slides early, she will email them to her teammate, Jordan, for feedback.',
    "If Jordan receives the slides early, he will adjust his part of the presentation to match Sarah's updates.",
    "If Jordan adjusts his section, the presentation will have a more unified structure.",
    "If the presentation has a unified structure, the manager will approve it without requesting revisions.",
    "If the manager approves it early, the client meeting will be scheduled ahead of time.",
    "If the client meeting is scheduled early, the finance team will adjust their quarterly reporting timeline.",
    "If the finance team adjusts their timeline, the company will finalize its budget before the end of the month otherwise it will do it later.",
    "If Jordan adjusts his section, then the presentation will be smaller and more streamlined.",
    "If the presentation is smaller and more streamlined then the investors will understand the value of the product better.",
    "If the investors understand the value of the product better then they will invest.",
    "If the investors invest then the company will finalize its budget before the end of the month otherwise it will do it later."
]

#random.shuffle(story3)
story = ' '.join(story3[::1])
#story = story3

cases = [
    #"Lila said, 'I thought I saw Sarah run past my school in the morning. But, the more I think about it, I think it was someone else.'",
    #"Lila said, 'I think the play is today.' and Pam said, 'I think the game is tonight.'",
    #"Lila said, 'If all goes well the play will take place today. and Pam said, 'I don't think the game is tonight.",
    #"Lila said, 'The play is either today or tomorrow.' and Pam said, 'The game is either tonight or tomorrow night.'",
    #"Lila said that she believes Sarah's 10-minute jog took her past the train station.'",
    "Lila said that the company looks it may finalize the budget well before the end of the month.",
    "Lila said hopes that the company may finalize the budget before the end of the month."
]

prompt1 = """
{}

I want to know if company will finalize its budget before the end of the month. I ask Lila about Sarah and she says:
{}

Give a score on -3 to 3 for the probability that the manager approved the presentation early? 3 is completely certain, 0 is completely uncertain. You can use decimals. Give only the score. Nothing else.


"""

prompt2 = """
{}

I want to know if company will finalize its budget after the end of the month. I ask Lila about Sarah and she says:
{}

Give a score on -3 to 3 for the probability that the manager approved the presentation too late? . 3 is completely certain, 0 is completely uncertain. You can use decimals. Give only the score. Nothing else.

"""

prompt3 = """
{}

There are two alternative worlds in which Lila makes the following statements:
Alternative 1:
{}

Alternative 2:
{}

Hypothesis: In which alternative is there a higher likelihood that the manager approved the presentation early?
Answer with Alternative 1 or Alternative 2. Nothing else.

"""

prompt4 = """
{}

There are two alternative worlds in which Lila made the following statements:
Alternative 1:
{}

Alternative 2:
{}

Hypothesis: In which alternative is there a higher likelihood that the manager approved the presentation too late?
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



