text = """
Context: 
Interview with Lila
I hope that they have a dancing bear at the circus today.

Interview with Mia:
I heard that there are going to be dancing bears at the circus today.

Interview with James:
The newspaper said that the circus is not in town today.

Interview with Omar:
I don't actually know if the circus is in town today. It is there usually this time of the year, so I would assume that it is there today.

Interview with Nora:
I was told that circus today has some 15 acts.

Interview with Ahmed:
I like the circus shows they put up every year. I was chatting with Mia and she said that there was going to be a circus today.

Interview with Sophia:
Lila insinuated that there was going to be circus today.

Interview with Daphne:
Today is Tuesday and there is no circus on tuesdays.

Hypothesis: The circus is in town today.

Attribute a 0 to 1 confidnce score to each of the above statements. Where +1 is very certain and 0 is very uncertain.
Attribute probabilities of the form P('there is a circus today|statement') and P('there is no circus today|statement') to each of the above statements.
Start with the prior probability of 0.5 for both P('there is a circus today') and P('there is no circus today').
Using Bayesian update, update each of the probabilities based on each statement in the context.
After processing all the statements, provide the final probabilities for the following three options:
1) The circus is in town today. 2) The circus is not in town today. 3) Neither is more likely than the other.


"""


from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3
from eventvec.server.tasks.entailment_classification.gpt_4.gpt_4_api import gpt_4
from eventvec.server.tasks.entailment_classification.gpt_4.sambanova import sambanova

system_prompt = ""
user_prompt = text
answer = sambanova(system_prompt, user_prompt)

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