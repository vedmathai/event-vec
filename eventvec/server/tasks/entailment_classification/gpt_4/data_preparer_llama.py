import numpy as np
import pprint
import re
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
import pprint
import random
from collections import defaultdict
from jadelogs import JadeLogger
import json
import os


from eventvec.server.config import Config
from eventvec.server.tasks.entailment_classification.gpt_4.llama_3_api import llama_3


from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.snli_data_reader import SNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.anli_data_reader import ANLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.connector_nli_data_reader import ConnectorNLIDatareader 

from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_data_reader import ChaosMNLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_snli_data_reader import ChaosSNLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_anli_data_reader import ChaosANLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_syntax_data_reader import ChaosMNLISyntaxDataReader  # noqa


all = {'135021n': 'entailment', '117177n': 'neutral', '8111n': 'neutral', '135401n': 'entailment', '46198n': 'neutral', '32889n': 'contradiction', '86472e': 'entailment', '49123n': 'contradiction', '82161n': 'neutral', '58557e': 'entailment', '26766c': 'entailment', '56634c': 'contradiction', '59934c': 'entailment', '87332c': 'contradiction', '98445c': 'contradiction', '129492c': 'contradiction', '105561e': 'entailment', '28078c': 'contradiction', '15197n': 'neutral', '46063n': 'entailment', '121910e': 'contradiction', '79507e': 'entailment', '9557e': 'entailment', '143751e': 'contradiction', '15727c': 'entailment', '118403e': 'entailment', '106091e': 'entailment', '18892n': 'contradiction', '95883n': 'neutral', '136360n': 'neutral', '59934n': 'neutral', '134356n': 'entailment', '18874n': 'neutral', '117680c': 'entailment', '76947n': 'neutral', '95186n': 'neutral', '14545e': 'entailment', '144753c': 'contradiction', '40710n': 'neutral', '66225n': 'neutral', '91768n': 'neutral', '36469e': 'entailment', '53499c': 'entailment', '101809c': 'neutral', '92771n': 'entailment', '27036c': 'neutral', '26495e': 'entailment', '12562n': 'entailment', '111005e': 'contradiction', '19186c': 'neutral', '125910n': 'neutral', '15100e': 'contradiction', '140005c': 'contradiction', '30221c': 'contradiction', '13387e': 'neutral', '146070e': 'entailment', '125238c': 'contradiction', '118652e': 'neutral', '111167n': 'neutral', '113967e': 'neutral', '47714e': 'entailment', '81812e': 'neutral', '62287n': 'entailment', '47877c': 'contradiction', '83247e': 'entailment', '102923n': 'entailment', '126583n': 'entailment', '58331c': 'contradiction', '102857n': 'entailment', '77175n': 'neutral', '42745e': 'entailment', '46198c': 'contradiction', '66225e': 'entailment', '105790n': 'entailment', '13652n': 'contradiction', '44493n': 'contradiction', '101809n': 'entailment', '102817n': 'neutral', '123038e': 'entailment', '82415n': 'neutral', '73751n': 'neutral', '16989e': 'entailment', '48376n': 'neutral', '27907n': 'entailment', '89209n': 'neutral', '86184c': 'contradiction', '67412n': 'entailment', '103354n': 'entailment', '128542e': 'neutral', '99194e': 'neutral', '48557c': 'contradiction', '44834e': 'entailment', '73191n': 'entailment', '118415n': 'neutral', '13964e': 'neutral', '9393n': 'entailment', '45108n': 'entailment', '13263n': 'contradiction', '86886n': 'contradiction', '1408c': 'neutral', '127490e': 'entailment', '1682n': 'neutral', '80930n': 'neutral', '40738n': 'entailment', '119421n': 'neutral', '75320c': 'neutral', '111011e': 'neutral', '118999n': 'contradiction', '83774c': 'contradiction', '113193n': 'neutral', '53074e': 'entailment', '130928n': 'contradiction', '18189e': 'contradiction', '23774n': 'contradiction', '138500e': 'entailment', '83248c': 'contradiction', '137715c': 'neutral', '46059n': 'entailment', '110061n': 'neutral', '32754e': 'entailment', '140440c': 'contradiction', '36924n': 'neutral', '138359n': 'neutral', '114971n': 'neutral', '54822n': 'neutral', '47877n': 'neutral', '16996e': 'entailment', '109510n': 'neutral', '107302e': 'entailment', '53074n': 'neutral', '9022e': 'entailment', '122531n': 'contradiction', '103616e': 'entailment', '103482c': 'neutral', '65353n': 'contradiction', '117487e': 'entailment', '142298n': 'neutral', '21671e': 'entailment', '8219c': 'contradiction', '53468n': 'entailment', '139409n': 'neutral', '117487c': 'contradiction', '87332n': 'entailment', '76483n': 'entailment', '117093n': 'neutral', '108847e': 'entailment', '35422c': 'contradiction', '14388c': 'contradiction', '141642n': 'neutral', '84055n': 'contradiction', '100349e': 'neutral', '16494c': 'contradiction', '124853e': 'entailment', '143608n': 'neutral', '62982n': 'neutral', '62177n': 'entailment', '31054n': 'entailment', '6105n': 'neutral', '43168c': 'contradiction', '105911c': 'contradiction', '95953c': 'contradiction', '12815n': 'contradiction', '39678e': 'contradiction', '101104n': 'neutral', '114971e': 'entailment', '55110c': 'neutral', '65199e': 'neutral', '15488c': 'neutral', '123675n': 'contradiction', '57151c': 'neutral', '107935n': 'neutral', '100373c': 'contradiction', '80458n': 'neutral', '27907e': 'contradiction', '53545c': 'contradiction', '133966e': 'entailment', '113280n': 'neutral', '118141n': 'neutral', '43094c': 'contradiction', '83657c': 'contradiction', '70711c': 'entailment', '74768e': 'contradiction', '53953n': 'neutral', '128176n': 'neutral', '96516e': 'entailment', '77116e': 'neutral', '79007n': 'entailment', '116111e': 'entailment', '86008n': 'entailment', '50415n': 'contradiction', '65199n': 'neutral', '63013n': 'neutral', '129201n': 'neutral', '86429n': 'neutral', '808n': 'neutral', '138966n': 'neutral', '95338n': 'contradiction', '95663n': 'neutral', '2873n': 'neutral', '43365c': 'contradiction', '59208n': 'neutral', '16086n': 'contradiction', '42321n': 'neutral', '111243n': 'entailment', '18991n': 'neutral', '72389c': 'contradiction', '90839n': 'entailment', '95155e': 'neutral', '71832n': 'neutral', '9047c': 'contradiction', '43225n': 'neutral', '917c': 'contradiction', '101467n': 'neutral', '127809n': 'entailment', '8257n': 'neutral', '36924e': 'neutral', '112592n': 'neutral', '45306n': 'neutral', '58377n': 'neutral', '4795n': 'entailment', '142604e': 'entailment', '94241e': 'entailment', '57163n': 'entailment', '50830n': 'neutral', '142630n': 'contradiction', '96956c': 'contradiction', '83722c': 'contradiction', '117900n': 'neutral', '91797c': 'contradiction', '77175c': 'neutral', '56909n': 'neutral', '53027n': 'contradiction', '113668n': 'neutral', '11534n': 'neutral', '133005c': 'neutral', '83298n': 'neutral', '141321n': 'contradiction', '142964n': 'neutral', '130680n': 'neutral', '82700c': 'contradiction', '6386n': 'neutral', '13221c': 'neutral', '38784n': 'contradiction', '92422e': 'entailment', '110671n': 'neutral', '3799n': 'contradiction', '15537n': 'entailment', '133456n': 'neutral', '97367n': 'neutral', '13760n': 'neutral', '131235n': 'contradiction', '80220n': 'neutral', '61429c': 'neutral', '98621n': 'contradiction', '49462c': 'contradiction', '67063n': 'entailment', '40710c': 'neutral', '70711n': 'neutral', '122197n': 'neutral', '85838e': 'entailment', '51356n': 'contradiction', '14556n': 'entailment', '24043e': 'entailment', '136860c': 'contradiction', '50484c': 'neutral', '42860n': 'entailment', '144173n': 'neutral', '80643e': 'entailment', '137708c': 'entailment', '82182e': 'neutral', '55241e': 'entailment', '104399n': 'neutral', '34920c': 'neutral', '82069e': 'neutral', '121910c': 'entailment', '2262n': 'entailment', '77025c': 'neutral', '119175n': 'neutral', '102817c': 'entailment', '45089c': 'contradiction', '140615e': 'neutral', '125700n': 'neutral', '57345e': 'entailment', '61726n': 'entailment', '78105e': 'entailment', '23725n': 'entailment', '4035c': 'neutral', '121677n': 'entailment', '136752n': 'neutral', '144207n': 'contradiction', '98116c': 'neutral', '67610e': 'neutral', '108243e': 'entailment', '837n': 'entailment', '42388e': 'neutral', '36514n': 'neutral', '56759n': 'neutral', '24103n': 'contradiction', '81469n': 'contradiction', '75838n': 'neutral', '137319n': 'neutral', '127073c': 'neutral', '13531e': 'entailment', '60212c': 'contradiction', '15110n': 'neutral', '33764c': 'neutral', '32851c': 'entailment', '138272n': 'neutral', '73278n': 'contradiction', '69806c': 'neutral', '130869c': 'entailment', '115391n': 'neutral', '102708e': 'neutral', '100637e': 'entailment', '2529c': 'contradiction', '43247n': 'entailment', '145495n': 'entailment', '58557n': 'entailment', '28387c': 'entailment', '17950c': 'contradiction', '5507n': 'contradiction', '3839c': 'contradiction', '47260n': 'neutral', '100248n': 'neutral', '131910n': 'entailment', '119901n': 'neutral', '96539c': 'neutral', '91913n': 'entailment', '107020c': 'entailment', '120896n': 'entailment', '105904c': 'neutral', '8005c': 'contradiction', '31263n': 'entailment', '129081n': 'neutral', '115830n': 'contradiction', '35290n': 'entailment', '130482c': 'contradiction', '14191n': 'entailment', '40867c': 'neutral', '122928e': 'entailment', '68946c': 'neutral', '100792c': 'contradiction', '1735n': 'contradiction', '93839c': 'contradiction', '30894n': 'neutral', '99791c': 'entailment', '102174e': 'entailment', '139028e': 'neutral', '37660c': 'contradiction', '88879n': 'contradiction', '32197n': 'neutral', '91709c': 'neutral', '134514n': 'entailment', '76957c': 'contradiction', '79265n': 'neutral', '117576c': 'neutral', '53211n': 'neutral', '122645n': 'neutral', '133842n': 'entailment', '54811n': 'entailment', '77590c': 'entailment', '17950e': 'entailment', '7091e': 'neutral', '71974n': 'neutral', '40867e': 'neutral', '84901n': 'neutral', '129464n': 'neutral', '119758n': 'neutral', '24103e': 'neutral', '127858c': 'contradiction', '61767e': 'contradiction', '33764n': 'entailment', '126768n': 'entailment', '115593c': 'neutral', '78322e': 'entailment', '31113e': 'contradiction', '49970c': 'contradiction', '102817e': 'entailment', '129081e': 'neutral', '72870n': 'neutral', '46576e': 'neutral', '49227n': 'neutral', '56075n': 'neutral', '30450c': 'contradiction', '73156n': 'contradiction', '142238n': 'neutral', '50480n': 'entailment', '33340e': 'neutral', '44861e': 'entailment', '18086n': 'neutral', '89995e': 'neutral', '77875e': 'neutral', '139677n': 'neutral', '8269e': 'neutral', '43764n': 'contradiction', '67938e': 'contradiction', '23765e': 'entailment', '43178e': 'entailment', '72315c': 'contradiction', '32851e': 'entailment', '7856c': 'contradiction', '77152e': 'entailment', '137715n': 'entailment', '4082e': 'entailment', '94674c': 'contradiction', '118403n': 'neutral', '71957n': 'contradiction'}


prompt_preamble = """
[INST] <<SYS>>

    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to decide whether the hypothesis
    can be inferred from the premise.
     
    You will be given three options: entailment, neutral, and contradiction. 
    If the hypothesis can be inferred from the premise, you should select entailment.
    If the hypothesis is unrelated to the premise, you should select neutral.
    If the hypothesis contradicts the premise, you should select contradiction.

    Event credence is the degree of belief that an event will occur on a score between -3 and 3 where -3 is certainly won't happen and 3 is certainly did happen.
    The linguistic factors that effect event credence are:
    - The presence of modals
    - The presence of adverbs
    - The presence of negation
    - The presence of tense
    - The infinitive verb that is subordinate to the main verb such as expected to, intended to, etc.
    - The event is a subordinated clause of a speech verb such as said, told, etc. or a belief verb such as believe, think, etc.
    
    Credence has an effect on the entailment task. Because if the premise and hypothesis are talking about the same event then they may entail if the credences are the same.
    Use this reasoning to predict the label for the examples.
    The first five are examples with the labels provided.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'answer: index: label'. 

    Examples:

    """

prompt_preamble = """
[INST] <<SYS>>

    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to decide whether the hypothesis
    may be inferred from the premise.

    The label is one of three options: entailment, neutral, and contradiction. 
    If the hypothesis can be inferred from the premise, you should select entailment.
    If the hypothesis is unrelated to the premise, you should select neutral.
    If the hypothesis contradicts the premise, you should select contradiction.
 
    
    The first five are examples with the labels provided.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'answer: index: label'. 
    
    Examples:    
    """

prompt_preamble_connector = """
[INST] <<SYS>>

    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to decide whether the hypothesis
    may be inferred from the premise.

    This is the task of natural language understanding, where given the premise and hypothesis, you would have to classify whether 
    1. strict: The premise strictly entails the hypothesis, which means the individual clauses entail and the overall sentence entails entails in both directions.

    2. non_strict: The premise non-strictly entails the hypothesis, which means the individual clauses entail but the overall sentence has some information loss such as causality.

    3. contradicts: Premise contradicts the hypothesis: Even though the clauses individually entail, the overall sentences contradict each other.

    Do not provide any explanation just provide the classification.
    The first five are examples with the labels provided.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'answer: index: label'. 
    
    Examples:    
    """

prompt_preamble_connector_2 = """
[INST] <<SYS>>

    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to decide whether the hypothesis
    may be inferred from the premise.

    This is the task of natural language understanding, where given the premise and hypothesis, you would have to classify whether 
    1. strict: The premise strictly entails the hypothesis, which means the individual clauses entail and the overall sentence entails in both directions.

    2. non_strict: The premise non-strictly entails the hypothesis, which means the individual clauses entail but the overall sentence has some information loss such as causality.

    3. contradicts: Premise contradicts the hypothesis: Even though the clauses individually entail, the overall sentences contradict each other.

    Do not provide any explanation just provide the classification.

    The first five are examples with the labels provided.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'answer: index: label'. 

    Hint: 
    Use the logic of sentence connectors 'because', 'and', 'but' and 'so'
    The combination of sentence connector and clause order determines the label.


    'and' contradicts 'though' a and b contradicts a though b and contradicts b though a.
    'and' contradicts 'so' a and b contradicts a so b and contradicts b so a.
    'and' contradicts 'because' a and b contradicts a because b and contradicts b.
    'and' contradicts 'but' a and b contradicts a but b and contradicts b but a.
    'and' strictly entails 'and'. a and b strictly entails a and b and entails b and a.

    'Because' is opposite of 'so'. a because b strictly entails b so a. a because b contradicts a so b.
    'because' contradicts 'but'. a because b contradicts a but b. a because b contradicts b but a.
    'because' non-strictly entails 'and'. a because b non-strictly entails a and b or b and a
    'because' contradicts 'though'. a because b contradicts a though b.
    'because' strictly entails 'because'. a because b strictly entails a because b.
     a because b contradicts b because a.

    'but' non-strictly entails 'and'. a but b non-strictly entails a and b. a but b non-strictly entails b and a.
    'but' contradicts 'because'. a but b contradicts a because b. a but b contradicts b because a.
    'but' contradicts 'so'. a but b contradicts a so b.
    'but' is the opposite of 'though'. a but b contradicts a though b. a but b strictly entails b though a.
    'but' strictly entails 'but' in one direction. a but b contradicts a but b.
    a but b contradicts b but a.

    'so' is the opposite of 'because'. a so b contradicts a because b. a so b strictly entails b because a.
    'so' contradicts 'but'. a so b contradicts a but b. a so b contradicts b but a.
    'so' non-strictly entails 'and'. a so b non-strictly entails a and b. a so b non-strictly entails b and a.
    'so' contradicts 'though'. a so b contradicts a though b. a so b contradicts b though a.
    'so' strictly entails 'so' in one direction. a so b strictly entails a so b. a so b contradicts b so a.


    'though' contradicts 'so'. a though b contradicts a so b. a though b contradicts b so a.
    'though' contradicts 'because'. a though b contradicts a because b. a though b contradicts b because a.
    'though' is the opposite of 'but'. a though b strictly entails b but a. a though b contradicts b but a.
    'though' non-strictly entails 'and'. a though b non-strictly entails a and b. a though b non-strictly entails b and a.
    'though' strictly entails 'though' in one direction. a though b strictly entails a though b. a though b contradicts b though a.


    
    Examples:    
    """

prompt_preamble_connector_3 = """
[INST] <<SYS>>

    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to decide whether the hypothesis
    may be inferred from the premise.

    This is the task of natural language understanding, where given the premise and hypothesis, you would have to classify whether 
    1. strict: The premise strictly entails the hypothesis, which means the individual clauses entail and the overall sentence entails in both directions.

    2. non_strict: The premise non-strictly entails the hypothesis, which means the individual clauses entail but the overall sentence has some information loss such as causality.

    3. contradicts: Premise contradicts the hypothesis: Even though the clauses individually entail, the overall sentences contradict each other.

    Do not provide any explanation just provide the classification.

    The first five are examples with the labels provided.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'answer: index: label'. 

    Hint: 
    Use the logic of sentence connectors 'because', 'and', 'but' and 'so'
    The combination of sentence connector and clause order determines the label.

    To fill in the mask analyse the following structure:

    And ask the following questions:
    A but B: if A then B is surprising.
    A so B: if A then B follows. A is the reason.
    A though B: if B then A is surprising.
    A because B: if A then B is the reason. 
    A and B: not contrasting not causal


    
    Examples:    
    """

prompt_preamble_contrast = """
[INST] <<SYS>>

    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to decide whether the hypothesis
    may be inferred from the premise.

    The label is one of three options: entailment, neutral, and contradiction. 
    If the hypothesis can be inferred from the premise, you should select entailment.
    If the hypothesis is unrelated to the premise, you should select neutral.
    If the hypothesis contradicts the premise, you should select contradiction.

    The use of 'but', 'though' etc. indicate constrast or contradiction.
    Usually when phrase 1 CONTRASTS phrase 2, phrase 2 may be cancelled or negated.
    Phrase 1 is just talking about a related topic as phrase 2 and not negating it.

    Such as:
    Original sentence: People say I am 5 feet tall but I am actually 5'2.
    1: I am actually 5'2 CONTRASTS People say I am 5 feet tall. 
    explanation: I am 5'2 directly cancels and clarifies the first proposition.

    Original Sentence: They say there are no aliens however we don't really know that.
    2: we don't really know that CONTRASTS They say there are no aliens. 
    explanation 2: The truth value of aliens not being there is negated, because the first phrase directly contradicts the first.
    
    Original Sentence: We can go for a walk though it is raining.
    3: it is raining CONTRASTS We can go for a walk.
    explanation 3: It is implied you would not normally go for a walk during a rain

    Sometimes the contrast is not explicit:
    Original Sentence: However, I am making potato salad.
    4: I am making potato salad CONTRASTS
    explanation 4: making a potato salad is contrasting something that is not mentioned but implies that something is not a good idea.

    Original Sentence: On the other hand, I am out of fuel.
    5: I am out of fuel CONTRASTS
    explanation 5: being out of fuel contrasts something that is not mentioned but implies I needed fuel.
    

    Use this in your reasoning to predict the label given the premise and hypothesis.
    The first five are examples with the labels provided.

    Your task is providing an explanation as to what is implied or negated by the contradition.
    Use the explanation if applicable to predict the label for the given examples. 
    provide in the format of 'answer: index: label'.
    
    Examples:    
    """

prompt_preamble_2 = """
[INST] <<SYS>>

    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to decide whether the hypothesis
    may be inferred from the premise.

    The first five are examples with the labels provided.

    Your task is to predict the label for the given examples. Do not provide reasoning and 
    provide in the format of 'answer: index: label'. 

    Examples:

    """

credence_prompt_preamble = """
    Textual entailment is the task of determining whether a given hypothesis can be inferred from a given premise.
    In this task, you will be given a premise and a hypothesis, and you will have to decide whether the hypothesis
    can be inferred from the premise.

    You are given three options: entailment, neutral, and contradiction. 
    If the hypothesis can be inferred from the premise, you should select entailment.
    If the hypothesis is unrelated to the premise, you should select neutral.
    If the hypothesis contradicts the premise, you should select contradiction.
    
    Credence has an effect on the entailment task. Because if the premise and hypothesis are talking about the same event then they may effect the label decision.
    Use the credence to inform the choice of label.

    Event credence is the degree of belief that an event will occur on a score between -3 and 3 where -3 is certainly won't happen and 3 is certainly did happen. It can be a decimal number.
        The linguistic factors that effect event credence are:
        - The presence of modals
        - The presence of adverbs
        - The presence of negation
        - The presence of tense
        - The infinitive verb that is subordinate to the main verb such as expected to, intended to, etc.
        - The event is a subordinated clause of a speech verb such as said, told, etc. or a belief verb such as believe, think, etc.
    
    
    Use these examples to predict credence:
    Example: John may go to the store. Event: go to the store. Credence: 1.5
    Example: Mary said that John will go to the store. Event: said. Credence: 2.8
    Example: Mary believes that John will go to the store. Event: said. Credence: 2.05
    Example: Mary said that John will go to the store. Event: go. Credence: 2.27
    Example: Mary believes that John will go to the store. Event: go. Credence: 1.97
    Example: If John goes to the store he will buy a cake. Event: go to the store. Credence: 1.12
    Example: John will not go to the store. Event: go to the store. Credence: -2.5

    The examples have the premise and hyposthesis on one line, the label on the third line.

    premise: <premise>
    hypothesis: <hypothesis>
    answer: <index> : <label>
    Examples:

    """
extra = """
Event credence is the degree of belief that an event will occur on a score between -3 and 3 where -3 is certainly won't happen and 3 is certainly did happen. It can be a decimal number.
        The linguistic factors that effect event credence are:
        - The presence of modals
        - The presence of adverbs
        - The presence of negation
        - The presence of tense
        - The infinitive verb that is subordinate to the main verb such as expected to, intended to, etc.
        - The event is a subordinated clause of a speech verb such as said, told, etc. or a belief verb such as believe, think, etc.
"""

credence_prompt_preamble2 = """
    For the last example provide whether the hypothesis entails, contradicts or is neutral to the premise.
     
    Provide the degree of belief that an event will occur on a score between -3 and 3 where -3 is certainly won't happen and 3 is certainly did happen.
    
    The question is in the form,
    index, premise, hypothesis.

    Fill in the blanks for the following example:


    
    Use this to predict the label for the last example.
    The first five are examples with the labels provided.
    Examples:

    """

"""The first five are examples with the labels provided."""
cot_prompt = """
Event credence is the degree of belief that an event will occur.
    The linguistic factors that effect event credence are:
    - The presence of modals
    - The presence of adverbs
    - The presence of negation
    - The presence of tense
    - The infinitive verb that is subordinate to the main verb such as expected to, intended to, etc.
    - The event is a subordinated clause of a speech verb such as said, told, etc. or a belief verb such as believe, think, etc.

    Use this reasoning to predict the label for the example.
"""

class NLIDataPreparer():
    def __init__(self):
        self._data_readers = {
            'mnli': MNLIDataReader(),
            'snli': SNLIDataReader(),
            'anli': ANLIDataReader(),
            'cnli': ConnectorNLIDatareader(),
        } 

        self._chaos_data_readers = {
            'mnli': ChaosMNLIDatareader(),
            'snli': ChaosSNLIDatareader(),
            'anli': ChaosANLIDatareader(),
            'mnli_syntax': ChaosMNLISyntaxDataReader(),
        }

    def load(self):
        k = 0
        file_name = 'llama_3_connectors_70b_helped_3.json'
        window = 1
        jl = JadeLogger()
        gpt_answer = {}
        true_answers = {}
        data_reader = self._data_readers['cnli']
        data = data_reader.read_file('test').data()[:1800]
        example_data = [data[0], data[43], data[124], data[630], data[173], data[250]]
        data = [datum for datum in data if datum not in example_data]
        system_prompt = str(prompt_preamble_connector_3)
        location = jl.file_manager.data_filepath(file_name)
        if os.path.exists(location):
            with open(location, 'rt') as f:
                gpt_answer = json.load(f)
        for datumi, datum in enumerate(example_data, start=1):


            system_prompt += f'{datum.uid()} \n Premise: ' + datum.sentence_1() + '\n'
            system_prompt += 'Hypothesis: ' + datum.sentence_2() + '\n'
            
            #system_prompt += f'For the data point {datum.uid()} with the premise "{datum.sentence_1()}" and the hypothesis "{datum.sentence_2()}".\n\n The premise credence, hypothesis credence and label is 2, 2\n Answer is: {datum.label()} \n\n'
            #system_prompt +=  '2 : '
            #system_prompt += '2 : '
            #system_prompt += f'{datum.label()} \n\n'
            #system_prompt += f'{datum.uid()} : 2.4 : 2.5 \n'
            #system_prompt += f'Answer: {datum.uid()} :  {datum.label()} \n\n'
            #system_prompt += f'Answer: {datum.uid()} : <premise credence> : <hypothesis credence> : {datum.label()} \n\n'
            #system_prompt += f'Contrast Explanation: <replace with explanation for contrast>\n'
            system_prompt += f'Answer: {datum.uid()} :' + datum.label() +  '\n\n'


        while k * window < len(data):
            print(k, len(data) // window)
            user_prompt_credence = ""
            user_prompt_normal = ""
            user_prompt = ''


            for datumi, datum in enumerate(data[0 + (k*window): (k+1) * window + 0], start=6):
                if str(datum.uid()) in gpt_answer:
                    continue
                print('premise:', datum.sentence_1())
                print('hypothesis:', datum.sentence_2())
                user_prompt_credence += f"""
                    <</SYS>>

                    Provide the labels for the following sentences in two line format of
                    Answer: <index>: <premise credence> : <hypothesis credence> : <label>
                """
                user_prompt_normal += f"""
                    <</SYS>>

                    Provide the labels for the following sentences in the format of 'answer: index: label'.
                """
                user_prompt = user_prompt_normal
                true_answers[datum.uid()] = datum.label()
                user_prompt += f'{datum.uid()} Premise: ' + datum.sentence_1() + '\n'
                user_prompt += 'Hypothesis: ' + datum.sentence_2() + '\n [/INST] \n'
                print("prompting", datum.uid(), datum.label())
            print('sending prompt')
            if len(user_prompt) == 0:
                k += 1
                continue
            answer = llama_3(system_prompt, user_prompt)
            print('received response')
            for line in answer.split('\n'):
                if 'answer' in line.lower():
                    print(line)
                    try: 
                        #_, index, premise_credence, hypothesis_credence, label = line.split(':')
                        _, index, label = line.split(':')
                        if index.strip() == datum.uid():
                            #gpt_answer[index.strip()] = [label.strip(), premise_credence.strip(), hypothesis_credence.strip()]

                            gpt_answer[index.strip()] = [label.strip(), '', '']

                    except ValueError:
                        continue
            print(self.f1_score(true_answers, gpt_answer))
            k += 1
            with open(location, 'wt') as f:
                f.write(json.dumps(gpt_answer))

    def f1_score(self, true_answers, gpt_answers):
        tp = 0
        fp = 0
        fn = 0
        f1 = 0
        for uid, label in true_answers.items():
            if uid not in gpt_answers:
                continue
            if len(gpt_answers[uid][0]) > 0 and label[0:2] == gpt_answers[uid][0][0:2]:
                tp += 1
            else:
                fp += 1
                fn += 1
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
        return f1

if __name__ == '__main__':
    Config.instance()
    data_preparer = NLIDataPreparer()
    data_preparer.load()