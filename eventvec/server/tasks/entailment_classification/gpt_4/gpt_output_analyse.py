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
import csv

from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer
from eventvec.server.tasks.entailment_classification.featurizers.clause_matcher import ClauseMatcher
from eventvec.server.config import Config

from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.snli_data_reader import SNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.anli_data_reader import ANLIDataReader  # noqa

from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_data_reader import ChaosMNLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.mnli_syntax_data_reader import MNLISyntaxDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_syntax_data_reader import ChaosMNLISyntaxDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_snli_data_reader import ChaosSNLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_anli_data_reader import ChaosANLIDatareader  # noqa

all = {'135021n': 'entailment', '117177n': 'neutral', '8111n': 'neutral', '135401n': 'entailment', '46198n': 'neutral', '32889n': 'contradiction', '86472e': 'entailment', '49123n': 'contradiction', '82161n': 'neutral', '58557e': 'entailment', '26766c': 'entailment', '56634c': 'contradiction', '59934c': 'entailment', '87332c': 'contradiction', '98445c': 'contradiction', '129492c': 'contradiction', '105561e': 'entailment', '28078c': 'contradiction', '15197n': 'neutral', '46063n': 'entailment', '121910e': 'contradiction', '79507e': 'entailment', '9557e': 'entailment', '143751e': 'contradiction', '15727c': 'entailment', '118403e': 'entailment', '106091e': 'entailment', '18892n': 'contradiction', '95883n': 'neutral', '136360n': 'neutral', '59934n': 'neutral', '134356n': 'entailment', '18874n': 'neutral', '117680c': 'entailment', '76947n': 'neutral', '95186n': 'neutral', '14545e': 'entailment', '144753c': 'contradiction', '40710n': 'neutral', '66225n': 'neutral', '91768n': 'neutral', '36469e': 'entailment', '53499c': 'entailment', '101809c': 'neutral', '92771n': 'entailment', '27036c': 'neutral', '26495e': 'entailment', '12562n': 'entailment', '111005e': 'contradiction', '19186c': 'neutral', '125910n': 'neutral', '15100e': 'contradiction', '140005c': 'contradiction', '30221c': 'contradiction', '13387e': 'neutral', '146070e': 'entailment', '125238c': 'contradiction', '118652e': 'neutral', '111167n': 'neutral', '113967e': 'neutral', '47714e': 'entailment', '81812e': 'neutral', '62287n': 'entailment', '47877c': 'contradiction', '83247e': 'entailment', '102923n': 'entailment', '126583n': 'entailment', '58331c': 'contradiction', '102857n': 'entailment', '77175n': 'neutral', '42745e': 'entailment', '46198c': 'contradiction', '66225e': 'entailment', '105790n': 'entailment', '13652n': 'contradiction', '44493n': 'contradiction', '101809n': 'entailment', '102817n': 'neutral', '123038e': 'entailment', '82415n': 'neutral', '73751n': 'neutral', '16989e': 'entailment', '48376n': 'neutral', '27907n': 'entailment', '89209n': 'neutral', '86184c': 'contradiction', '67412n': 'entailment', '103354n': 'entailment', '128542e': 'neutral', '99194e': 'neutral', '48557c': 'contradiction', '44834e': 'entailment', '73191n': 'entailment', '118415n': 'neutral', '13964e': 'neutral', '9393n': 'entailment', '45108n': 'entailment', '13263n': 'contradiction', '86886n': 'contradiction', '1408c': 'neutral', '127490e': 'entailment', '1682n': 'neutral', '80930n': 'neutral', '40738n': 'entailment', '119421n': 'neutral', '75320c': 'neutral', '111011e': 'neutral', '118999n': 'contradiction', '83774c': 'contradiction', '113193n': 'neutral', '53074e': 'entailment', '130928n': 'contradiction', '18189e': 'contradiction', '23774n': 'contradiction', '138500e': 'entailment', '83248c': 'contradiction', '137715c': 'neutral', '46059n': 'entailment', '110061n': 'neutral', '32754e': 'entailment', '140440c': 'contradiction', '36924n': 'neutral', '138359n': 'neutral', '114971n': 'neutral', '54822n': 'neutral', '47877n': 'neutral', '16996e': 'entailment', '109510n': 'neutral', '107302e': 'entailment', '53074n': 'neutral', '9022e': 'entailment', '122531n': 'contradiction', '103616e': 'entailment', '103482c': 'neutral', '65353n': 'contradiction', '117487e': 'entailment', '142298n': 'neutral', '21671e': 'entailment', '8219c': 'contradiction', '53468n': 'entailment', '139409n': 'neutral', '117487c': 'contradiction', '87332n': 'entailment', '76483n': 'entailment', '117093n': 'neutral', '108847e': 'entailment', '35422c': 'contradiction', '14388c': 'contradiction', '141642n': 'neutral', '84055n': 'contradiction', '100349e': 'neutral', '16494c': 'contradiction', '124853e': 'entailment', '143608n': 'neutral', '62982n': 'neutral', '62177n': 'entailment', '31054n': 'entailment', '6105n': 'neutral', '43168c': 'contradiction', '105911c': 'contradiction', '95953c': 'contradiction', '12815n': 'contradiction', '39678e': 'contradiction', '101104n': 'neutral', '114971e': 'entailment', '55110c': 'neutral', '65199e': 'neutral', '15488c': 'neutral', '123675n': 'contradiction', '57151c': 'neutral', '107935n': 'neutral', '100373c': 'contradiction', '80458n': 'neutral', '27907e': 'contradiction', '53545c': 'contradiction', '133966e': 'entailment', '113280n': 'neutral', '118141n': 'neutral', '43094c': 'contradiction', '83657c': 'contradiction', '70711c': 'entailment', '74768e': 'contradiction', '53953n': 'neutral', '128176n': 'neutral', '96516e': 'entailment', '77116e': 'neutral', '79007n': 'entailment', '116111e': 'entailment', '86008n': 'entailment', '50415n': 'contradiction', '65199n': 'neutral', '63013n': 'neutral', '129201n': 'neutral', '86429n': 'neutral', '808n': 'neutral', '138966n': 'neutral', '95338n': 'contradiction', '95663n': 'neutral', '2873n': 'neutral', '43365c': 'contradiction', '59208n': 'neutral', '16086n': 'contradiction', '42321n': 'neutral', '111243n': 'entailment', '18991n': 'neutral', '72389c': 'contradiction', '90839n': 'entailment', '95155e': 'neutral', '71832n': 'neutral', '9047c': 'contradiction', '43225n': 'neutral', '917c': 'contradiction', '101467n': 'neutral', '127809n': 'entailment', '8257n': 'neutral', '36924e': 'neutral', '112592n': 'neutral', '45306n': 'neutral', '58377n': 'neutral', '4795n': 'entailment', '142604e': 'entailment', '94241e': 'entailment', '57163n': 'entailment', '50830n': 'neutral', '142630n': 'contradiction', '96956c': 'contradiction', '83722c': 'contradiction', '117900n': 'neutral', '91797c': 'contradiction', '77175c': 'neutral', '56909n': 'neutral', '53027n': 'contradiction', '113668n': 'neutral', '11534n': 'neutral', '133005c': 'neutral', '83298n': 'neutral', '141321n': 'contradiction', '142964n': 'neutral', '130680n': 'neutral', '82700c': 'contradiction', '6386n': 'neutral', '13221c': 'neutral', '38784n': 'contradiction', '92422e': 'entailment', '110671n': 'neutral', '3799n': 'contradiction', '15537n': 'entailment', '133456n': 'neutral', '97367n': 'neutral', '13760n': 'neutral', '131235n': 'contradiction', '80220n': 'neutral', '61429c': 'neutral', '98621n': 'contradiction', '49462c': 'contradiction', '67063n': 'entailment', '40710c': 'neutral', '70711n': 'neutral', '122197n': 'neutral', '85838e': 'entailment', '51356n': 'contradiction', '14556n': 'entailment', '24043e': 'entailment', '136860c': 'contradiction', '50484c': 'neutral', '42860n': 'entailment', '144173n': 'neutral', '80643e': 'entailment', '137708c': 'entailment', '82182e': 'neutral', '55241e': 'entailment', '104399n': 'neutral', '34920c': 'neutral', '82069e': 'neutral', '121910c': 'entailment', '2262n': 'entailment', '77025c': 'neutral', '119175n': 'neutral', '102817c': 'entailment', '45089c': 'contradiction', '140615e': 'neutral', '125700n': 'neutral', '57345e': 'entailment', '61726n': 'entailment', '78105e': 'entailment', '23725n': 'entailment', '4035c': 'neutral', '121677n': 'entailment', '136752n': 'neutral', '144207n': 'contradiction', '98116c': 'neutral', '67610e': 'neutral', '108243e': 'entailment', '837n': 'entailment', '42388e': 'neutral', '36514n': 'neutral', '56759n': 'neutral', '24103n': 'contradiction', '81469n': 'contradiction', '75838n': 'neutral', '137319n': 'neutral', '127073c': 'neutral', '13531e': 'entailment', '60212c': 'contradiction', '15110n': 'neutral', '33764c': 'neutral', '32851c': 'entailment', '138272n': 'neutral', '73278n': 'contradiction', '69806c': 'neutral', '130869c': 'entailment', '115391n': 'neutral', '102708e': 'neutral', '100637e': 'entailment', '2529c': 'contradiction', '43247n': 'entailment', '145495n': 'entailment', '58557n': 'entailment', '28387c': 'entailment', '17950c': 'contradiction', '5507n': 'contradiction', '3839c': 'contradiction', '47260n': 'neutral', '100248n': 'neutral', '131910n': 'entailment', '119901n': 'neutral', '96539c': 'neutral', '91913n': 'entailment', '107020c': 'entailment', '120896n': 'entailment', '105904c': 'neutral', '8005c': 'contradiction', '31263n': 'entailment', '129081n': 'neutral', '115830n': 'contradiction', '35290n': 'entailment', '130482c': 'contradiction', '14191n': 'entailment', '40867c': 'neutral', '122928e': 'entailment', '68946c': 'neutral', '100792c': 'contradiction', '1735n': 'contradiction', '93839c': 'contradiction', '30894n': 'neutral', '99791c': 'entailment', '102174e': 'entailment', '139028e': 'neutral', '37660c': 'contradiction', '88879n': 'contradiction', '32197n': 'neutral', '91709c': 'neutral', '134514n': 'entailment', '76957c': 'contradiction', '79265n': 'neutral', '117576c': 'neutral', '53211n': 'neutral', '122645n': 'neutral', '133842n': 'entailment', '54811n': 'entailment', '77590c': 'entailment', '17950e': 'entailment', '7091e': 'neutral', '71974n': 'neutral', '40867e': 'neutral', '84901n': 'neutral', '129464n': 'neutral', '119758n': 'neutral', '24103e': 'neutral', '127858c': 'contradiction', '61767e': 'contradiction', '33764n': 'entailment', '126768n': 'entailment', '115593c': 'neutral', '78322e': 'entailment', '31113e': 'contradiction', '49970c': 'contradiction', '102817e': 'entailment', '129081e': 'neutral', '72870n': 'neutral', '46576e': 'neutral', '49227n': 'neutral', '56075n': 'neutral', '30450c': 'contradiction', '73156n': 'contradiction', '142238n': 'neutral', '50480n': 'entailment', '33340e': 'neutral', '44861e': 'entailment', '18086n': 'neutral', '89995e': 'neutral', '77875e': 'neutral', '139677n': 'neutral', '8269e': 'neutral', '43764n': 'contradiction', '67938e': 'contradiction', '23765e': 'entailment', '43178e': 'entailment', '72315c': 'contradiction', '32851e': 'entailment', '7856c': 'contradiction', '77152e': 'entailment', '137715n': 'entailment', '4082e': 'entailment', '94674c': 'contradiction', '118403n': 'neutral', '71957n': 'contradiction'}

modal_adjective = {'47877c': 'contradiction', '46198c': 'contradiction', '9022e': 'entailment', '77025c': 'neutral', '32851c': 'entailment', '117576c': 'neutral', '126768n': 'neutral', '33340e': 'neutral', '32851e': 'entailment'}
sub_expects = {'32889n': 'contradiction', '12562n': 'entailment', '102857n': 'neutral', '66225e': 'entailment', '13652n': 'neutral', '75320c': 'contradiction', '83774c': 'contradiction', '113193n': 'neutral', '138359n': 'neutral', '122531n': 'contradiction', '65353n': 'neutral', '2873n': 'neutral', '9047c': 'contradiction', '42860n': 'entailment', '144173n': 'neutral', '104399n': 'neutral', '60212c': 'contradiction', '115391n': 'neutral', '8005c': 'contradiction', '14191n': 'entailment'}
negated = {'8111n': 'neutral', '135401n': 'neutral', '56634c': 'contradiction', '59934c': 'entailment', '28078c': 'contradiction', '121910e': 'contradiction', '118403e': 'entailment', '106091e': 'entailment', '59934n': 'neutral', '144753c': 'contradiction', '53499c': 'neutral', '26495e': 'entailment', '111005e': 'contradiction', '15100e': 'contradiction', '113967e': 'neutral', '102923n': 'neutral', '44493n': 'contradiction', '27907n': 'entailment', '89209n': 'neutral', '86184c': 'contradiction', '45108n': 'neutral', '86886n': 'contradiction', '40738n': 'neutral', '18189e': 'neutral', '83248c': 'contradiction', '46059n': 'contradiction', '140440c': 'contradiction', '36924n': 'neutral', '114971n': 'neutral', '103482c': 'neutral', '65353n': 'neutral', '87332n': 'neutral', '16494c': 'contradiction', '31054n': 'entailment', '105911c': 'contradiction', '12815n': 'neutral', '114971e': 'entailment', '107935n': 'neutral', '100373c': 'contradiction', '27907e': 'contradiction', '53545c': 'contradiction', '118141n': 'neutral', '83657c': 'neutral', '96516e': 'entailment', '86008n': 'neutral', '138966n': 'contradiction', '16086n': 'contradiction', '42321n': 'neutral', '917c': 'contradiction', '127809n': 'neutral', '36924e': 'neutral', '58377n': 'neutral', '57163n': 'entailment', '83298n': 'neutral', '142964n': 'neutral', '38784n': 'contradiction', '3799n': 'neutral', '70711n': 'neutral', '122197n': 'neutral', '77025c': 'neutral', '61726n': 'neutral', '121677n': 'entailment', '144207n': 'contradiction', '108243e': 'contradiction', '138272n': 'neutral', '73278n': 'contradiction', '69806c': 'neutral', '100637e': 'entailment', '17950c': 'contradiction', '96539c': 'neutral', '31263n': 'entailment', '115830n': 'contradiction', '102174e': 'entailment', '37660c': 'contradiction', '134514n': 'entailment', '76957c': 'contradiction', '117576c': 'neutral', '17950e': 'entailment', '73156n': 'neutral', '44861e': 'entailment', '67938e': 'neutral', '43178e': 'entailment', '32851e': 'entailment', '77152e': 'entailment'}
sub_if = {'135021n': 'entailment', '26766c': 'contradiction', '15197n': 'neutral', '79507e': 'neutral', '118403e': 'entailment', '91768n': 'neutral', '101809c': 'neutral', '27036c': 'neutral', '125910n': 'neutral', '125238c': 'contradiction', '77175n': 'neutral', '101809n': 'entailment', '102817n': 'neutral', '73751n': 'neutral', '113193n': 'neutral', '110061n': 'neutral', '76483n': 'entailment', '117093n': 'neutral', '70711c': 'neutral', '53953n': 'neutral', '86008n': 'neutral', '50415n': 'contradiction', '9047c': 'contradiction', '917c': 'contradiction', '8257n': 'neutral', '58377n': 'neutral', '13221c': 'contradiction', '110671n': 'entailment', '70711n': 'neutral', '51356n': 'contradiction', '137708c': 'contradiction', '34920c': 'entailment', '98116c': 'neutral', '24103n': 'neutral', '137319n': 'neutral', '145495n': 'entailment', '100248n': 'neutral', '100792c': 'contradiction', '139028e': 'neutral', '119758n': 'entailment', '24103e': 'neutral', '78322e': 'entailment', '102817e': 'neutral', '77875e': 'neutral', '7856c': 'contradiction', '118403n': 'contradiction'}
sub_belief = {'117177n': 'neutral', '8111n': 'neutral', '49123n': 'contradiction', '87332c': 'neutral', '106091e': 'entailment', '76947n': 'neutral', '14545e': 'neutral', '144753c': 'contradiction', '53499c': 'neutral', '12562n': 'entailment', '111167n': 'neutral', '81812e': 'neutral', '62287n': 'neutral', '47877c': 'contradiction', '102817n': 'neutral', '118415n': 'neutral', '1682n': 'neutral', '80930n': 'neutral', '118999n': 'neutral', '23774n': 'contradiction', '138500e': 'entailment', '47877n': 'entailment', '109510n': 'neutral', '142298n': 'neutral', '8219c': 'contradiction', '100349e': 'neutral', '83657c': 'neutral', '138966n': 'contradiction', '95155e': 'neutral', '71832n': 'neutral', '45306n': 'neutral', '142630n': 'neutral', '13221c': 'contradiction', '98621n': 'neutral', '42860n': 'entailment', '80643e': 'entailment', '102817c': 'neutral', '45089c': 'contradiction', '140615e': 'neutral', '108243e': 'contradiction', '60212c': 'contradiction', '69806c': 'neutral', '115391n': 'neutral', '100637e': 'entailment', '91913n': 'entailment', '107020c': 'neutral', '8005c': 'contradiction', '129081n': 'neutral', '40867c': 'neutral', '76957c': 'contradiction', '77590c': 'entailment', '71974n': 'neutral', '40867e': 'neutral', '61767e': 'contradiction', '31113e': 'contradiction', '49970c': 'contradiction', '102817e': 'neutral', '129081e': 'neutral', '72870n': 'neutral', '30450c': 'contradiction', '142238n': 'neutral', '44861e': 'entailment', '18086n': 'neutral', '139677n': 'neutral'}
sub_said = {'117177n': 'neutral', '56634c': 'contradiction', '46063n': 'neutral', '9557e': 'entailment', '106091e': 'entailment', '18874n': 'neutral', '111005e': 'contradiction', '19186c': 'contradiction', '140005c': 'contradiction', '13387e': 'neutral', '81812e': 'neutral', '13652n': 'neutral', '123038e': 'entailment', '48376n': 'neutral', '48557c': 'contradiction', '13964e': 'entailment', '45108n': 'neutral', '119421n': 'neutral', '46059n': 'contradiction', '107302e': 'entailment', '103482c': 'neutral', '65353n': 'neutral', '139409n': 'neutral', '124853e': 'neutral', '95953c': 'contradiction', '39678e': 'contradiction', '101104n': 'neutral', '55110c': 'neutral', '80458n': 'neutral', '116111e': 'entailment', '129201n': 'neutral', '95338n': 'contradiction', '43365c': 'contradiction', '95155e': 'neutral', '101467n': 'neutral', '112592n': 'neutral', '4795n': 'entailment', '11534n': 'neutral', '133005c': 'neutral', '130680n': 'neutral', '110671n': 'entailment', '97367n': 'neutral', '61429c': 'contradiction', '85838e': 'contradiction', '136860c': 'contradiction', '50484c': 'neutral', '80643e': 'entailment', '104399n': 'neutral', '4035c': 'neutral', '42388e': 'neutral', '122928e': 'entailment', '30894n': 'neutral', '33764n': 'entailment', '126768n': 'neutral', '31113e': 'contradiction', '49227n': 'neutral', '77875e': 'neutral', '67938e': 'neutral', '137715n': 'entailment'}
is_belief = {'106091e': 'entailment', '136360n': 'neutral', '117680c': 'contradiction', '12562n': 'entailment', '13387e': 'neutral', '118652e': 'neutral', '47877c': 'contradiction', '103354n': 'neutral', '1682n': 'neutral', '87332n': 'neutral', '141642n': 'neutral', '6105n': 'neutral', '43094c': 'neutral', '86008n': 'neutral', '808n': 'entailment', '111243n': 'entailment', '18991n': 'neutral', '71832n': 'neutral', '94241e': 'entailment', '50830n': 'neutral', '38784n': 'contradiction', '13531e': 'neutral', '15110n': 'neutral', '99791c': 'neutral', '88879n': 'contradiction', '91709c': 'neutral', '61767e': 'contradiction'}
is_speech = {'105561e': 'entailment', '121910e': 'contradiction', '143751e': 'neutral', '95186n': 'neutral', '40710n': 'neutral', '111005e': 'contradiction', '13387e': 'neutral', '58331c': 'contradiction', '123038e': 'entailment', '99194e': 'neutral', '48557c': 'contradiction', '1408c': 'neutral', '137715c': 'contradiction', '62982n': 'neutral', '95953c': 'contradiction', '113280n': 'neutral', '118141n': 'neutral', '74768e': 'neutral', '128176n': 'entailment', '96516e': 'entailment', '77116e': 'neutral', '79007n': 'entailment', '90839n': 'neutral', '43225n': 'neutral', '4795n': 'entailment', '83298n': 'neutral', '82700c': 'neutral', '40710c': 'contradiction', '122197n': 'neutral', '14556n': 'neutral', '82069e': 'neutral', '121910c': 'entailment', '78105e': 'neutral', '23725n': 'neutral', '108243e': 'contradiction', '33764c': 'neutral', '5507n': 'contradiction', '3839c': 'contradiction', '47260n': 'neutral', '131910n': 'neutral', '31263n': 'entailment', '130482c': 'contradiction', '53211n': 'neutral', '54811n': 'entailment', '50480n': 'neutral', '43764n': 'neutral', '67938e': 'neutral', '23765e': 'entailment', '94674c': 'contradiction', '71957n': 'neutral'}
has_modal = {'46198n': 'neutral', '86472e': 'neutral', '82161n': 'neutral', '58557e': 'entailment', '98445c': 'contradiction', '129492c': 'neutral', '15197n': 'neutral', '15727c': 'contradiction', '118403e': 'entailment', '18892n': 'contradiction', '95883n': 'neutral', '136360n': 'neutral', '134356n': 'neutral', '76947n': 'neutral', '144753c': 'contradiction', '40710n': 'neutral', '66225n': 'neutral', '36469e': 'entailment', '101809c': 'neutral', '92771n': 'entailment', '26495e': 'entailment', '140005c': 'contradiction', '30221c': 'contradiction', '13387e': 'neutral', '146070e': 'entailment', '111167n': 'neutral', '47714e': 'entailment', '62287n': 'neutral', '83247e': 'entailment', '126583n': 'neutral', '102857n': 'neutral', '77175n': 'neutral', '42745e': 'entailment', '46198c': 'contradiction', '66225e': 'entailment', '105790n': 'entailment', '101809n': 'entailment', '102817n': 'neutral', '123038e': 'entailment', '73751n': 'neutral', '16989e': 'entailment', '48376n': 'neutral', '86184c': 'contradiction', '67412n': 'neutral', '103354n': 'neutral', '128542e': 'neutral', '44834e': 'entailment', '73191n': 'entailment', '9393n': 'entailment', '13263n': 'neutral', '127490e': 'entailment', '1682n': 'neutral', '111011e': 'neutral', '113193n': 'neutral', '53074e': 'entailment', '130928n': 'neutral', '138500e': 'entailment', '110061n': 'neutral', '32754e': 'entailment', '114971n': 'neutral', '54822n': 'contradiction', '16996e': 'entailment', '109510n': 'neutral', '53074n': 'neutral', '9022e': 'entailment', '103616e': 'neutral', '117487e': 'entailment', '21671e': 'entailment', '53468n': 'neutral', '117487c': 'contradiction', '87332n': 'neutral', '76483n': 'entailment', '108847e': 'entailment', '35422c': 'neutral', '84055n': 'neutral', '16494c': 'contradiction', '143608n': 'neutral', '62982n': 'neutral', '62177n': 'entailment', '43168c': 'contradiction', '105911c': 'contradiction', '12815n': 'neutral', '101104n': 'neutral', '114971e': 'entailment', '65199e': 'neutral', '15488c': 'contradiction', '123675n': 'contradiction', '57151c': 'neutral', '100373c': 'contradiction', '27907e': 'contradiction', '113280n': 'neutral', '118141n': 'neutral', '70711c': 'neutral', '53953n': 'neutral', '79007n': 'entailment', '86008n': 'neutral', '65199n': 'neutral', '63013n': 'neutral', '129201n': 'neutral', '86429n': 'neutral', '95338n': 'contradiction', '95663n': 'neutral', '59208n': 'neutral', '16086n': 'contradiction', '42321n': 'neutral', '72389c': 'neutral', '8257n': 'neutral', '142604e': 'entailment', '57163n': 'entailment', '142630n': 'neutral', '96956c': 'contradiction', '83722c': 'contradiction', '117900n': 'neutral', '91797c': 'contradiction', '77175c': 'neutral', '56909n': 'neutral', '53027n': 'contradiction', '113668n': 'neutral', '11534n': 'neutral', '141321n': 'contradiction', '142964n': 'neutral', '6386n': 'neutral', '13221c': 'contradiction', '92422e': 'neutral', '3799n': 'neutral', '15537n': 'neutral', '133456n': 'neutral', '97367n': 'neutral', '13760n': 'neutral', '131235n': 'neutral', '80220n': 'neutral', '61429c': 'contradiction', '49462c': 'contradiction', '67063n': 'entailment', '70711n': 'neutral', '122197n': 'neutral', '136860c': 'contradiction', '137708c': 'contradiction', '82182e': 'neutral', '55241e': 'neutral', '82069e': 'neutral', '2262n': 'neutral', '119175n': 'neutral', '102817c': 'neutral', '125700n': 'neutral', '57345e': 'entailment', '23725n': 'neutral', '136752n': 'neutral', '98116c': 'neutral', '67610e': 'neutral', '108243e': 'contradiction', '837n': 'neutral', '36514n': 'entailment', '56759n': 'neutral', '81469n': 'contradiction', '75838n': 'neutral', '127073c': 'neutral', '138272n': 'neutral', '73278n': 'contradiction', '69806c': 'neutral', '130869c': 'entailment', '102708e': 'neutral', '2529c': 'contradiction', '43247n': 'neutral', '58557n': 'contradiction', '28387c': 'entailment', '47260n': 'neutral', '119901n': 'neutral', '96539c': 'neutral', '120896n': 'entailment', '105904c': 'neutral', '129081n': 'neutral', '35290n': 'neutral', '130482c': 'contradiction', '68946c': 'neutral', '1735n': 'neutral', '93839c': 'neutral', '102174e': 'entailment', '32197n': 'neutral', '134514n': 'entailment', '76957c': 'contradiction', '79265n': 'neutral', '117576c': 'neutral', '133842n': 'neutral', '54811n': 'entailment', '7091e': 'neutral', '84901n': 'neutral', '129464n': 'neutral', '127858c': 'contradiction', '115593c': 'neutral', '102817e': 'neutral', '129081e': 'neutral', '72870n': 'neutral', '46576e': 'contradiction', '56075n': 'neutral', '30450c': 'contradiction', '142238n': 'neutral', '50480n': 'neutral', '89995e': 'neutral', '8269e': 'neutral', '43764n': 'neutral', '43178e': 'entailment', '77152e': 'entailment', '4082e': 'entailment', '118403n': 'contradiction'}

features = {
    'modal_adjective': modal_adjective,
    'sub_if': sub_if,
    'sub_belief': sub_belief,
    'negated': negated,
    'is_belief': is_belief,
    'has_modal': has_modal,
    'is_speech': is_speech,
    'sub_said': sub_said,
    'sub_expects': sub_expects,
}

all = set(all.keys())
class GPTAnalyse():
    def __init__(self):
        self._data_readers = {
            'mnli': MNLIDataReader(),
            'snli': SNLIDataReader(),
            'anli': ANLIDataReader(),
            'mnli_syntax': MNLISyntaxDataReader(),
        } 

        self._chaos_data_readers = {
            'mnli': ChaosMNLIDatareader(),
            'snli': ChaosSNLIDatareader(),
            'anli': ChaosANLIDatareader(),
            'mnli_syntax': ChaosMNLISyntaxDataReader(),
        }

    def load(self):
        k = 0
        self._jl = JadeLogger()
        chaos_data_reader = self._chaos_data_readers['mnli']
        data_reader = self._data_readers['anli']
        data =  data_reader.read_file('test').data()[:]
        fc = FactualityCategorizer()
        cm = ClauseMatcher()
        file2correct = defaultdict(lambda: set())
        gpt_answers = {}
        cache = {}
        files = [
            'llama_3_1.json',
            'llama_3_2.json',
            'llama_3_4.json',
            'llama_3_3.json',
            'llama_3_5.json',
            'llama_3_6.json',
            'llama_3_7.json',
            'llama_3_8.json',
            'llama_3_9.json',
            'llama_3_10.json',
            'llama_3_11.json',
            'llama_3_12.json',
            'llama_3_14.json',
            'llama_3_15.json',




            'llama_3_credence_1.json',
            'llama_3_credence_2.json',
            'llama_3_credence_3.json',
            'llama_3_credence_4.json',
            'llama_3_credence_5.json',
            'llama_3_credence_6.json',
            'llama_3_credence_7.json',
            'llama_3_credence_8.json',
            'llama_3_credence_9.json',
            'llama_3_credence_10.json',
            'llama_3_credence_11.json',
            'llama_3_credence_12.json',
            'llama_3_credence_14.json',
            'llama_3_credence_15.json',
            'llama_3_credence_16.json',





            'llama_3_1_full.json',
            'llama_3_2_full.json',
            'llama_3_3_full.json',
            'llama_3_4_full.json',
            'llama_3_5_full.json',


            'llama_3_credence_1_full.json',
            'llama_3_credence_2_full.json',
            'llama_3_credence_3_full.json',
            'llama_3_credence_4_full.json',
            'llama_3_credence_5_full.json',



            'llama_3_mnli_1_full.json',
            'llama_3_mnli_2_full.json',
            'llama_3_mnli_3_full.json',
            'llama_3_mnli_4_full.json',
            'llama_3_mnli_5_full.json',

            'llama_3_mnli_credence_1_full.json',
            'llama_3_mnli_credence_2_full.json',
            'llama_3_mnli_credence_3_full.json',
            'llama_3_mnli_credence_4_full.json',
            'llama_3_mnli_credence_5_full.json',


            'llama_3_smaller_1.json',
            #'llama_3_smaller_2.json',
            #'llama_3_smaller_3.json',

            "llama_3_smaller_credence_1.json",
            #'llama_3_credence_smaller_1.json',
            #'llama_3_credence_smaller_2.json',
            #'llama_3_credence_smaller_3.json',

            "llama_3_syntax_1_full.json",



            #'gpt_answers_4o_1.json',
            #'gpt_answers_4o_2.json',
            #'gpt_answers_4o_3.json',
            #'gpt_answers_4o_credence_1.json',
            #'gpt_answers_4o_credence_2.json',
            #'gpt_answers_4o_credence_3.json',
            #'gpt_answers_4_credence_3.json',
            #'gpt_answers_4_credence_4.json',
            #'gpt_answers_few_shot_no_explain.json',
            #'gpt_answers_credence_output_2.json',
            #'gpt_answers_credence_output.json',
            #'gpt_answers_no_shot.json',
            #'gpt_answers_few_shot.json',
            #'gpt_answers_few_shot_2.json',
            #'gpt_answers_few_shot_3.json',
            #'gpt_answers_few_shot_4.json',
            #'gpt_answers_with_cot.json'
        ]
        files = [
            'llama_3_mnli_1_full.json',
            'llama_3_mnli_2_full.json',
            'llama_3_mnli_3_full.json',
            'llama_3_mnli_4_full.json',
            'llama_3_mnli_5_full.json',

            'llama_3_mnli_credence_1_full.json',
            'llama_3_mnli_credence_2_full.json',
            'llama_3_mnli_credence_3_full.json',
            'llama_3_mnli_credence_4_full.json',
            'llama_3_mnli_credence_5_full.json',
        ]

        files = [
            'llama_3_anli_1_full.json',
            'llama_3_anli_2_full.json',
            'llama_3_anli_3_full.json',
            'llama_3_anli_4_full.json',
            'llama_3_anli_5_full.json',
            'llama_3_anli_credence_1_full.json',
            'llama_3_anli_credence_2_full.json',
            'llama_3_anli_credence_3_full.json',
            'llama_3_anli_credence_4_full.json',
            'llama_3_anli_credence_5_full.json',

        ]
        uid2data = {}
        for filename in files:
            print(filename)

            all_uids = set()
            true_answers = {}
            location = self._jl.file_manager.data_filepath(filename)

            with open(location, 'rt') as f:
                gpt_answer = json.loads(f.read())
            for feature_name, examples in list(features.items())[:1]:
                for i in [
                    (0, 0.749),
                    (0.749, 0.934),
                    (0.934, 1.058),
                    (1.058, 1.58),
                ][:1]:
                    print(feature_name)
                    counter = 0
                    for d in data:
                        uid2data[d.uid()] = d
                        #dist = d.label_dist()
                        #dist = sorted(dist)
                        #if  not(0.25 < dist[-1] - dist[-2] or dist[-1] - dist[-2] < 0.02):
                        if d.uid() not in cache:
                            event_string, event_string_2 = cm.match(d.sentence_1(), d.sentence_2())
                            features1 = fc.categorize(d.sentence_1(), event_string).to_dict()
                            features2 = fc.categorize(d.sentence_2(), event_string_2).to_dict()
                            if (any (v is True for v in features1.values()) or any (v is True for v in features2.values())):
                                cache[d.uid()] = True
                            else:
                                cache[d.uid()] = False
                        if True or cache[d.uid()] is True:
                            true_answers[d.uid()] = d.label()
                            if d.uid() in gpt_answer:
                                counter += 1
                                if isinstance(gpt_answer[d.uid()], list) and len(gpt_answer[d.uid()][0]) > 0 :
                                    gpt_answer[d.uid()] = gpt_answer[d.uid()][0][0]
                                elif len(gpt_answer[d.uid()]) > 0:
                                    gpt_answer[d.uid()] = gpt_answer[d.uid()][0]
                                if gpt_answer[d.uid()]== d.label()[0]:
                                    file2correct[filename].add(d.uid())
                    gpt_answers[filename] = gpt_answer
                    print(' ' * 4, i, '{:.3f}'.format(self.f1_score(true_answers, gpt_answer)))
                print(counter)
        #self.print_confusion(file2correct, uid2data, gpt_answers)
        #print(all_uids)
        



    def print_confusion(self, file2correct, uid2data, gpt_answers):
        credence_only = file2correct['gpt_answers_credence_output_2.json'] & file2correct['gpt_answers_credence_output.json'] - (file2correct['gpt_answers_few_shot.json'] | file2correct['gpt_answers_few_shot_2.json'])
        non_credence_only = (file2correct['gpt_answers_few_shot.json'] & file2correct['gpt_answers_few_shot_2.json']) - (file2correct['gpt_answers_credence_output_2.json'] | file2correct['gpt_answers_credence_output.json'])
        both_correct = (file2correct['gpt_answers_few_shot.json'] & file2correct['gpt_answers_few_shot_2.json']) & (file2correct['gpt_answers_credence_output_2.json'] & file2correct['gpt_answers_credence_output.json'])
        both_wrong = set(uid2data.keys()) - (file2correct['gpt_answers_few_shot.json'] | file2correct['gpt_answers_few_shot_2.json'] | file2correct['gpt_answers_credence_output_2.json'] | file2correct['gpt_answers_credence_output.json'])
        classes = {
            'credence_only': credence_only,
            'non_credence_only': non_credence_only,
            'both_correct': both_correct,
            'both_wrong': both_wrong
        }
        confusion_data = [['uid', 'class', 'premise', 'hypothesis', 'true_label', 'gpt_answer_few_shot', 'gpt_answer_credence']]
        for key, value in classes.items():
            for uidi, uid in enumerate(value):
                if uidi > min(max(len(credence_only), len(non_credence_only)), len(both_correct), len(both_wrong)):
                    break
                confusion_data.append([
                    uid,
                    key,
                    uid2data[uid].sentence_1(),
                    uid2data[uid].sentence_2(),
                    uid2data[uid].label()[0],
                    gpt_answers['gpt_answers_few_shot.json'].get(uid),
                    gpt_answers['gpt_answers_credence_output_2.json'].get(uid),
                ])
        with open(self._jl.file_manager.data_filepath('gpt_confusion_data.csv'), 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(confusion_data[0])
            for row in sorted(confusion_data[1:], key=lambda x: (x[1], x[4], str([5]), str(x[6]))):
                writer.writerow(row)


    def f1_score(self, true_answers, gpt_answers):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        f1s = []
        for uid, label in true_answers.items():
            if uid not in gpt_answers:
                continue
            if isinstance(gpt_answers[uid], list) and len(gpt_answers[uid][0]) > 0:
                gpt_answer = gpt_answers[uid][0][0]
            elif len(gpt_answers[uid]) > 0:
                gpt_answer = gpt_answers[uid][0]
            if label[0] == gpt_answer:
                tp[gpt_answer] += 1
            else:
                fp[label[0]] += 1
                fn[gpt_answer] += 1
        for key in ['e', 'n', 'c']:
            f1 = 0
            precision = 0
            recall = 0
            if tp[key] + fp[key] != 0:
                precision = tp[key] / (tp[key] + fp[key])
            if tp[key] + fn[key] != 0:
                recall = tp[key] / (tp[key] + fn[key])
            if precision + recall != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            f1s.append(f1)
        return np.mean(f1s)

if __name__ == '__main__':
    Config.instance()
    data_preparer = GPTAnalyse()
    data_preparer.load()