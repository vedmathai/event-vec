said_verbs = set(["observe", "observes", "observed", "describe", "describes", "described", "discuss", "discusses", "discussed",
					  "report", "reports", "reported", "outline", "outlines", "outlined", "remark", "remarks", "remarked", 	
					  "state", "states", "stated", "go on to say that", "goes on to say that", "went on to say that", 	
					  "quote that", "quotes that", "quoted that", "say", "says", "said", "mention", "mentions", "mentioned",
					  "articulate", "articulates", "articulated", "write", "writes", "wrote", "relate", "relates", "related",
					  "convey", "conveys", "conveyed", "recognise", "recognises", "recognised", "clarify", "clarifies", "clarified",
					  "acknowledge", "acknowledges", "acknowledged", "concede", "concedes", "conceded", "accept", "accepts", "accepted",
					  "refute", "refutes", "refuted", "uncover", "uncovers", "uncovered", "admit", "admits", "admitted",
					  "demonstrate", "demonstrates", "demonstrated", "highlight", "highlights", "highlighted", "illuminate", "illuminates", "illuminated", 							  
                      "support", "supports", "supported", "conclude", "concludes", "concluded", "elucidate", "elucidates", "elucidated",
					  "reveal", "reveals", "revealed", "verify", "verifies", "verified", "argue", "argues", "argued", "reason", "reasons", "reasoned",
					  "maintain", "maintains", "maintained", "contend", "contends", "contended", 
					    "feel", "feels", "felt", "consider", "considers", "considered", 						  
                      "assert", "asserts", "asserted", "dispute", "disputes", "disputed", "advocate", "advocates", "advocated",
					  "opine", "opines", "opined", "think", "thinks", "thought", "imply", "implies", "implied", "posit", "posits", "posited",
					  "show", "shows", "showed", "illustrate", "illustrates", "illustrated", "point out", "points out", "pointed out",
					  "prove", "proves", "proved", "find", "finds", "found", "explain", "explains", "explained", "agree", "agrees", "agreed",
					  "confirm", "confirms", "confirmed", "identify", "identifies", "identified", "evidence", "evidences", "evidenced",
					  "attest", "attests", "attested", "believe", "believes", "believed", "claim", "claims", "claimed", "justify", "justifies", "justified", 							  
                      "insist", "insists", "insisted", "assume", "assumes", "assumed", "allege", "alleges", "alleged", "deny", "denies", "denied",
					   "disregard", "disregards", "disregarded", 
					   "surmise", "surmises", "surmised", "note", "notes", "noted",
					  "suggest", "suggests", "suggested", "challenge", "challenges", "challenged", "critique", "critiques", "critiqued",
					  "emphasise", "emphasises", "emphasised", "declare", "declares", "declared", "indicate", "indicates", "indicated",
					  "comment", "comments", "commented", "uphold", "upholds", "upheld"])

future_said_verbs = set([
    'anticipate', 'anticipates', 'anticipated', "hypothesise", "hypothesises", "hypothesised", "propose", "proposes", "proposed", "theorise", "theorises", "theorised", "posit", "posits", "posited",
    "speculate", "speculates", "speculated", "suppose", "supposes", "supposed", "conjecture", "conjectures", "conjectured", "envisioned", "envision", "envisions", "forecasts", 'foresee', 'forecast', 'forecasted',
    'foresaw', 'estimate', 'estimated', 'estimates'
])

confident_said_verbs = set([
    "observe", "observes", "observed", "describe", "describes", "described", "discuss", "discusses", "discussed",
    "report", "reports", "reported", "outline", "outlines", "outlined", "remark", "remarks", "remarked", 	
    "state", "states", "stated", 	
    "quote", "quotes", "quoted", "say", "says", "said", "mention", "mentions", "mentioned",
    "articulate", "articulates", "articulated", "write", "writes", "wrote", "relate", "relates", 
    "convey", "conveys", "conveyed", "recognise", "recognises", "recognised", "clarify", "clarifies", "clarified",
    "acknowledge", "acknowledges", "acknowledged", "concede", "concedes", "conceded", "accept", "accepts", "accepted",
     "uncover", "uncovers", "uncovered", "admit", "admits", "admitted",
    "demonstrate", "demonstrates", "demonstrated", "highlight", "highlights", "highlighted", "illuminate", "illuminates", "illuminated", 							  
    "support", "supports", "supported", "conclude", "concludes", "concluded", "elucidate", "elucidates", "elucidated",
    "reveal", "reveals", "revealed", "verify", "verifies", "verified", 
    "maintain", "maintains", "maintained", "contend", "contends", "contended", 
    "show", "shows", "showed", "illustrate", "illustrates", "illustrated", "point out", "points out", "pointed out",
    "prove", "proves", "proved", "find", "finds", "found", "explain", "explains", "explained", "agree", "agrees", "agreed",
    "confirm", "confirms", "confirmed", "identify", "identifies", "identified", "evidence", "evidences", "evidenced",
    "attest", "attests", "attested", 							  
     "note", "notes", "noted",
    "emphasise", "emphasises", "emphasised", "declare", "declares", "declared", "indicate", "indicates", "indicated",
    "uphold", "upholds", "upheld"
 ])

believe_verbs = set([
    "argue", "argues", "argued", "reason", "reasons", "reasoned",  "feel", "feels", "felt", "refute", "refutes", "refuted",  "consider", "considers", "considered", 	
    "dispute", "disputes", "disputed", "advocate", "advocates", "advocated", "insist", "insists", "insisted", "assume", "assumes", "assumed", "allege", "alleges", "alleged", "deny", "denies", "denied",
    "disregard", "disregards", "disregarded", "challenge", "challenges", "challenged", "critique", "critiques", "critiqued", "comment", "comments", "commented",
    "surmise", "surmises", "surmised", "suggest", "suggests", "suggested", 
    "assert", "asserts", "asserted", "believe", "believes", "believed", "claim", "claims", "claimed", "justify", "justifies", "justified", 
    "opine", "opines", "opined", "think", "thinks", "thought", "imply", "implies", "implied", "posit", "posits", "posited",
    'anticipate', 'anticipates', 'anticipated', "hypothesise", "hypothesises", "hypothesised", "propose", "proposes", "proposed", "theorise", "theorises", "theorised", "posit", "posits", "posited",
    "speculate", "speculates", "speculated", "suppose", "supposes", "supposed", "conjecture", "conjectures", "conjectured", "envisioned", "envision", "envisions", "forecasts", 'foresee', 'forecast', 'forecasted',
    'foresaw', 'estimate', 'estimated', 'estimates'
])

expect_neg = set([
    "fail", "fails", "failed", "failing",
    "struggle", "struggles", "struggled", "struggling",
    "neglect", "neglects", "neglected", "neglecting",
    "refuse", "refuses", "refused", "refusing",
    "attempt", "attempts", "attempted", "attempting",
    "decline", "declines", "declined", "declining",
    "prohibit", "prohibits", "prohibited", "prohibiting",
    "hinder", "hinders", "hindered", "hindering",
    "forbid", "forbids", "forbade", "forbidden", "forbidding",
    "discourage", "discourages", "discouraged", "discouraging",
    "deny", "denies", "denied", "denying",
    "forget", "forgets", "forgot", "forgetting",
])

expect_verbs = set([
    'expect', 'expects', 'expected', 'expecting', 'hope', 'hoped', 'hopes', 'hoping', 'looks', 'looked', 'look', 'looking', 'due',
    'try', 'tried', 'tries', 'trying', 'threatening', 'threatened',
    "plan", "plans", "planned", "planning",
    "intend", "intends", "intended", "intending",
    "promise", "promises", "promised", "promising",
    "manage", "manages", "managed", "managing",
    "agree", "agrees", "agreed", "agreeing",
    "offer", "offers", "offered", "offering",
    "prepare", "prepares", "prepared", "preparing",
    "arrange", "arranges", "arranged", "arranging",
    "desire", "desires", "desired", "desiring",
    "aim", "aims", "aimed", "aiming",
    "pretend", "pretends", "pretended", "pretending",
    "threaten", "threatens", "threatening",
    "claim", "claims", "claimed", "claiming",
    "assure", "assures", "assured", "assuring",
    "guarantee", "guarantees", "guaranteed", "guaranteeing",
    "advise", "advises", "advised", "advising",
    "warn", "warns", "warned", "warning",
    "encourage", "encourages", "encouraged", "encouraging",
    "instruct", "instructs", "instructed", "instructing",
    "permit", "permits", "permitted", "permitting",
    "oblige", "obliges", "obliged", "obliging",
    "urge", "urges", "urged", "urging",
    "enable", "enables", "enabled", "enabling",
    "allow", "allows", "allowed", "allowing",
    "compel", "compels", "compelled", "compelling",
    "convince", "convinces", "convinced", "convincing",
    "persuade", "persuades", "persuaded", "persuading",
    "require", "requires", "required", "requiring",

])

past_perf_aux = [
    'had',
]

pres_perf_aux = [
    'has',
    'have',
]

future_modals = [
    'will',
    'going to',
    'would',
    'could',
    'might',
    'may',
    'can',
    'going to',
    'to be',
    'shall',
    'should',
    'ca',
    'wo',
]

negation_words = [
    'none', 'unsuccesful', 'unknown', 'unlike', 'not', 'never', 'lack', 'except', 'rarely',
    'without', 'prevent', 'nobody', 'cannot', 'unable', 'no', 'exception', 'refused',
    'illegal', 'unusual', 'absense', 'no longer', 'oppose', 'nothing', 'could not',
    'unpopular', 'unclear', 'unformal', 'neither', 'uncommon', 'dislike', 'unofficial',
    'stop', 'avoid', 'unarmed', 'absent'
]

modal_adverbs = list(set([
    'probably', 'possibly', 'clearly', 'obviously', 'presumably', 'evidently', 'apparently', 'supposedly',
    'conceivably', 'undoubtedly', 'allegedly', 'reportedly', 'arguably', 'unquestionably', 'seemingly',
    'certainly', 'definitely', 'surely', 'undoubtedly', 'indeed', 'truly', 'honestly', 'frankly',
    'honestly', 'truthfully', 'sincerely', 'genuinely', 'actually', 'really', 'virtually',
    'practically', 'effectively', 'essentially', 'fundamentally',
    'roughly', 'nearly', 'virtually', 'basically',
    'approximately', 'roughly', 'almost', 'undoubtedly', 'doubtfully', 'doubtlessly', 'dubiously',
]))

modal_adjectives = list(set([
    'possible', 'impossible', 'probable', 'improbable', 'likely', 'unlikely', 'certain', 'uncertain',
    'clear', 'unclear', 'obvious', 'presumed', 'evident', 'apparent', 'supposed', 'conceivable', 'unconceivable',
    'unquestionable', 'questionable', 
    'arguable', 'seeming', 'reported', 'alleged', 'undoubted', 'doubtful', 'doubtless', 'dubious', 'indubitable', 'indubious',
    'indefinite', 'untrue', 'false', 'impractical', 'ineffective', 'unessential', 'fundamental',
    'definite', 'sure', 'undoubted', 'indeed', 'true', 'honest', 'frank', 'truthful',
    'sincere', 'genuine', 'actual', 'real', 'virtual', 'practical', 'effective', 'essential',
    'fundamental', 'rough', 'nearly', 'virtual', 'basic', 'approximate', 'rough', 'almost',
]))

contrasting_conjunctions = {
    'but', 'however', 'though', 'all the same', 'be that as it may', 'despite', 'even so', 'in spite of', 'nevertheless', 'regardless', 'yet', 'on the other hand', 'instead', 'rather', 'in fact', 'in reality', 'in contrast', 'in comparison', 'on the contrary', 'conversly'
}