"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

DEP_TO_ID={'det': 0, 'nsubj': 1, 'cop': 2, 'amod': 3, 'prep_of': 4, 'prep_with': 5, 'nn': 6, 'prep_including': 7, 'conj_and': 8, 'nsubjpass': 9, 'partmod': 10, 'agent': 11, 'dobj': 12, 'advmod': 13, 'rel': 14, 'auxpass': 15, 'dep': 16, 'appos': 17, 'prep_in': 18, 'mark': 19, 'aux': 20, 'prep_for': 21, 'num': 22, 'advcl': 23, 'prep_against': 24, 'prep_on': 25, 'prep': 26, 'prep_to': 27, 'prep_than': 28, 'conj_or': 29, 'prep_by': 30, 'xsubj': 31, 'xcomp': 32, 'conj_but': 33, 'tmod': 34, 'prep_at': 35, 'prep_from': 36, 'complm': 37, 'expl': 38, 'ccomp': 39, 'prep_between': 40, 'rcmod': 41, 'conj_negcc': 42, 'prep_upon': 43, 'prt': 44, 'prep_like': 45, 'prep_unlike': 46, 'neg': 47, 'preconj': 48, 'conj_only': 49, 'prepc_in': 50, 'prep_during': 51, 'prep_without': 52, 'parataxis': 53, 'prep_via': 54, 'prep_before': 55, 'prep_despite': 56, 'prep_after': 57, 'prep_versus': 58, 'prep_following': 59, 'iobj': 60, 'prepc_on': 61, 'prep_within': 62, 'poss': 63, 'pobj': 64, 'cc': 65, 'csubj': 66, 'prep_together_with': 67, 'prepc_with': 68, 'prep_through': 69, 'prepc_without': 70, 'prep_compared_with': 71, 'prep_such_as': 72, 'prepc_than': 73, 'purpcl': 74, 'prep_compared_to': 75, 'conj_plus': 76, 'prep_under': 77, 'prep_per': 78, 'conj_+': 79, 'prep_prior_to': 80, 'prepc_for': 81, 'prep_as': 82, 'prep_because_of': 83, 'prep_by_means_of': 84, 'prep_among': 85, 'prep_regardless_of': 86, 'prepc_before': 87, 'prep_since': 88, 'acomp': 89, 'prep_regarding': 90, 'prep_over': 91, 'prep_concerning': 92, 'predet': 93, 'quantmod': 94, 'infmod': 95, 'prep_except_for': 96, 'prep_involving': 97, 'conj_nor': 98, 'prep_due_to': 99, 'measure': 100, 'prep_instead_of': 101, 'prepc_by': 102, 'prep_into': 103, 'abbrev': 104, 'prepc_of': 105, 'prep_except': 106, 'prep_in_addition_to': 107, 'prepc_while': 108, 'conj_and\\/or': 109, 'prep_towards': 110, 'prep_based_on': 111, 'punct': 112, 'attr': 113, 'csubjpass': 114, 'prepc_from': 115, 'prep_about': 116, 'prepc_after': 117, 'prepc_to': 118, 'conj_pain': 119, 'prepc_compared_to': 120, 'prep_in_place_of': 121, 'prep_followed_by': 122, 'pred': 123, 'prepc_such_as': 124, 'prepc_prior_to': 125, 'prep_vs.': 126, 'pcomp': 127, 'prepc_upon': 128, 'prep_in_case_of': 129, 'prep_above': 130, 'conj': 131, 'prepc_beyond': 132, 'prepc_including': 133, 'prepc_in_addition_to': 134, 'prepc_under': 135, 'prepc_except': 136, 'prep_besides': 137, 'prepc_during': 138, 'prepc_against': 139, 'prep_beyond': 140, 'prep_along_with': 141, 'number': 142, 'prep_onto': 143, 'prep_below': 144, 'prep_until': 145, 'conj_versus': 146, 'prep_with_respect_to': 147, 'prepc_at': 148, 'conj_vs': 149, 'prepc_as': 150, 'prepc_due_to': 151, 'prep_subsequent_to': 152, 'prep_de': 153, 'prep_pending': 154}
POS_DDI_TO_ID={PAD_TOKEN: 0, UNK_TOKEN: 1,'DT': 2, 'NN': 3, 'VBP': 4, 'RB': 5, 'VBG': 6, 'IN': 7, 'JJ': 8, 'CC': 9, 'NNS': 10, '.': 11, ',': 12, 'VBN': 13, 'TO': 14, 'WRB': 15, 'VB': 16, '(': 17, ')': 18, 'PRP': 19, 'VBD': 20, 'CD': 21, 'JJS': 22, 'VBZ': 23, 'JJR': 24, 'MD': 25, 'NNP': 26, 'PRP$': 27, 'WDT': 28, 'EX': 29, 'FW': 30, ':': 31, 'RP': 32, 'NNPS': 33, 'RBS': 34, '<': 35, 'WP': 36, 'RBR': 37, 'POS': 38, '"': 39, '``': 40, "''": 41, 'SYM': 42, '>': 43, 'PDT': 44, 'LS': 45, 'WP$': 46}


NEGATIVE_LABEL = 'no_relation'

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
LABEL_CDR_TO_ID = {'Null': 0, 'CID': 1}
POS_CDR_TO_ID={PAD_TOKEN: 0, UNK_TOKEN: 1,'NNP': 2, 'VBZ': 3, 'DT': 4, 'JJ': 5, 'NN': 6, 'IN': 7, '.': 8, ',': 9, 'RB': 10, 'NNS': 11, 'CC': 12, 'VBN': 13, 'CD': 14, 'TO': 15, 'VBD': 16, 'VB': 17, '-LRB-': 18, '-RRB-': 19, ':': 20, 'VBP': 21, 'WDT': 22, 'MD': 23, 'PRP': 24, 'WRB': 25, 'EX': 26, 'RBR': 27, 'VBG': 28, 'POS': 29, 'JJS': 30, 'WP': 31, 'JJR': 32, 'PRP$': 33, 'RP': 34, 'FW': 35, 'NNPS': 36, 'RBS': 37, 'SYM': 38, 'PDT': 39, 'LS': 40, "''": 41, 'WP$': 42}


INFINITY_NUMBER = 1e12
