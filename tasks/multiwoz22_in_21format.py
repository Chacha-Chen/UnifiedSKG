# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MultiWOZ 2.1 for DST, seq2seq methods specially."""

import datasets
import os
from shutil import copyfile
import json
import re
from third_party.zero_shot_dst.T5DST.utils.fix_label import fix_general_label_error

IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']
MAX_LENGTH = 50
_DESCRIPTION = """
delexicalized multiwoz 2.2 in 2.1 format. 
There are 3,406 single-domain dialogues that include booking if the domain allows for that and 7,032 multi-domain dialogues consisting of at least 2 up to 5 domains. To enforce reproducibility of results, the corpus was randomly split into a train, test and development set. The test and development sets contain 1k examples each. Even though all dialogues are coherent, some of them were not finished in terms of task description. Therefore, the validation and test sets only contain fully successful dialogues thus enabling a fair comparison of models. There are no dialogues from hospital and police domains in validation and testing sets.

Each dialogue consists of a goal, multiple user and system utterances as well as a belief state. Additionally, the task description in natural language presented to turkers working from the visitor’s side is added. Dialogues with MUL in the name refers to multi-domain dialogues. Dialogues with SNG refers to single-domain dialogues (but a booking sub-domain is possible). The booking might not have been possible to complete if fail_book option is not empty in goal specifications – turkers did not know about that.

The belief state have three sections: semi, book and booked. Semi refers to slots from a particular domain. Book refers to booking slots for a particular domain and booked is a sub-list of book dictionary with information about the booked entity (once the booking has been made). The goal sometimes was wrongly followed by the turkers which may results in the wrong belief state. The joint accuracy metrics includes ALL slots.
"""
_LICENSE = "Apache License 2.0"

_CITATION = """
[Eric et al. 2019]
@article{eric2019multiwoz,
  title={MultiWOZ 2.1: Multi-Domain Dialogue State Corrections and State Tracking Baselines},
  author={Eric, Mihail and Goel, Rahul and Paul, Shachi and Sethi, Abhishek and Agarwal, Sanchit and Gao, Shuyag and Hakkani-Tur, Dilek},
  journal={arXiv preprint arXiv:1907.01669},
  year={2019}
}
"""

SLOT_DESCRIPTION_PATH = "third_party/zero_shot_dst/T5DST/utils/slot_description.json"


digitpat = re.compile('\d+')
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

fin = open('/home/ubuntu/chacha_code/UnifiedSKG/third_party/zero_shot_dst/T5DST/utils/mapping.pair', 'r')
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def normalize(text, clean_value=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    if clean_value:
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # normalize postcode
        ms = re.findall(
            '([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
            text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    if clean_value:
        # replace time and and price
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [value_price] ', text)
        # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS



def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def get_summary_bstate(bstate, get_domain=False):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']
    summary_bstate = []
    summary_bvalue = []
    active_domain = []
    for domain in domains:
        domain_active = False

        booking = []
        # print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if len(bstate[domain]['book']['booked']) != 0:
                    booking.append(1)
                    # summary_bvalue.append("book {} {}:{}".format(domain, slot, "Yes"))
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                    summary_bvalue.append(["{}-book {}".format(domain, slot.strip().lower()),
                                           normalize(bstate[domain]['book'][slot].strip().lower(),
                                                     False)])  # (["book", domain, slot, bstate[domain]['book'][slot]])
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                slot_enc[1] = 1
                summary_bvalue.append(
                    ["{}-{}".format(domain, slot.strip().lower()), "dontcare"])  # (["semi", domain, slot, "dontcare"])
            elif bstate[domain]['semi'][slot]:
                summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()),
                                       normalize(bstate[domain]['semi'][slot].strip().lower(),
                                                 False)])  # (["semi", domain, slot, bstate[domain]['semi'][slot]])
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
            active_domain.append(domain)
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    if get_domain:
        return active_domain
    else:
        return summary_bstate, summary_bvalue


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        # print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    # last_bvs = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            # print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                # print('not ascii')
                return None
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                # print('not ascii')
                return None
            # belief_summary, belief_value_summary = get_summary_bstate(d['log'][i]['metadata'])
            # d['log'][i]['belief_summary'] = str(belief_summary)
            # d['log'][i]['belief_value_summary'] = belief_value_summary
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    # sys_a = [t['dialogue_acts'] for t in d_orig['sys_log']]
    # bvs = [t['belief_value_summary'] for t in d_orig['sys_log']]
    # domain = [t['domain'] for t in d_orig['usr_log']]
    for item in zip(usr, sys):
        dial.append({'usr': item[0], 'sys': item[1]})
    return dial

def divideData(data, root_path, target_path):
    """Given test and validation sets, divide
    the data for three different sets"""
    root_path = root_path
    print('root path',root_path)
    print('target path', target_path)

    os.makedirs(target_path, exist_ok=True)

    copyfile(os.path.join(root_path, 'ontology.json'), os.path.join(target_path, 'ontology.json'))

    testListFile = []
    fin = open(os.path.join(root_path, 'testListFile.json'), 'r')
    for line in fin:
        testListFile.append(line[:-1])
    fin.close()

    valListFile = []
    fin = open(os.path.join(root_path, 'valListFile.json'), 'r')
    for line in fin:
        valListFile.append(line[:-1])
    fin.close()

    trainListFile = open(os.path.join(target_path, 'trainListFile'), 'w')

    test_dials = []
    val_dials = []
    train_dials = []

    # dictionaries
    # word_freqs_usr = OrderedDict()
    # word_freqs_sys = OrderedDict()

    count_train, count_val, count_test = 0, 0, 0

    ontology = {}

    for dialogue_name in data:
        # print dialogue_name
        dial_item = data[dialogue_name]
        domains = []
        for dom_k, dom_v in dial_item['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL:  # check whether contains some goal entities
                domains.append(dom_k)

        # turn_exmaple = {"system": "none", "user": "none", "state": {"active_intent": "none", "slot_values": {}}}
        dial = get_dial(data[dialogue_name])
        if dial:
            dial_example = {"dial_id": dialogue_name, "domains": list(set(domains)), "turns": []}
            # dialogue = {}
            # dialogue['dialogue_idx'] = dialogue_name
            # dialogue['domains'] = list(set(domains)) #list(set([d['domain'] for d in dial]))
            # last_bs = []
            # dialogue['dialogue'] = []

            for turn_i, turn in enumerate(dial):
                # usr, usr_o, sys, sys_o, sys_a, domain
                turn_exmaple = {"system": "none", "user": "none", "state": {"active_intent": "none", "slot_values": {}}}
                turn_exmaple['system'] = dial[turn_i - 1]['sys'] if turn_i > 0 else "none"
                # turn_exmaple['state']["slot_values"] = {s[0]: s[1] for s in turn['bvs']}
                turn_exmaple['user'] = turn['usr']
                dial_example['turns'].append(turn_exmaple)

                for ss, vv in turn_exmaple['state']["slot_values"].items():
                    if ss not in ontology:
                        ontology[ss] = []
                    if vv not in ontology[ss]:
                        ontology[ss].append(vv)

            if dialogue_name in testListFile:
                test_dials.append(dial_example)
                count_test += 1
            elif dialogue_name in valListFile:
                val_dials.append(dial_example)
                count_val += 1
            else:
                trainListFile.write(dialogue_name + '\n')
                train_dials.append(dial_example)
                count_train += 1

    print("# of dialogues: Train {}, Val {}, Test {}".format(count_train, count_val, count_test))

    # save all dialogues
    print(target_path)
    with open(os.path.join(target_path, "train_dials.json"), 'w') as f:
        json.dump(train_dials, f, indent=4)

    with open(os.path.join(target_path, "dev_dials.json"), 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open(os.path.join(target_path, "test_dials.json"), 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open(os.path.join(target_path, 'ontology.json'), 'w') as f:
        json.dump(ontology, f, indent=4)


class MultiWozV2_2(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2.2.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_label = True  # We default it to fix the label

    def _info(self):
        features = datasets.Features(
            {
                "ID": datasets.Value("string"),
                "turn_id": datasets.Value("int32"),
                "ontology_path": datasets.Value("string"),
                "dialog": {
                    "sys": datasets.Sequence(datasets.Value("string")),
                    "usr": datasets.Sequence(datasets.Value("string"))
                },
                # "domains": datasets.Sequence(datasets.Value("string")),
                "ontology_slots": datasets.Sequence(datasets.Value("string")),
                # "ontology_values": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                "turn_belief": datasets.Sequence(datasets.Value("string")),
                "expanded_turn_belief": datasets.Sequence(
                    {
                        "slot": datasets.Value("string"),
                        "value": datasets.Value("string")
                    })
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            homepage="https://github.com/budzianowski/multiwoz",
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # the code is from https://github.com/facebookresearch/Zero-Shot-DST/blob/main/T5DST/create_data.py
        print('Create WOZ-like dialogues. Get yourself a coffee, this might take a while.')
        # These words is for saluting MultiWOZ, so many error has been made in a time of coffee.
        # root_path = dl_manager.download_and_extract(URL)
        print(os.getcwd())
        root_path = '/home/ubuntu/chacha_code/multiwoz/data/multi-woz'
        fin1 = open(os.path.join(root_path, 'delex.json'), 'r')
        data = json.load(fin1)
        assert len(data) == 10437 #8437+1000+1000
        target_path = os.path.join(root_path, "processed_data")
        if not os.path.exists(target_path):
            # delex_data = createData(root_path)
            print('Divide dialogues...')
            divideData(data, root_path, target_path)

        return [
            datasets.SplitGenerator(
                name=split_enum,
                gen_kwargs={
                    "root_path": target_path,  # the processed woz data as the new root data path
                    "file_path": file_path,
                    "section": section
                },
            )
            for file_path, split_enum, section in [
                (os.path.join(target_path, 'train_dials.json'), datasets.Split.TRAIN, "train"),
                (os.path.join(target_path, 'dev_dials.json'), datasets.Split.VALIDATION, "dev"),
                (os.path.join(target_path, 'test_dials.json'), datasets.Split.TEST, "test"),
            ]
        ]

    def _generate_examples(self, root_path, file_path, section):
        datasets_idx = -1
        ontology_path = os.path.join(root_path, "ontology.json")
        # SLOTS = get_slot_information(json.load(open(ontology_path, 'r')))
        SLOTS = get_slot_information(json.load(open(ontology_path, 'r')))
        # print(SLOTS)
        ontology_slots = []
        ontology_values = []
        description = json.load(open(SLOT_DESCRIPTION_PATH, 'r'))
        # print(description.keys())
        # for slot in SLOTS:
        #     ontology_slots.append(slot)
        #     ontology_values.append(description[slot]["values"])

        domain_counter = {}

        # read files
        with open(file_path) as f:
            dials = json.load(f)
            for dial_dict in dials:
                dialog = {"sys": [], "usr": []}

                # Counting domains
                for domain in dial_dict["domains"]:
                    if domain not in EXPERIMENT_DOMAINS:
                        continue
                    if domain not in domain_counter.keys():
                        domain_counter[domain] = 0
                    domain_counter[domain] += 1

                # Reading data

                for ti, turn in enumerate(dial_dict["turns"]):
                    turn_id = ti

                    # accumulate dialogue utterances
                    dialog["sys"].append(turn["system"])
                    dialog["usr"].append(turn["user"])

                    # slot_values is the dst representation of this turn
                    if self.fix_label:
                        slot_values = fix_general_label_error(turn["state"]["slot_values"], SLOTS)
                    else:
                        slot_values = turn["state"]["slot_values"]

                    # Generate domain-dependent slot list
                    turn_belief_list = [str(k) + '-' + str(v) for k, v in slot_values.items()]

                    # SimpleTOD team(https://github.com/salesforce/simpletod) find that
                    # "For this task, we include none slot values in the sequence. We observed that this will improve SimpleTOD performance on DST by reducing false positive rates."
                    expanded_turn_belief_list = []
                    for slot in SLOTS:
                        expanded_turn_belief_list.append({"slot": str(slot), "value": str(slot_values.get(slot, 'none').strip())})

                    data_detail = {
                        "ID": dial_dict["dial_id"],
                        "turn_id": turn_id,
                        "ontology_path": ontology_path,
                        # "domains": dial_dict["domains"],
                        "dialog": dialog,
                        "ontology_slots": ontology_slots,
                        # "ontology_values": ontology_values,
                        "turn_belief": turn_belief_list,
                        "expanded_turn_belief": expanded_turn_belief_list
                    }
                    datasets_idx += 1
                    yield datasets_idx, data_detail

if __name__ == "__main__":
    os.chdir('/home/ubuntu/chacha_code/UnifiedSKG')
    datasets.load_dataset('/home/ubuntu/chacha_code/UnifiedSKG/tasks/multiwoz22_in_21format.py')