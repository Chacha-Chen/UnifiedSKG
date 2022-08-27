import copy
import json
import os

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer

from utils.processor import get_default_processor

import urllib.request
# from mwzeval.normalization import normalize_data


import re

from functools import partial
from sacremoses import MosesTokenizer, MosesDetokenizer

class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets["validation"], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets["test"], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
datasets.Features(
    {
        "dialogue_id": datasets.Value("string"),
        "db_root_path": datasets.Value("string"),
        "services": datasets.Sequence(datasets.Value("string")),
        "db_paths": datasets.Sequence(datasets.Value("string")),
        "turns": datasets.Sequence(
            {
                "turn_id": datasets.Value("string"),
                "speaker": datasets.ClassLabel(names=["USER", "SYSTEM"]),
                "utterance": datasets.Value("string"),
                "frames": no_use_in_our_setting,
                "dialogue_acts": no_use_in_our_setting ,
            }
        ),
    }
)
"""


def load_db(db_paths: list, return_proportion=True):
    """
    Load dbs in Multi-WoZ dialogue.
    @param db_paths: a list of db_path since each dialogue in Multi-WoZ may have multiple db
    @param return_proportion: whether to return the length proportion of each table in all db_tables
    @return: the db table format like
    """
    # load the db tables from json file
    db_tables = []

    for db_path in db_paths:
        if db_path.endswith("taxi_db.json"):
            continue  # we will skip loading the db of taxi_db since it has some issue.
        with open(os.path.join(db_path)) as f:
            comment_removed_raw_str = "".join(
                [line for line in f.readlines() if not line.startswith("#")]
            )
            # have to do so since json file like hospital_db.json have comment in the front of file.
            db = json.loads(comment_removed_raw_str)
        header = list(db[0].keys())
        rows = []
        for db_item in db:
            row = []
            for column_name, cell_value in db_item.items():
                if isinstance(cell_value, list):
                    row.append(str(tuple(cell_value)))
                elif isinstance(cell_value, dict):
                    row.append(
                        ", ".join(
                            ["{}: {}".format(k, v) for k, v in cell_value.items()]
                        )
                    )
                elif isinstance(cell_value, str):
                    row.append(cell_value)

            rows.append(row)
        db_table = {"header": header, "rows": rows}
        db_tables.append(db_table)

    proportions = []
    for db_table in db_tables:
        table_length = len(db_table["header"])
        for row in db_table["rows"]:
            table_length += len(row)
        proportions.append(table_length)
    total = sum(proportions)
    proportions = [proportion / total for proportion in proportions]
    if return_proportion:
        return db_tables, proportions
    else:
        return db_tables


def get_constructed_history_and_golden_response(usr_utterances, sys_utterances):
    """
    This function construct the reversed order concat of dialogue history from dialogues from users and system.
    as well as the last response(gold response) from user.
    @param usr_utterances:
    @param sys_utterances:
    @return:
    """
    # "[prefix] [utterance n] || [sys_utterance n-1] [utterance n-1] | [sys_utterance n-2] [usr_utterance n-2] | ..."
    assert len(usr_utterances) == len(sys_utterances)

    reversed_utterance_head = [sys_utt.strip() + " | " + usr_utt.strip() for sys_utt, usr_utt in zip(reversed(sys_utterances[:-1]), reversed(usr_utterances[:-1]))]

    reversed_utterance_head_str = " | ".join(reversed_utterance_head)

    return (usr_utterances[-1].strip() + " || " + reversed_utterance_head_str, sys_utterances[-1])

# def get_history(usr_utterances, sys_utterances):
#     """
#     This function construct the reversed order concat of dialogue history from dialogues from users and system.
#     as well as the last response(gold response) from user.
#     @param usr_utterances:
#     @param sys_utterances:
#     @return:
#     """
#     # "[prefix] [utterance n] || [sys_utterance n-1] [utterance n-1] | [sys_utterance n-2] [usr_utterance n-2] | ..."
#     assert len(usr_utterances) == len(sys_utterances)
#
#     reversed_utterance_head = [sys_utt.strip() + " | " + usr_utt.strip() for sys_utt, usr_utt in zip(reversed(sys_utterances[:-1]), reversed(usr_utterances[:-1]))]
#
#     reversed_utterance_head_str = " | ".join(reversed_utterance_head)
#
#     return  reversed_utterance_head_str, sys_utterances[-1]

def delexicalize_utterance(utterance, span_info):
    span_info.sort(key=(lambda x: x[-2]))  # sort spans by start index
    new_utterance = ""
    prev_start = 0
    for span in span_info:
        intent, slot_name, value, start, end = span
        if start < prev_start or value == "dontcare":
            continue
        new_utterance += utterance[prev_start:start]
        new_utterance += f"[{slot_name}]"
        prev_start = end
    new_utterance += utterance[prev_start:]
    return new_utterance

def parse_state(turn):
    state = {}
    for frame in turn["frames"]:
        domain = frame["service"]
        domain_state = {}
        slots = frame["state"]["slots_values"]
        for name, value in zip(slots['slots_values_name'],slots['slots_values_list']):
        # for name, value in slots.items():
            if "dontcare" in value:
                continue
            domain_state[name.split('-')[1]] = value[0]

        if domain_state:
            state[domain] = domain_state

    return state

def normalize_data(extend_data):
    """ In-place normalization of raw dictionary with input data. Normalize slot names, slot values, remove plurals and detokenize utterances. """

    mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
    slot_name_re = re.compile(r'\[([\w\s\d]+)\](es|s|-s|-es|)')
    slot_name_normalizer = partial(slot_name_re.sub, lambda x: normalize_slot_name(x.group(1)))

    # for dialogue in input_data.values():
    extend_data["seq_out"] = slot_name_normalizer(extend_data["seq_out"].lower())
    extend_data["seq_out"] = md.detokenize(mt.tokenize(extend_data["seq_out"].replace('-s', '').replace('-ly', '')))

    # for i in range(len(extend_data['turns']['frames'])):
    #     # if "state"  in turn['frames']:
    #     domains = extend_data['turns']['frames'][i]["service"]
    #     for j in range(len(domains)):
    #         slots = extend_data['turns']['frames'][i]["state"][j]["slots_values"]
    #         new_values = []
    #         new_slots = []
    #         for slot, value in zip(slots['slots_values_name'], slots['slots_values_list']):
    #             slot = slot.lower().replace(' ', '')
    #             if slot == "arriveby":
    #                 slot = "arrive"
    #             elif slot == "leaveat":
    #                 slot = "leave"
    #             new_values.append(normalize_state_slot_value(slot, value))
    #             new_slots.append(slot)
    #         extend_data['turns']['frames'][i]["state"][j]["slots_values"]['slots_values_name'] = new_slots
    #         extend_data['turns']['frames'][i]["state"][j]["slots_values"]['slots_values_list'] = new_values


    return extend_data

def normalize_slot_name(slot_name):
    """ Map a slot name to the new unified ontology. """

    slot_name = slot_name.lower()
    slot_name_mapping = {
     'ADDRESS'   : ['address', 'attraction_address', 'hospital_address', 'hotel_address', 'police_address', 'restaurant_address', 'value_address'],
     'AREA'      : ['area', 'value_area', 'attraction_area', 'restaurant_area', 'hotel_area'],
     'TIME'      : ['booktime', 'value_time', 'time', 'duration', 'value_duration', 'train_duration', 'arriveby', 'taxi_arriveby', 'value_arrive', 'arrive by', 'train_arriveby', 'leaveat', 'value_leave', 'leave at', 'train_leaveat', 'train_leave', 'train_arrive', 'taxi_leaveat'],
     'DAY'       : ['day', 'value_day', 'bookday', 'train_day'],
     'PLACE'     : ['destination', 'value_destination', 'departure', 'value_departure', 'value_place', 'train_departure', 'train_destination', 'taxi_destination', 'taxi_departure'],
     'FOOD'      : ['food', 'value_food', 'restaurant_food'],
     'NAME'      : ['name', 'attraction_name', 'hospital_name', 'hotel_name', 'police_name', 'restaurant_name', 'value_name'],
     'PHONE'     : ['phone', 'attraction_phone', 'hospital_phone', 'hotel_phone', 'police_phone', 'restaurant_phone', 'taxi_phone', 'value_phone'],
     'POST'      : ['postcode', 'attraction_postcode', 'hospital_postcode', 'hotel_postcode', 'restaurant_postcode', 'value_postcode', 'police_postcode'],
     'PRICE'     : ['price', 'value_price', 'entrancefee', 'entrance fee', 'train_price', 'attraction_entrancefee', 'pricerange', 'value_pricerange', 'price range', 'restaurant_pricerange', 'hotel_pricerange', 'attraction_pricerange', 'attraction_price'],
     'REFERENCE' : ['ref', 'attraction_reference', 'hotel_reference', 'restaurant_reference', 'train_reference', 'value_reference', 'reference'],
     'COUNT'     : ['stars', 'value_stars', 'hotel_stars', 'bookstay', 'value_stay', 'stay', 'bookpeople', 'value_people', 'people', 'choice', 'value_choice', 'value_count', 'attraction_choice', 'hotel_choice', 'restaurant_choice', 'train_choice'],
     'TYPE'      : ['type', 'taxi_type', 'taxi_car', 'value_type', 'value_car', 'car', 'restaurant_type', 'hotel_type', 'attraction_type'],
     'TRAINID'   : ['trainid', 'train_id', 'value_id', 'id', 'train', 'train_trainid'],
     'INTERNET'  : ['internet', 'hotel_internet'],
     'PARKING'   : ['parking', 'hotel_parking'],
     'ID'        : ['hospital_id', 'attraction_id', 'restaurant_id'],
     'DEPARTMENT': ['value_department', 'department', 'hospital_department'],
     'OPEN'      : ['openhours']
    }
    reverse_slot_name_mapping = {s : k for k, v in slot_name_mapping.items() for s in v}
    if slot_name not in reverse_slot_name_mapping:
        print(f"Unknown slot name: {slot_name}. Please use another slot names or customize the slot mapping!")
        return ''
    return reverse_slot_name_mapping[slot_name]


def normalize_state_slot_value(slot_name, value):
    """ Normalize slot value:
        1) replace too distant venue names with canonical values
        2) replace too distant food types with canonical values
        3) parse time strings to the HH:MM format
        4) resolve inconsistency between the database entries and parking and internet slots
    """

    def type_to_canonical(type_string):
        if type_string == "swimming pool":
            return "swimmingpool"
        elif type_string == "mutliple sports":
            return "multiple sports"
        elif type_string == "night club":
            return "nightclub"
        elif type_string == "guest house":
            return "guesthouse"
        return type_string

    def name_to_canonical(name, domain=None):
        """ Converts name to another form which is closer to the canonical form used in database. """

        name = name.strip().lower()
        name = name.replace(" & ", " and ")
        name = name.replace("&", " and ")
        name = name.replace(" '", "'")

        if domain is None or domain == "restaurant":
            if name == "hotel du vin bistro":
                return "hotel du vin and bistro"
            elif name == "the river bar and grill":
                return "the river bar steakhouse and grill"
            elif name == "nando's":
                return "nandos"
            elif name == "city center b and b":
                return "city center north b and b"
            elif name == "acorn house":
                return "acorn guest house"
            elif name == "caffee uno":
                return "caffe uno"
            elif name == "cafe uno":
                return "caffe uno"
            elif name == "rosa's":
                return "rosas bed and breakfast"
            elif name == "restaurant called two two":
                return "restaurant two two"
            elif name == "restaurant 2 two":
                return "restaurant two two"
            elif name == "restaurant two 2":
                return "restaurant two two"
            elif name == "restaurant 2 2":
                return "restaurant two two"
            elif name == "restaurant 1 7" or name == "restaurant 17":
                return "restaurant one seven"

        if domain is None or domain == "hotel":
            if name == "lime house":
                return "limehouse"
            elif name == "cityrooms":
                return "cityroomz"
            elif name == "whale of time":
                return "whale of a time"
            elif name == "huntingdon hotel":
                return "huntingdon marriott hotel"
            elif name == "holiday inn exlpress, cambridge":
                return "express by holiday inn cambridge"
            elif name == "university hotel":
                return "university arms hotel"
            elif name == "arbury guesthouse and lodge":
                return "arbury lodge guesthouse"
            elif name == "bridge house":
                return "bridge guest house"
            elif name == "arbury guesthouse":
                return "arbury lodge guesthouse"
            elif name == "nandos in the city centre":
                return "nandos city centre"

        if domain is None or domain == "attraction":
            if name == "broughton gallery":
                return "broughton house gallery"
            elif name == "scudamores punt co":
                return "scudamores punting co"
            elif name == "cambridge botanic gardens":
                return "cambridge university botanic gardens"
            elif name == "the junction":
                return "junction theatre"
            elif name == "trinity street college":
                return "trinity college"
            elif name in ['christ college', 'christs']:
                return "christ's college"
            elif name == "history of science museum":
                return "whipple museum of the history of science"
            elif name == "parkside pools":
                return "parkside swimming pool"
            elif name == "the botanical gardens at cambridge university":
                return "cambridge university botanic gardens"
            elif name == "cafe jello museum":
                return "cafe jello gallery"

        return name

    def time_to_canonical(time):
        """ Converts time to the only format supported by database, e.g. 07:15. """
        time = time.strip().lower()

        if time == "afternoon": return "13:00"
        if time == "lunch" or time == "noon" or time == "mid-day" or time == "around lunch time": return "12:00"
        if time == "morning": return "08:00"
        if time.startswith("one o'clock p.m"): return "13:00"
        if time.startswith("ten o'clock a.m"): return "10:00"
        if time == "seven o'clock tomorrow evening":  return "07:00"
        if time == "three forty five p.m":  return "15:45"
        if time == "one thirty p.m.":  return "13:30"
        if time == "six fourty five":  return "06:45"
        if time == "eight thirty":  return "08:30"

        if time.startswith("by"):
            time = time[3:]

        if time.startswith("after"):
            time = time[5:].strip()

        if time.startswith("afer"):
            time = time[4:].strip()

        if time.endswith("am"):   time = time[:-2].strip()
        if time.endswith("a.m."): time = time[:-4].strip()

        if time.endswith("pm") or time.endswith("p.m."):
            if time.endswith("pm"):   time = time[:-2].strip()
            if time.endswith("p.m."): time = time[:-4].strip()
            tokens = time.split(':')
            if len(tokens) == 2:
                return str(int(tokens[0]) + 12) + ':' + tokens[1]
            if len(tokens) == 1 and tokens[0].isdigit():
                return str(int(tokens[0]) + 12) + ':00'

        if len(time) == 0:
            return "00:00"

        if time[-1] == '.' or time[-1] == ',' or time[-1] == '?':
            time = time[:-1]

        if time.isdigit() and len(time) == 4:
            return time[:2] + ':' + time[2:]

        if time.isdigit(): return time.zfill(2) + ":00"

        if ':' in time:
            time = ''.join(time.split(' '))

        if len(time) == 4 and time[1] == ':':
            tokens = time.split(':')
            return tokens[0].zfill(2) + ':' + tokens[1]

        return time

    def food_to_canonical(food):
        """ Converts food name to caninical form used in database. """

        food = food.strip().lower()

        if food == "eriterean": return "mediterranean"
        if food == "brazilian": return "portuguese"
        if food == "sea food": return "seafood"
        if food == "portugese": return "portuguese"
        if food == "modern american": return "north american"
        if food == "americas": return "north american"
        if food == "intalian": return "italian"
        if food == "italain": return "italian"
        if food == "asian or oriental": return "asian"
        if food == "english": return "british"
        if food == "australasian": return "australian"
        if food == "gastropod": return "gastropub"
        if food == "brutish": return "british"
        if food == "bristish": return "british"
        if food == "europeon": return "european"

        return food

    if slot_name in ["name", "destination", "departure"]:
        return name_to_canonical(value)
    elif slot_name == "type":
        return type_to_canonical(value)
    elif slot_name == "food":
        return food_to_canonical(value)
    elif slot_name in ["arrive", "leave", "arriveby", "leaveat", "time"]:
        return time_to_canonical(value)
    elif slot_name in ["parking", "internet"]:
        return "yes" if value == "free" else value
    else:
        return value

with urllib.request.urlopen("https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/dialog_acts.json") as url:
    print("Downloading MultiWOZ_2.2/dialog_act.json ")
    dialog_acts = json.loads(url.read().decode())


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, "multi_woz_22_train.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            count_processed = 0
            DEBUG_POINT = 5

            # mwz22_data = {}

            for dialog in self.raw_datasets:
                count_processed += 1
                if args.seq2seq.debug:
                    if count_processed > DEBUG_POINT: break
                usr_utterences = [utter for (utter,speaker) in zip(dialog["turns"]["utterance"],dialog["turns"]['speaker']) if speaker == 0]
                sys_utterences = [utter for (utter,speaker) in zip(dialog["turns"]["utterance"],dialog["turns"]['speaker']) if speaker == 1]

                new_turns = [{new_k: new_val[x] for new_k, new_val in dialog['turns'].items()}
                              for x in range(len(dialog['turns']['turn_id']))]
                for i in range(len(new_turns)):
                    turn = new_turns[i]
                    # processing dialogue_acts
                    # new_dialogue_acts = [{new_k: new_val[x] for new_k, new_val in turn['dialogue_acts'].items()}
                    #               for x in range(len(turn['dialogue_acts']['dialog_act']))]
                    # processing frames
                    new_frames = [{new_k: new_val[x] for new_k, new_val in turn['frames'].items()}
                                  for x in range(len(turn['frames']['service']))]
                    new_turns[i]['frames'] = new_frames



                # parsed_turns = []
                for i in range(len(new_turns)):
                    extend_data = copy.deepcopy(dialog)
                    t = new_turns[i]
                    if i % 2 == 0:
                        state = parse_state(t)
                        continue

                    (
                        history,
                        gold_response,
                    ) = get_constructed_history_and_golden_response(
                        usr_utterances = usr_utterences[:int((i+1)/2)],
                        sys_utterances = sys_utterences[:int((i+1)/2)],
                    )
                    # history_without_usr_utterance = get_history(
                    #     usr_utterances = usr_utterences[:int((i+1)/2)],
                    #     sys_utterances = sys_utterences[:int((i+1)/2)],
                    # )
                    extend_data.update(
                        {

                        "struct_in": "",
                        "text_in": history.lower(),
                        # if not args.model.use_context_prompt else usr_utterences[:int((i+1)/2)][-1].strip(),
                        # "history": history_without_usr_utterance.lower(),
                        "seq_out": delexicalize_utterance(t["utterance"],
                                                           dialog_acts[dialog["dialogue_id"]][t["turn_id"]][
                                                               "span_info"]).lower(),

                        "state": state
                    })
                    # extend_data = normalize_data(extend_data)
                    self.extended_data.append(extend_data)
            # normalize_data(self.extended_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TrainOldDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.raw_datasets = raw_datasets
        tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)

        cache_path = os.path.join(cache_root, "multi_woz_22_train.cache")
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            count_processed = 0
            DEBUG_POINT = 5
            for raw_data in self.raw_datasets:
                # Expand the dialogue data
                count_processed += 1
                if args.seq2seq.debug:
                    if count_processed > DEBUG_POINT: break

                usr_utterences = [utter for (utter,speaker) in zip(raw_data["turns"]["utterance_delex"],raw_data["turns"]['speaker']) if speaker == 0]
                sys_utterences = [utter for (utter,speaker) in zip(raw_data["turns"]["utterance_delex"],raw_data["turns"]['speaker']) if speaker == 1]
                assert len(usr_utterences) == len(sys_utterences)

                for i in range(1,len(usr_utterences)+1):
                    extend_data = copy.deepcopy(raw_data)
                    # extend_data["usr"] = [
                    #     utter for (utter,speaker) in zip(raw_data["turns"]["utterance_delex"],raw_data["turns"]['speaker']) if speaker == 0]
                    #
                    # extend_data["sys"] = [
                    #     utter for (utter,speaker) in zip(raw_data["turns"]["utterance_delex"],raw_data["turns"]['speaker']) if speaker == 1]

                    extend_data["usr"] = usr_utterences[:i]
                    extend_data["sys"] = sys_utterences[:i]
                    (
                        history,
                        gold_response,
                    ) = get_constructed_history_and_golden_response(
                        usr_utterances=extend_data["usr"],
                        sys_utterances=extend_data["sys"],
                    )

                    ## Chacha comment all db tables
                    # db_tables, proportions = load_db(raw_data["db_paths"])
                    #
                    # linear_table_s = []
                    #
                    # history_length = len(tokenizer.tokenize(history))
                    # table_truncation_max_length_for_table = (
                    #         args.seq2seq.table_truncation_max_length
                    #         - history_length
                    # )
                    # for table_context, proportion, table_name in zip(db_tables, proportions, raw_data["services"]):
                    #     tab_processor = get_default_processor(
                    #         max_cell_length=200,
                    #         # the max_cell_length is bigger in the MultiWoZ,
                    #         # e.g. you can check "openhours" in "db/attraction_db.json"
                    #         tokenizer=tokenizer,
                    #         max_input_length=int(
                    #             table_truncation_max_length_for_table * proportion + history_length
                    #         ),
                    #         # MARK*: We assign the max length by proportion of each table
                    #     )
                    #
                    #     # modify a table internally
                    #     for truncate_func in tab_processor.table_truncate_funcs:
                    #         truncate_func.truncate_table(table_context, history, [])
                    #     # linearize a table into a string
                    #     linear_table = tab_processor.table_linearize_func.process_table(
                    #         table_context
                    #     )
                    #     linear_table = "{}: {}".format(table_name, linear_table)
                    #     linear_table_s.append(linear_table)
                    #
                    # linear_tables = " || ".join(linear_table_s)

                    extend_data.update(
                        {
                            "struct_in": "",
                            "text_in": history.lower(),
                            "seq_out": gold_response.lower(),
                        }
                    )
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
# class Subclass(GenericClass):
#     '''this is a subclass'''

class DevDataset(TrainDataset):
    '''this is a same traindataset'''
    # def __init__(self, args, raw_datasets, cache_root):
    #     self.raw_datasets = raw_datasets
    #     tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
    #
    #     cache_path = os.path.join(cache_root, "multi_woz_22_dev.cache")
    #     if os.path.exists(cache_path) and args.dataset.use_cache:
    #         self.extended_data = torch.load(cache_path)
    #     else:
    #         self.extended_data = []
    #         for raw_data in self.raw_datasets:
    #             # Expand the dialogue data
    #             for i in range(len(raw_data["turns"])):
    #                 extend_data = copy.deepcopy(raw_data)
    #                 extend_data["usr"] = [
    #                     turn["utterance"]
    #                     for turn in extend_data["turn"]
    #                     if turn["speaker"] == "USER"
    #                 ]
    #                 extend_data["sys"] = [
    #                     turn["utterance"]
    #                     for turn in extend_data["turn"]
    #                     if turn["speaker"] == "SYSTEM"
    #                 ]
    #                 extend_data["usr"] = extend_data["usr"][:i]
    #                 extend_data["sys"] = extend_data["sys"][:i]
    #                 (
    #                     history,
    #                     gold_response,
    #                 ) = get_constructed_history_and_golden_response(
    #                     usr_utterances=extend_data["usr"],
    #                     sys_utterances=extend_data["sys"],
    #                 )
    #
    #                 db_tables, proportions = load_db(raw_data["db_paths"])
    #
    #                 linear_table_s = []
    #
    #                 history_length = len(tokenizer.tokenize(history))
    #                 table_truncation_max_length_for_table = (
    #                         args.seq2seq.table_truncation_max_length
    #                         - history_length
    #                 )
    #                 for table_context, proportion, table_name in zip(db_tables, proportions, raw_data["services"]):
    #                     tab_processor = get_default_processor(
    #                         max_cell_length=200,
    #                         # the max_cell_length is bigger in the MultiWoZ,
    #                         # e.g. you can check "openhours" in "db/attraction_db.json"
    #                         tokenizer=tokenizer,
    #                         max_input_length=int(
    #                             table_truncation_max_length_for_table * proportion + history_length
    #                         ),
    #                         # MARK*: We assign the max length by proportion of each table
    #                     )
    #
    #                     # modify a table internally
    #                     for truncate_func in tab_processor.table_truncate_funcs:
    #                         truncate_func.truncate_table(table_context, history, [])
    #                     # linearize a table into a string
    #                     linear_table = tab_processor.table_linearize_func.process_table(
    #                         table_context
    #                     )
    #                     linear_table = "{}: {}".format(table_name, linear_table)
    #                     linear_table_s.append(linear_table)
    #
    #                 linear_tables = " || ".join(linear_table_s)
    #
    #                 extend_data.update(
    #                     {
    #                         "struct_in": linear_tables.lower(),
    #                         "text_in": history.lower(),
    #                         "seq_out": gold_response.lower(),
    #                     }
    #                 )
    #                 self.extended_data.append(extend_data)
    #         if args.dataset.use_cache:
    #             torch.save(self.extended_data, cache_path)
    #
    # def __getitem__(self, index) -> T_co:
    #     return self.extended_data[index]
    #
    # def __len__(self):
    #     return len(self.extended_data)


class TestDataset(TrainDataset):
    '''this is a same traindataset'''
    # def __init__(self, args, raw_datasets, cache_root):
    #     self.raw_datasets = raw_datasets
    #     tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
    #
    #     cache_path = os.path.join(cache_root, "multi_woz_22_train.cache")
    #     if os.path.exists(cache_path) and args.dataset.use_cache:
    #         self.extended_data = torch.load(cache_path)
    #     else:
    #         self.extended_data = []
    #         dialog_cnt = 0
    #         DEBUG_POINT = 5
    #         for raw_data in self.raw_datasets:
    #             dialog_cnt += 1
    #             if args.seq2seq.debug:
    #                 if dialog_cnt > DEBUG_POINT: break
    #                 # print(dialog_cnt)
    #             # Expand the dialogue data
    #             for i in range(1,len(raw_data["turns"]['utterance'])+1):
    #                 extend_data = copy.deepcopy(raw_data)
    #                 extend_data["usr"] = [
    #                     utter for (utter,speaker) in zip(raw_data["turns"]["utterance"],raw_data["turns"]['speaker']) if speaker == 0]
    #                 #     extend_data["turns"]["utterance"]
    #                 #     for turn in extend_data["turns"]
    #                 #     if extend_data["turns"]["speaker"] == "USER"
    #                 # ]
    #                 extend_data["sys"] = [
    #                     # [
    #                     utter for (utter,speaker) in zip(raw_data["turns"]["utterance"],raw_data["turns"]['speaker']) if speaker == 1]
    #                 #     turn["utterance"]
    #                 #     for turn in extend_data["turn"]
    #                 #     if turn["speaker"] == "SYSTEM"
    #                 # ]
    #                 extend_data["usr"] = extend_data["usr"][:i]
    #                 extend_data["sys"] = extend_data["sys"][:i]
    #                 (
    #                     history,
    #                     gold_response,
    #                 ) = get_constructed_history_and_golden_response(
    #                     usr_utterances=extend_data["usr"],
    #                     sys_utterances=extend_data["sys"],
    #                 )

                    ## Chacha comment all db tables
                    # db_tables, proportions = load_db(raw_data["db_paths"])
                    #
                    # linear_table_s = []
                    #
                    # history_length = len(tokenizer.tokenize(history))
                    # table_truncation_max_length_for_table = (
                    #         args.seq2seq.table_truncation_max_length
                    #         - history_length
                    # )
                    # for table_context, proportion, table_name in zip(db_tables, proportions, raw_data["services"]):
                    #     tab_processor = get_default_processor(
                    #         max_cell_length=200,
                    #         # the max_cell_length is bigger in the MultiWoZ,
                    #         # e.g. you can check "openhours" in "db/attraction_db.json"
                    #         tokenizer=tokenizer,
                    #         max_input_length=int(
                    #             table_truncation_max_length_for_table * proportion + history_length
                    #         ),
                    #         # MARK*: We assign the max length by proportion of each table
                    #     )
                    #
                    #     # modify a table internally
                    #     for truncate_func in tab_processor.table_truncate_funcs:
                    #         truncate_func.truncate_table(table_context, history, [])
                    #     # linearize a table into a string
                    #     linear_table = tab_processor.table_linearize_func.process_table(
                    #         table_context
                    #     )
                    #     linear_table = "{}: {}".format(table_name, linear_table)
                    #     linear_table_s.append(linear_table)
                    #
                    # linear_tables = " || ".join(linear_table_s)

            #         extend_data.update(
            #             {
            #                 "struct_in": "",
            #                 "text_in": history.lower(),
            #                 "seq_out": gold_response.lower(),
            #             }
            #         )
            #         self.extended_data.append(extend_data)
            # if args.dataset.use_cache:
            #     torch.save(self.extended_data, cache_path)
    # def __init__(self, args, raw_datasets, cache_root):
    #     self.raw_datasets = raw_datasets
    #     tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
    #
    #     cache_path = os.path.join(cache_root, "multi_woz_22_test.cache")
    #     if os.path.exists(cache_path) and args.dataset.use_cache:
    #         self.extended_data = torch.load(cache_path)
    #     else:
    #         self.extended_data = []
    #         for raw_data in self.raw_datasets:
    #             # Expand the dialogue data
    #             for i in range(len(raw_data["turns"])):
    #                 extend_data = copy.deepcopy(raw_data)
    #                 extend_data["usr"] = [
    #                     turn["utterance"]
    #                     for turn in extend_data["turn"]
    #                     if turn["speaker"] == "USER"
    #                 ]
    #                 extend_data["sys"] = [
    #                     turn["utterance"]
    #                     for turn in extend_data["turn"]
    #                     if turn["speaker"] == "SYSTEM"
    #                 ]
    #                 extend_data["usr"] = extend_data["usr"][:i]
    #                 extend_data["sys"] = extend_data["sys"][:i]
    #                 (
    #                     history,
    #                     gold_response,
    #                 ) = get_constructed_history_and_golden_response(
    #                     usr_utterances=extend_data["usr"],
    #                     sys_utterances=extend_data["sys"],
    #                 )
    #
    #                 db_tables, proportions = load_db(raw_data["db_paths"])
    #
    #                 linear_table_s = []
    #
    #                 history_length = len(tokenizer.tokenize(history))
    #                 table_truncation_max_length_for_table = (
    #                         args.seq2seq.table_truncation_max_length
    #                         - history_length
    #                 )
    #                 for table_context, proportion, table_name in zip(db_tables, proportions, raw_data["services"]):
    #                     tab_processor = get_default_processor(
    #                         max_cell_length=200,
    #                         # the max_cell_length is bigger in the MultiWoZ,
    #                         # e.g. you can check "openhours" in "db/attraction_db.json"
    #                         tokenizer=tokenizer,
    #                         max_input_length=int(
    #                             table_truncation_max_length_for_table * proportion + history_length
    #                         ),
    #                         # MARK*: We assign the max length by proportion of each table
    #                     )
    #
    #                     # modify a table internally
    #                     for truncate_func in tab_processor.table_truncate_funcs:
    #                         truncate_func.truncate_table(table_context, history, [])
    #                     # linearize a table into a string
    #                     linear_table = tab_processor.table_linearize_func.process_table(
    #                         table_context
    #                     )
    #                     linear_table = "{}: {}".format(table_name, linear_table)
    #                     linear_table_s.append(linear_table)
    #
    #                 linear_tables = " || ".join(linear_table_s)
    #
    #                 extend_data.update(
    #                     {
    #                         "struct_in": linear_tables.lower(),
    #                         "text_in": history.lower(),
    #                         "seq_out": gold_response.lower(),
    #                     }
    #                 )
    #                 self.extended_data.append(extend_data)
    #         if args.dataset.use_cache:
    #             torch.save(self.extended_data, cache_path)
    #
    # def __getitem__(self, index) -> T_co:
    #     return self.extended_data[index]
    #
    # def __len__(self):
    #     return len(self.extended_data)
