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
"""MultiWOZ v2.2: Multi-domain Wizard of OZ version 2.2"""


import json

import datasets
 

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{corr/abs-2007-12720,
  author    = {Xiaoxue Zang and
               Abhinav Rastogi and
               Srinivas Sunkara and
               Raghav Gupta and
               Jianguo Zhang and
               Jindong Chen},
  title     = {MultiWOZ 2.2 : {A} Dialogue Dataset with Additional Annotation Corrections
               and State Tracking Baselines},
  journal   = {CoRR},
  volume    = {abs/2007.12720},
  year      = {2020},
  url       = {https://arxiv.org/abs/2007.12720},
  archivePrefix = {arXiv},
  eprint    = {2007.12720}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Multi-Domain Wizard-of-Oz dataset (MultiWOZ), a fully-labeled collection of human-human written conversations spanning over multiple domains and topics.
MultiWOZ 2.1 (Eric et al., 2019) identified and fixed many erroneous annotations and user utterances in the original version, resulting in an
improved version of the dataset. MultiWOZ 2.2 is a yet another improved version of this dataset, which identifies and fizes dialogue state annotation errors
across 17.3% of the utterances on top of MultiWOZ 2.1 and redefines the ontology by disallowing vocabularies of slots with a large number of possible values
(e.g., restaurant name, time of booking) and introducing standardized slot span annotations for these slots.
"""

_LICENSE = "Apache License 2.0"

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL_LIST = [
    ("dialogue_acts", "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2/dialog_acts.json")
]
_URL_LIST += [
    (
        f"train_{i:03d}",
        f"https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2/train/dialogues_{i:03d}.json",
    )
    for i in range(1, 18)
]
_URL_LIST += [
    (
        f"dev_{i:03d}",
        f"https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2/dev/dialogues_{i:03d}.json",
    )
    for i in range(1, 3)
]
_URL_LIST += [
    (
        f"test_{i:03d}",
        f"https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2/test/dialogues_{i:03d}.json",
    )
    for i in range(1, 3)
]

_URLs = dict(_URL_LIST)


class MultiWozV22(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("2.2.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="v2.2", version=datasets.Version("2.2.0"), description="MultiWOZ v2.2"),
        datasets.BuilderConfig(
            name="v2.2_active_only",
            version=datasets.Version("2.2.0"),
            description="MultiWOZ v2.2, only keeps around frames with an active intent",
        ),
    ]

    DEFAULT_CONFIG_NAME = "v2.2_active_only"

    def _info(self):
        features = datasets.Features(
            {
                "dialogue_id": datasets.Value("string"),
                "services": datasets.Sequence(datasets.Value("string")),
                "turns": datasets.Sequence(
                    {
                        "turn_id": datasets.Value("string"),
                        "speaker": datasets.ClassLabel(names=["USER", "SYSTEM"]),
                        "utterance": datasets.Value("string"),
                        "frames": datasets.Sequence(
                            {
                                "service": datasets.Value("string"),
                                "state": {
                                    "active_intent": datasets.Value("string"),
                                    "requested_slots": datasets.Sequence(datasets.Value("string")),
                                    "slots_values": datasets.Sequence(
                                        {
                                            "slots_values_name": datasets.Value("string"),
                                            "slots_values_list": datasets.Sequence(datasets.Value("string")),
                                        }
                                    ),
                                },
                                "slots": datasets.Sequence(
                                    {
                                        "slot": datasets.Value("string"),
                                        "value": datasets.Value("string"),
                                        "start": datasets.Value("int32"),
                                        "exclusive_end": datasets.Value("int32"),
                                        "copy_from": datasets.Value("string"),
                                        "copy_from_value": datasets.Sequence(datasets.Value("string")),
                                    }
                                ),
                            }
                        ),
                        "dialogue_acts": datasets.Features(
                            {
                                "dialog_act": datasets.Sequence(
                                    {
                                        "act_type": datasets.Value("string"),
                                        "act_slots": datasets.Sequence(
                                            datasets.Features(
                                                {
                                                    "slot_name": datasets.Value("string"),
                                                    "slot_value": datasets.Value("string"),
                                                }
                                            ),
                                        ),
                                    }
                                ),
                                "span_info": datasets.Sequence(
                                    {
                                        "act_type": datasets.Value("string"),
                                        "act_slot_name": datasets.Value("string"),
                                        "act_slot_value": datasets.Value("string"),
                                        "span_start": datasets.Value("int32"),
                                        "span_end": datasets.Value("int32"),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            homepage="https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2",
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download_and_extract(_URLs)
        self.stored_dialogue_acts = json.load(open(data_files["dialogue_acts"]))
        return [
            datasets.SplitGenerator(
                name=spl_enum,
                gen_kwargs={
                    "filepaths": data_files,
                    "split": spl,
                },
            )
            for spl, spl_enum in [
                ("train", datasets.Split.TRAIN),
                ("dev", datasets.Split.VALIDATION),
                ("test", datasets.Split.TEST),
            ]
        ]

    def _generate_examples(self, filepaths, split):
        id_ = -1
        file_list = [fpath for fname, fpath in filepaths.items() if fname.startswith(split)]
        for filepath in file_list:
            dialogues = json.load(open(filepath))
            for dialogue in dialogues:
                id_ += 1
                mapped_acts = self.stored_dialogue_acts.get(dialogue["dialogue_id"], {})
                res = {
                    "dialogue_id": dialogue["dialogue_id"],
                    "services": dialogue["services"],
                    "turns": [
                        {
                            "turn_id": turn["turn_id"],
                            "speaker": turn["speaker"],
                            "utterance": turn["utterance"],
                            "frames": [
                                {
                                    "service": frame["service"],
                                    "state": {
                                        "active_intent": frame["state"]["active_intent"] if "state" in frame else "",
                                        "requested_slots": frame["state"]["requested_slots"]
                                        if "state" in frame
                                        else [],
                                        "slots_values": {
                                            "slots_values_name": [
                                                sv_name for sv_name, sv_list in frame["state"]["slot_values"].items()
                                            ]
                                            if "state" in frame
                                            else [],
                                            "slots_values_list": [
                                                sv_list for sv_name, sv_list in frame["state"]["slot_values"].items()
                                            ]
                                            if "state" in frame
                                            else [],
                                        },
                                    },
                                    "slots": [
                                        {
                                            "slot": slot["slot"],
                                            "value": "" if "copy_from" in slot else slot["value"],
                                            "start": slot.get("exclusive_end", -1),
                                            "exclusive_end": slot.get("start", -1),
                                            "copy_from": slot.get("copy_from", ""),
                                            "copy_from_value": slot["value"] if "copy_from" in slot else [],
                                        }
                                        for slot in frame["slots"]
                                    ],
                                }
                                for frame in turn["frames"]
                                if (
                                    "active_only" not in self.config.name
                                    or frame.get("state", {}).get("active_intent", "NONE") != "NONE"
                                )
                            ],
                            "dialogue_acts": {
                                "dialog_act": [
                                    {
                                        "act_type": act_type,
                                        "act_slots": {
                                            "slot_name": [sl_name for sl_name, sl_val in dialog_act],
                                            "slot_value": [sl_val for sl_name, sl_val in dialog_act],
                                        },
                                    }
                                    for act_type, dialog_act in mapped_acts.get(turn["turn_id"], {})
                                    .get("dialog_act", {})
                                    .items()
                                ],
                                "span_info": [
                                    {
                                        "act_type": span_info[0],
                                        "act_slot_name": span_info[1],
                                        "act_slot_value": span_info[2],
                                        "span_start": span_info[3],
                                        "span_end": span_info[4],
                                    }
                                    for span_info in mapped_acts.get(turn["turn_id"], {}).get("span_info", [])
                                ],
                            },
                        }
                        for turn in dialogue["turns"]
                    ],
                }
                yield id_, res



# import copy
# import os
#
# import torch
# from datasets import DatasetDict
# from torch.utils.data import Dataset
# from torch.utils.data.dataset import T_co
# from tqdm import tqdm
#
#
# class Constructor(object):
#     def __init__(self, args):
#         self.args = args
#
#     def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
#         if not len(raw_datasets) == 3:
#             raise AssertionError("Train, Dev, Test sections of dataset expected.")
#         train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)
#         dev_dataset = DevDataset(self.args, raw_datasets["validation"], cache_root)
#         test_dataset = TestDataset(self.args, raw_datasets["test"], cache_root)
#
#         return train_dataset, dev_dataset, test_dataset
#
#
# """
# datasets.Features(
#     {
#         "ID": datasets.Value("string"),
#         "turn_id": datasets.Value("int32"),
#         "ontology_path": datasets.Value("string"),
#         "dialog": {
#             "sys": datasets.Sequence(datasets.Value("string")),
#             "usr": datasets.Sequence(datasets.Value("string"))
#         },
#         "domains": datasets.Sequence(datasets.Value("string")),
#         "ontology_slots": datasets.Sequence(datasets.Value("string")),
#         "ontology_values": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
#         "turn_belief": datasets.Sequence(datasets.Value("string")),
#         "expanded_turn_belief": datasets.Sequence(
#                                 {
#                                     "slot": datasets.Value("string"),
#                                     "value": datasets.Value("string")
#                                 })
#     }
# )
# """
#
#
# def get_constructed_history_and_golden_response(usr_utterances, sys_utterances):
#     """
#     This function construct the reversed order concat of dialogue history from dialogues from users and system.
#     as well as the last response(gold response) from user.
#     @param usr_utterances:
#     @param sys_utterances:
#     The whole dialog:
#     for i in range(len(usr_utterances)):
#         print(sys_utterances[i])
#         print(usr_utterances[i])
#     @return:
#     """
#     # "[prefix] [utterance n] || [sys_utterance n-1] [utterance n-1] | [sys_utterance n-2] [usr_utterance n-2] | ..."
#     assert len(usr_utterances) == len(sys_utterances)
#
#     reversed_utterance_head = [
#         usr_utt.strip() + " | " + sys_utt.strip()
#         for usr_utt, sys_utt in zip(
#             reversed(usr_utterances[:-1]), reversed(sys_utterances[:-1])
#         )
#     ]
#
#     reversed_utterance_head_str = usr_utterances[-1] + " || " + sys_utterances[-1] + " | " + " | ".join(
#         reversed_utterance_head)
#     return reversed_utterance_head_str
#
#
# class TrainDataset(Dataset):
#     def __init__(self, args, raw_datasets, cache_root):
#         self.raw_datasets = raw_datasets
#
#         cache_path = os.path.join(cache_root, "multiwoz_train.cache")
#         if os.path.exists(cache_path) and args.dataset.use_cache:
#             self.extended_data = torch.load(cache_path)
#         else:
#             self.extended_data = []
#             for raw_data in self.raw_datasets:
#                 # Expand the dialogue data
#                 for i in range(1,len(raw_data["turns"]['utterance'])+1):
#                     extend_data = copy.deepcopy(raw_data)
#                     extend_data["usr"] = [
#                         utter for (utter,speaker) in zip(raw_data["turns"]["utterance"],raw_data["turns"]['speaker']) if speaker == 0]
#
#                     extend_data["sys"] = [ "none"]
#                     extend_data["sys"].extend(
#                         [utter for (utter,speaker) in zip(raw_data["turns"]["utterance"],raw_data["turns"]['speaker']) if speaker == 1])
#
#                     extend_data["usr"] = extend_data["usr"][:i]
#                     extend_data["sys"] = extend_data["sys"][:i]
#
#                     history = get_constructed_history_and_golden_response(extend_data['usr'],
#                                                                           extend_data['sys'])
#
#
#                     if (not args.seq2seq.mode) or (args.seq2seq.mode == "sequential"):
#                         output_text = ", ".join(["{}-{}".format(slot, value).replace("-", " ") for slot, value in
#                                                  zip(extend_data['expanded_turn_belief']['slot'],
#                                                      extend_data['expanded_turn_belief']['value'])])
#
#
#                     extend_data.update(
#                         {
#                             "struct_in": "",
#                             "text_in": history.lower(),
#                             "seq_out": output_text.lower(),
#                         }
#                     )
#                     self.extended_data.append(extend_data)
#             if args.dataset.use_cache:
#                 torch.save(self.extended_data, cache_path)
#
#     def __getitem__(self, index) -> T_co:
#         return self.extended_data[index]
#
#     def __len__(self):
#         return len(self.extended_data)
#
#
# class DevDataset(Dataset):
#     def __init__(self, args, raw_datasets, cache_root):
#         self.raw_datasets = raw_datasets
#
#         cache_path = os.path.join(cache_root, "multiwoz_dev.cache")
#         if os.path.exists(cache_path) and args.dataset.use_cache:
#             self.extended_data = torch.load(cache_path)
#         else:
#             self.extended_data = []
#             for raw_data in tqdm(self.raw_datasets):
#                 extend_data = copy.deepcopy(raw_data)
#
#                 history = get_constructed_history_and_golden_response(extend_data['dialog']['usr'],
#                                                                       extend_data['dialog']['sys'])
#                 slot_ontology_values_str = ""
#                 for ontology_slot, ontology_values in zip(extend_data["ontology_slots"],
#                                                           extend_data["ontology_values"]):
#                     if not ontology_values:
#                         ontology_item = "{}: {}".format(ontology_slot, "none")
#                     else:
#                         ontology_item = "{}: {}".format(ontology_slot, ", ".join(ontology_values))
#                     slot_ontology_values_str += "{}; ".format(ontology_item)
#
#                 if (not args.seq2seq.mode) or (args.seq2seq.mode == "sequential"):
#                     output_text = ", ".join(["{}-{}".format(slot, value).replace("-", " ") for slot, value in
#                                              zip(extend_data['expanded_turn_belief']['slot'],
#                                                  extend_data['expanded_turn_belief']['value'])])
#
#                     extend_data.update(
#                         {
#                             "struct_in": slot_ontology_values_str.lower(),
#                             "text_in": history.lower(),
#                             "seq_out": output_text.lower(),
#                         }
#                     )
#                     self.extended_data.append(extend_data)
#
#                 elif args.seq2seq.mode == "separate":
#                     for slot, value in zip(extend_data['expanded_turn_belief']['slot'],
#                                            extend_data['expanded_turn_belief']['value']):
#                         # When changing the order of "sk input, question and context", we need to modify here too.
#                         # we admit it was our mistake of design in that part.
#                         slot_history = "{}: {}".format(slot, history)
#                         output_text = value
#
#                         extend_extend_data = copy.deepcopy(extend_data)
#                         del extend_extend_data['expanded_turn_belief']
#                         del extend_extend_data['ontology_slots']
#                         del extend_extend_data['ontology_values']
#
#                         extend_extend_data.update(
#                             {
#                                 "struct_in": slot_ontology_values_str.lower(),
#                                 "text_in": slot_history.lower(),
#                                 "seq_out": output_text.lower(),
#                                 "slot": slot
#                             }
#                         )
#
#                         self.extended_data.append(extend_extend_data)
#
#                 else:
#                     raise ValueError("Other seq2seq method not support yet!")
#
#             if args.dataset.use_cache:
#                 torch.save(self.extended_data, cache_path)
#
#     def __getitem__(self, index) -> T_co:
#         return self.extended_data[index]
#
#     def __len__(self):
#         return len(self.extended_data)
#
#
# class TestDataset(Dataset):
#     def __init__(self, args, raw_datasets, cache_root):
#         self.raw_datasets = raw_datasets
#
#         cache_path = os.path.join(cache_root, "multiwoz_test.cache")
#         if os.path.exists(cache_path) and args.dataset.use_cache:
#             self.extended_data = torch.load(cache_path)
#         else:
#             self.extended_data = []
#             for raw_data in tqdm(self.raw_datasets):
#                 extend_data = copy.deepcopy(raw_data)
#
#                 history = get_constructed_history_and_golden_response(extend_data['dialog']['usr'],
#                                                                       extend_data['dialog']['sys'])
#                 slot_ontology_values_str = ""
#                 for ontology_slot, ontology_values in zip(extend_data["ontology_slots"],
#                                                           extend_data["ontology_values"]):
#                     if not ontology_values:
#                         ontology_item = "{}: {}".format(ontology_slot, "none")
#                     else:
#                         ontology_item = "{}: {}".format(ontology_slot, ", ".join(ontology_values))
#                     slot_ontology_values_str += "{}; ".format(ontology_item)
#
#                 if (not args.seq2seq.mode) or (args.seq2seq.mode == "sequential"):
#                     output_text = ", ".join(["{}-{}".format(slot, value).replace("-", " ") for slot, value in
#                                              zip(extend_data['expanded_turn_belief']['slot'],
#                                                  extend_data['expanded_turn_belief']['value'])])
#
#                     extend_data.update(
#                         {
#                             "struct_in": slot_ontology_values_str.lower(),
#                             "text_in": history.lower(),
#                             "seq_out": output_text.lower(),
#                         }
#                     )
#                     self.extended_data.append(extend_data)
#
#                 elif args.seq2seq.mode == "separate":
#                     for slot, value in zip(extend_data['expanded_turn_belief']['slot'],
#                                            extend_data['expanded_turn_belief']['value']):
#                         # When changing the order of "sk input, question and context", we need to modify here too.
#                         # we admit it was our mistake of design in that part.
#                         slot_history = "{}: {}".format(slot, history)
#                         output_text = value
#
#                         extend_extend_data = copy.deepcopy(extend_data)
#                         del extend_extend_data['expanded_turn_belief']
#                         del extend_extend_data['ontology_slots']
#                         del extend_extend_data['ontology_values']
#
#                         extend_extend_data.update(
#                             {
#                                 "struct_in": slot_ontology_values_str.lower(),
#                                 "text_in": slot_history.lower(),
#                                 "seq_out": output_text.lower(),
#                                 "slot": slot
#                             }
#                         )
#                         self.extended_data.append(extend_extend_data)
#
#                 else:
#                     raise ValueError("Other seq2seq method not support yet!")
#
#             if args.dataset.use_cache:
#                 torch.save(self.extended_data, cache_path)
#
#     def __getitem__(self, index) -> T_co:
#         return self.extended_data[index]
#
#     def __len__(self):
#         return len(self.extended_data)
