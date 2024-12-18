import json


def pre_process(INPUT_FILE, OUTPUT_FILE):
    fw=open(OUTPUT_FILE,"w")
    d=[]

    # instruction="please extract the triples from the given sentence and recognize the entity type, the relation denotes the relationship between head entity and tail entity, " \
    #             "the relation must in our predefined relation set: ('negative', 'not_relate', 'positive'). the entity type must be in the entity type set: ('folic_acid', 'milk_thistle', 'ginger', 'event', 'chamomile', 'garlic', 'black_cohosh', 'ginkgo', 'lavender', 'melatonin', 'cranberry', 'ginseng', 'glucosamine', 'dandelion', 'saw_palmetto', 'green_tea').response Format: head entity type:head entity|relation|tail entity type: tail entity."

    instruction =("Please predict dietary supplement usage within a given sentence"
                    " this usage which must be in ('continue', 'discontinue', 'uncertain', 'start')")
    print(instruction)

    usage_types=[]
    relation_types=[]
    with open(INPUT_FILE,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            line=json.loads(line)

            label=line["label"]
            ds_type=line["Ds_type"]
            sentence=line["sentence"]

            Dic_={}
            Dic_["instruction"] =instruction
            Dic_["context"] =sentence + "\nWhat is the usage of "+ds_type+"?"
            Dic_["response"] = str([str({"usage":label})])
        
            Dic_["category"] ="usage_extraction"

            fw.write(json.dumps(Dic_))
            fw.write("\n")



pre_process("DS_status_train.json", "DS_train.json")
pre_process("DS_status_test.json", "DS_test.json")

