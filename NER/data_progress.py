import json


def get_entity_list(line):
    triples=line["triple"]

    entities = set()
    
    for triple in triples:
        h=triple["head"]
        h_type=triple["head_type"]

        r = triple["relation"]

        t = triple["tail"]
        t_type=triple["tail_type"]

        entities.add(str({h:h_type}))
        entities.add(str({t:t_type}))
    
    return list(entities)



def pre_process(INTPUT_FILE, OUTPUT_FILE):

    fw=open(OUTPUT_FILE,"w")
    d=[]

    instruction=("Please extract the dietary supplement from the given sentence and recognize the entity type."
                "the entity type must be in the entity type set: ('event','folic_acid', 'milk_thistle', 'ginger',"
                " 'chamomile', 'garlic', 'black_cohosh', 'ginkgo', 'lavender', 'melatonin', 'cranberry',"
                " 'ginseng', 'glucosamine', 'dandelion', 'saw_palmetto', 'green_tea')."
                "Response Format: [{entity1:entity type1}, {entity type2: tail entity2}, ...]")

    entity_types=[]
    relation_types=[]
    with open(INTPUT_FILE,"r",encoding="utf-8") as fr: #renew_triplet_T_train
        for line in fr.readlines():

            line=json.loads(line)
            response = str(get_entity_list(line))
            sentence=line["sentence"]
            
            Dic_={}
            Dic_["instruction"] =instruction
            Dic_["context"] =sentence
            Dic_["response"] = response

            Dic_["category"] ="NER"

            fw.write(json.dumps(Dic_))
            fw.write("\n")


"""
{'folic_acid', 'milk_thistle', 'ginger', 'event', 'chamomile', 'garlic', 'black_cohosh', 'ginkgo', 'lavender', 'melatonin', 'cranberry', 'ginseng', 'glucosamine', 'dandelion', 'saw_palmetto', 'green_tea'}
{'negative', 'not_relate', 'positive'}
"""


pre_process("DS_triple_data_train.json", "DS_train.json")
pre_process("DS_triple_data_test.json", "DS_test.json")
