import json

def preprocess(INTPUT_FILE, OUTPUT_FILE):
    fw=open(OUTPUT_FILE,"w")
    d=[]

    instruction="Please extract the triples from the given sentence and recognize the entity type. " \
                "The relation type must in our predefined relation set: ('negative', 'not_relate', 'positive'). " \
                "The entity type must be in the entity type set: ('event','folic_acid', 'milk_thistle', 'ginger'," \
                " 'chamomile', 'garlic', 'black_cohosh', 'ginkgo', 'lavender', 'melatonin', 'cranberry', 'ginseng'," \
                " 'glucosamine', 'dandelion', 'saw_palmetto', 'green_tea')."

    print(instruction)

    entity_types=[]
    relation_types=[]

    with open(INTPUT_FILE,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            line=json.loads(line)

            triples=line["triple"]
            for triple in triples:
                del triple['head_type']
                del triple['tail_type']
            sentence=line["sentence"]

            Dic_={}
            Dic_["instruction"] = instruction
            Dic_["context"] = sentence
            Dic_["response"] = []
            for triple in triples:
                Dic_["response"].append(str(triple))
            Dic_["response"] = str(Dic_["response"])

            Dic_["category"] ="triple_extraction"

            fw.write(json.dumps(Dic_))
            fw.write("\n")


"""
{'folic_acid', 'milk_thistle', 'ginger', 'event', 'chamomile', 'garlic', 'black_cohosh', 'ginkgo', 'lavender', 'melatonin', 'cranberry', 'ginseng', 'glucosamine', 'dandelion', 'saw_palmetto', 'green_tea'}
{'negative', 'not_relate', 'positive'}
"""



preprocess("DS_triple_data_train.json", "DS_train.json")
preprocess("DS_triple_data_test.json", "DS_test.json")