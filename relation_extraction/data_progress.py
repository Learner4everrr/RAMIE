import json


def pre_process(INPUT_FILE, OUTPUT_FILE):
    fw=open(OUTPUT_FILE,"w")
    d=[]

    instruction = (f"Please predict relationship between the given head entity and tail entity for a given sentence. "
                    "The relation must be in ('negative', 'not_relate', 'positive')."
                    "Response format example: ['negative']")


    entity_types=[]
    relation_types=[]
    with open(INPUT_FILE,"r",encoding="utf-8") as fr: #renew_triplet_T_train
        for line in fr.readlines():
            line=json.loads(line)

            triples=line["triple"]
            sentence=line["sentence"]
            T=[]
            for triple in triples:
                h=triple["head"]
                h_type=triple["head_type"]
                r = triple["relation"]
                t = triple["tail"]
                t_type=triple["tail_type"]
                # entity_types.append(h_type)
                # entity_types.append(t_type)
                # relation_types.append(r)
                

                Dic_={}
                Dic_["instruction"] =instruction
                Dic_["context"] =sentence + " The relationship between "+ h + " and "+ t +" is?"
                Dic_["response"] =str([str({"relation":r})])

                Dic_["category"] ="relation_extraction"

                fw.write(json.dumps(Dic_))
                fw.write("\n")



pre_process("DS_triple_data_train.json", "DS_train.json")
pre_process("DS_triple_data_test.json", "DS_test.json")