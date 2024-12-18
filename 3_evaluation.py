import json
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--generated_file_dir", type=str, default="test_generation.json", help="Test data")
    args = parser.parse_args()
    return args


def calclulate_f1(statics_dict, prefix=""):
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}


def get_result(args):
    state_dict = {"p": 0, "c": 0, "g": 0}

    i=-1
    with open(args.generated_file_dir + "/test_generation.json", "r", encoding="utf-8") as fr:   # relation_test_sample_sht_r_1000.json   relation_test_KNN_5000.json   FB_path_random_1_5000.json
        for line in fr.readlines()[1:]:
            try:
                line=json.loads(line.strip())
                
                pred_triples=set()
                gold_triples=set()

                gold_triples_=line["ground_truth"]
                gold_triples_ = eval(gold_triples_)
                for i in range(len(gold_triples_)):
                    gold_triples_[i] = eval(gold_triples_[i])
                print(gold_triples_)

                predictions=line["predicted"].split("\n\n### Response: \n")[1].split("\n")[0]
                print(predictions)
                predictions = eval(predictions)
                for i in range(len(predictions)):
                    predictions[i] = eval(predictions[i])
                print()
                # print(type(predictions[0]))
            except:
                print("One item failed")
                continue

            for pre in predictions:
                pred_triples.add(str(pre))
            for gold in gold_triples_:
                gold_triples.add(str(gold))
        
            state_dict["p"] += len(pred_triples)
            state_dict["g"] += len(gold_triples)
            state_dict["c"] += len(pred_triples & gold_triples)
    return state_dict
    


# usage_extraction
"""
relation: extraction: {'all-prec': 0.9396551724137931, 'all-recall': 0.9396551724137931, 'all-f1': 0.9396551724137931}
TE: {'all-prec': 0.5616438356164384, 'all-recall': 0.5616438356164384, 'all-f1': 0.5616438356164384}
NER: 0
LP: {'all-prec': 0.771551724137931, 'all-recall': 0.771551724137931, 'all-f1': 0.771551724137931}
Uage: {'all-prec': 0.908695652173913, 'all-recall': 0.908695652173913, 'all-f1': 0.908695652173913}
"""


def main():
    args = get_arguments()

    state_dict = get_result(args)

    all_metirc_results = calclulate_f1(state_dict, 'all')
    print(all_metirc_results)

    with open(args.generated_file_dir+"/metrics.txt", 'a') as fw:
        fw.write(str(all_metirc_results))
        fw.write("\n\n")



if __name__ == "__main__":
    main()