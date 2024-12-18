
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--examplefile', type=str, default='dataset/ade/train.json', help='training file location')
parser.add_argument('--inputfile', type=str, default='dataset/ade/train.json', help='training file location')
parser.add_argument('--outputfile', type=str, default='dataset/ade/train.json', help='training file location')
args = parser.parse_args()


def random(args):
    all_example = []
    with open(args.examplefile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            all_example.append(data)


    all_data = []
    with open(args.inputfile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            random_choice = random.choice(all_example)
            data["example"] = random_choice['context'] + '\nResponse:' + random_choice['response']
            all_data.append(data)


    print(all_data[:2])

    with open(args.outputfile, 'w', encoding='utf-8') as file:
        for data in all_data:
            json.dump(data,file)
            file.write("\n")

# def manual_cosine_similarity(vec1, vec2):
#     dot_product = torch.dot(vec1, vec2)
#     norm_vec1 = torch.norm(vec1)
#     norm_vec2 = torch.norm(vec2)
#     return dot_product / (norm_vec1 * norm_vec2)

def find_most_similar_indices(tensor_a, tensor_b=None):
    """
    找到tensor_a中每个向量的最相似向量的索引。
    
    如果只提供一个张量tensor_a，则在该张量内部进行搜索（忽略自身）。
    如果提供两个张量tensor_a和tensor_b，则在tensor_b中找到与tensor_a中每个向量最相似的向量的索引。
    """
    if tensor_b is None:
        # 如果只提供一个张量，则在自身内部查找最相似的向量（忽略自身）
        tensor_b = tensor_a
        same_tensor = True
    else:
        # 如果提供两个张量，则在tensor_b中查找最相似的向量
        same_tensor = False

    # 扩展张量
    tensor_a_unsqueezed = tensor_a.unsqueeze(1)  # shape: (n, 1, m)
    tensor_b_unsqueezed = tensor_b.unsqueeze(0)  # shape: (1, n, m)

    # 计算余弦相似度矩阵
    cos_sim_matrix = F.cosine_similarity(tensor_a_unsqueezed, tensor_b_unsqueezed, dim=2)

    if same_tensor:
        # 如果在自身内部查找，忽略对角线上的自身相似度
        cos_sim_matrix.fill_diagonal_(-float('inf'))

    # 对每个向量，找到相似度最大的向量的索引
    _, closest_indices = cos_sim_matrix.max(dim=1)

    return closest_indices

def medcpt(args):
    model = AutoModel.from_pretrained("/scratch/ahcie-gpu2/common-models/MedCPT-Query-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("/scratch/ahcie-gpu2/common-models/MedCPT-Query-Encoder")

    all_example = []
    with open(args.examplefile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            example = data['context'] + '\nResponse:' + data['response']
            all_example.append(example)
    encoded = tokenizer(
            all_example, 
            truncation=True, 
            padding=True, 
            return_tensors='pt',
            max_length=512,
        )
    with torch.no_grad():
        example_embeds = model(**encoded).last_hidden_state[:, 0, :]
    
    all_data = []
    input_examples = []
    with open(args.inputfile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            example = data['context'] + '\nResponse:' + data['response']
            input_examples.append(example)
            all_data.append(data)
    
    if "train" in args.outputfile:
        closest_indices = find_most_similar_indices(example_embeds)
    else:
        encoded = tokenizer(
            input_examples, 
            truncation=True, 
            padding=True, 
            return_tensors='pt',
            max_length=512,
        )
        with torch.no_grad():
            input_embeds = model(**encoded).last_hidden_state[:, 0, :]
        closest_indices = find_most_similar_indices(input_embeds, example_embeds)
    

    with open(args.outputfile, 'w', encoding='utf-8') as file:
        for data, index in zip(all_data, closest_indices):
            data["example"] = all_example[index]
            json.dump(data, file)
            file.write("\n")


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def contriever(args):
    model = AutoModel.from_pretrained("/scratch/ahcie-gpu2/common-models/contriever")
    tokenizer = AutoTokenizer.from_pretrained("/scratch/ahcie-gpu2/common-models/contriever")

    all_example = []
    with open(args.examplefile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            example = data['context'] + '\nResponse:' + data['response']
            all_example.append(example)

    print("1"*10)

    encoded = tokenizer(
            all_example, 
            truncation=True, 
            padding=True, 
            return_tensors='pt',
        )
    with torch.no_grad():
        outputs = model(**encoded)
    example_embeds = mean_pooling(outputs[0], encoded['attention_mask'])
    # print(example_embeds)
    
    all_data = []
    input_examples = []
    with open(args.inputfile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            example = data['context'] + '\nResponse:' + data['response']
            input_examples.append(example)
            all_data.append(data)
    print("2"*10)
    if "train" in args.outputfile:
        closest_indices = find_most_similar_indices(example_embeds)
    else:
        encoded = tokenizer(
            input_examples, 
            truncation=True, 
            padding=True, 
            return_tensors='pt',
        )
        with torch.no_grad():
            outputs = model(**encoded)
        input_embeds = mean_pooling(outputs[0], encoded['attention_mask'])
        closest_indices = find_most_similar_indices(input_embeds, example_embeds)
    
    print("3"*10)
    with open(args.outputfile, 'w', encoding='utf-8') as file:
        for data, index in zip(all_data, closest_indices):
            data["example"] = all_example[index]
            json.dump(data, file)
            file.write("\n")




def bmretriever(args):
    model = AutoModel.from_pretrained("/scratch/ahcie-gpu2/common-models/BMRetriever-410M")
    tokenizer = AutoTokenizer.from_pretrained("/scratch/ahcie-gpu2/common-models/BMRetriever-410M")

    all_example = []
    with open(args.examplefile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            example = data['context'] + '\nResponse:' + data['response']
            all_example.append(example)

    print("1"*10)

    encoded = tokenizer(
            all_example, 
            truncation=True, 
            padding=True, 
            return_tensors='pt',
        )
    with torch.no_grad():
        outputs = model(**encoded)
    example_embeds = mean_pooling(outputs[0], encoded['attention_mask'])
    print(example_embeds)
    # exit()
    
    all_data = []
    input_examples = []
    with open(args.inputfile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            example = data['context'] + '\nResponse:' + data['response']
            input_examples.append(example)
            all_data.append(data)
    print("2"*10)
    if "train" in args.outputfile:
        closest_indices = find_most_similar_indices(example_embeds)
    else:
        encoded = tokenizer(
            input_examples, 
            truncation=True, 
            padding=True, 
            return_tensors='pt',
        )
        with torch.no_grad():
            outputs = model(**encoded)
        input_embeds = mean_pooling(outputs[0], encoded['attention_mask'])
        closest_indices = find_most_similar_indices(input_embeds, example_embeds)
    
    print("3"*10)
    with open(args.outputfile, 'w', encoding='utf-8') as file:
        for data, index in zip(all_data, closest_indices):
            data["example"] = all_example[index]
            json.dump(data, file)
            file.write("\n")



def bgelarge(args):
    # model = AutoModel.from_pretrained("/scratch/ahcie-gpu2/common-models/BMRetriever-410M")
    # tokenizer = AutoTokenizer.from_pretrained("/scratch/ahcie-gpu2/common-models/BMRetriever-410M")
    tokenizer = AutoTokenizer.from_pretrained('/scratch/ahcie-gpu2/common-models/bge-large-en-v1.5')
    model = AutoModel.from_pretrained('/scratch/ahcie-gpu2/common-models/bge-large-en-v1.5')


    all_example = []
    with open(args.examplefile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            example = data['context'] + '\nResponse:' + data['response']
            all_example.append(example)

    print("1"*10)

    encoded_input = tokenizer(all_example, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    example_embeds = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    all_data = []
    input_examples = []
    with open(args.inputfile, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            example = data['context'] + '\nResponse:' + data['response']
            input_examples.append(example)
            all_data.append(data)
    print("2"*10)

    if "train" in args.outputfile:
        closest_indices = find_most_similar_indices(example_embeds)
    else:
        encoded_input = tokenizer(input_examples, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        input_embeds = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        closest_indices = find_most_similar_indices(input_embeds, example_embeds)
    
    print("3"*10)
    with open(args.outputfile, 'w', encoding='utf-8') as file:
        for data, index in zip(all_data, closest_indices):
            data["example"] = all_example[index]
            json.dump(data, file)
            file.write("\n")

bgelarge(args)