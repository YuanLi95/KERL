import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import torch.nn.utils.rnn as rnn_utils

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import  deepspeed
from PIL import Image
import math
batch_size = 1

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions,golden_answers, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.golden_answers = golden_answers

    def __getitem__(self, index):
        line = self.questions[index]
        # print(line)
        image_file = line["image"]
        qs = line["text"]
        answer =  self.golden_answers[index]


        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size, answer

    def __len__(self):
        return len(self.questions)


def collate_fn(batch,tokenizer):
    input_ids, image_tensors, image_sizes, answer = zip(*batch)

    input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, answer


# DataLoader
def create_data_loader(questions, golden_answers, image_folder, tokenizer, image_processor, model_config, batch_size=batch_size, num_workers=1):
    dataset = CustomDataset(questions, golden_answers, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer))
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loading model from {model_path}")
    print(f"Model name: {model_name}")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)



    f= open(args.test_file, 'r', encoding='utf-8')
    data = json.load(f)

    questions=[]
    golden_answers=[]

    for item in data:
        id = item['id']
        image_file = item['image']
        conversations = item['conversations']
        questions.append({"id":id, "image":image_file,"text":conversations[0]["value"]})
        golden_answers.append(conversations[1]["value"])

    answers_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions,golden_answers, args.image_folder, tokenizer, image_processor, model.config)

    
    model = model.to(device='cuda:0', non_blocking=True)
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, shape: {param.shape}")

 
    index = 0
    question_line = questions[index:index+batch_size]
    for (input_ids, image_tensor, image_sizes, answer)in tqdm(data_loader):

        input_ids = input_ids.to(device='cuda:0', non_blocking=True)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(device='cuda:0', non_blocking=True)
        # print(input_ids)
        # print(image_tensor)
        with torch.inference_mode():
            output_ids = model.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor.to(dtype=torch.float16, device='cuda:0', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(answer[0])
        print("**" * 20)
        print(outputs[0])
        print("----------------" * 20)
        for i in range(len(outputs)):
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": question_line[i]["id"],
                "prompt": question_line[i]["text"],
                "output": outputs[i].strip(),
                "golden_answers": answer[i],
                "answer_id": ans_id,
                "model_id": model_name
            }))
            ans_file.write("\n")
        index += 1

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../../checkpoints/llava-v1.5-17b-lora_no_knowledge")
    parser.add_argument("--model_base", type=str, default="./vicuna-7b-v1.6")
    parser.add_argument("--image_folder", type=str, default="../../datasets/image10000/")
    parser.add_argument("--output_file", type=str, default="./llava-v1.5-17b-lora_no_knowledge/output.json")
    parser.add_argument("--test_file", type=str, default=".")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    eval_model(args)
