import argparse
import json
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import os
import torch
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vqa_score")
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    stop_words = stopwords.words('english')
    import spacy
    lemmatizer = spacy.load("en_core_web_sm")
    vqa_file = open(args.gt_path, 'r')
    vqa_json = json.load(vqa_file)
    avg_acc = []
    dir_list = os.listdir(args.result_dir)
    for result_file in dir_list:
        prediction_file = open(f"{args.result_dir}/{result_file}", 'r')
        output_file = open(f"{args.result_dir}/convert_{result_file}", 'w')
        predictions = []
        predictions_nll = []
        gts = []
        acc = []
        output = []
        predictions_confidence = []
        prediction_lines = prediction_file.readlines()
        for i in range(len(prediction_lines)):
            prediction = json.loads(prediction_lines[i])
            try:
                answers = prediction['cot']#[0]
            except:
                import ipdb
                ipdb.set_trace()
            ensumble_prediction = []
            for answer in answers:
                answer = answer.split()
                if len(answer) > 1:
                    answer = [ele for ele in answer if ele not in stop_words]
                try:
                    answer = ' '.join(e for e in answer)        # if e.isalnum()
                    
                    doc = lemmatizer(answer)
                    words = []
                    for token in doc:
                        if token.pos_ in ["NOUN", "VERB"]:
                            words.append(token.lemma_)
                        else:
                            words.append(token.text)
                    answer = " ".join(words)
                    
                    ensumble_prediction.append(answer.lower())      # 
                except:
                    import ipdb
                    ipdb.set_trace()
                    ensumble_prediction.append('unknown')

            pred = np.array(ensumble_prediction)
            if args.mode=='confidence':
                confidence = prediction['verify']
            predictions.append(ensumble_prediction)
            predictions_nll.append(prediction['nll'])
            if args.mode == 'confidence':
                predictions_confidence.append(prediction['verify'])
                for j in range(len(confidence)):
                    if confidence[j] == 'A':
                        confidence[j] = 0
                    elif confidence[j] == 'B':
                        confidence[j] = 0.25
                    elif confidence[j] == 'C':
                        confidence[j] = 0.5
                    elif confidence[j] == 'D':
                        confidence[j] = 0.75
                confidence = np.array(confidence)
                pred_set = set(pred)
                vote = dict()
                
                for ele in pred_set:
                    vote.update({ele: np.sum(confidence[pred==ele]).item()})
                pred = sorted(vote.items(), key=lambda x:(-x[1],x[0]))[0][0]
            elif args.mode == 'vote':
                pred = predictions[i]
                pred = max(pred, key=pred.count)        # 选出出现次数最多的元素
            # elif args.mode == 'nll':
            #     pred = np.array(predictions[i])
            #     nlls = torch.tensor(predictions_nll[i]).squeeze(-1)
            #     pred_set = set(pred)
            #     vote = dict()
            #     for ele in pred_set:
            #         vote.update({ele: torch.mean(nlls[pred==ele]).item()})
            #     pred = sorted(vote.items(), key=lambda x:x[1])[0][0]
            #     # min_index = torch.argmin(nlls)
            #     # pred = predictions[i][min_index.item()]

            # elif args.mode == 'nll':
                pred = np.array(predictions[i])
                nlls = torch.tensor(predictions_nll[i]).squeeze(-1)
                choose_index = nlls<0.6
                if torch.sum(choose_index) == 0:
                    pred = max(predictions[i], key=predictions[i].count)
                else:
                    pred = pred[choose_index].tolist()
                    pred = max(pred, key=pred.count)

            elif args.mode == 'nll':
                pred = np.array(predictions[i])
                nlls = torch.tensor(predictions_nll[i]).squeeze(-1)
                pred_set = set(pred)
                vote = dict()
                min_indexes = torch.sort(nlls).indices[:4]
                pred = pred[min_indexes].tolist()
                pred = max(pred, key=pred.count)

            if vqa_json[i]['image'] == prediction['image']:
                output.append({'question_id': vqa_json[i]['question_id'], 'answer':pred})
            else:
                print("error!")
            
        output_file.write(json.dumps(output))
        output_file.close()

        vqa = VQA('cache/okvqa/annotations/mscoco_val2014_annotations.json', 'cache/okvqa/annotations/OpenEnded_mscoco_val2014_questions.json')
        
        vqa_result = vqa.loadRes(
            resFile=f"{args.result_dir}/convert_{result_file}", quesFile='cache/okvqa/annotations/OpenEnded_mscoco_val2014_questions.json'
        )
        # create vqaEval object by taking vqa and vqaRes
        # n is precision of accuracy (number of places after decimal), default is 2
        vqa_scorer = VQAEval(vqa, vqa_result, n=2)
        print("Start VQA evaluation.")
        vqa_scorer.evaluate()
        # print accuracies
        overall_acc = vqa_scorer.accuracy["overall"]
        avg_acc.append(overall_acc)
    print(avg_acc)
    print(np.mean(np.array(avg_acc)), np.std(np.array(avg_acc)))

# 1. 对每个问题，先投票选出一个答案
# 2. 把数据转成question-id，answer的格式，并存入文件
# 3. 调用vqa tool，计算score
# input_file = open('{okvqa}_cot_dir/xxl_okvqa_subset_rl_20221227141/1.json', 'r')
# vqa_file = open('cache/okvqa/annotations/vqa_val_eval.json', 'r')
# output_file = open('{okvqa}_cot_dir/xxl_okvqa_subset_rl_20221227141/1_convert.json', 'w')
# input_lines = input_file.readlines()
# vqa_lines = json.load(vqa_file)
# output = []
# for i in range(len(vqa_lines)):
#     result = json.loads(input_lines[i])
#     vqa = vqa_lines[i]
#     if result['image'] == vqa['image']:
#         output.append({'question_id': vqa['question_id'], 'answer': result['cot']})
# output_file.write(json.dumps(output))
# input_file.close()
# output_file.close()