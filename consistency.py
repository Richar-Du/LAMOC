import argparse
import json
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import os
import torch
import operator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vqa_score")
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    stop_words = stopwords.words('english')
    import spacy
    lemmatizer = spacy.load("en_core_web_sm")
    gt_file = open(args.gt_path, 'r')
    vqa_json = json.load(gt_file)
    avg_acc = []
    for result_file in os.listdir(args.result_dir):
        prediction_file = open(f"{args.result_dir}/{result_file}", 'r')
        predictions = []
        predictions_nll = []
        gts = []
        acc = []
        predictions_confidence = []
        for line in prediction_file.readlines():
            vqa = json.loads(line)
            answers = vqa['cot']#[0]
            ensumble_prediction = []
            for answer in answers:
                answer = answer.split()
                if len(answer) > 1:
                    answer = [ele for ele in answer if ele not in stop_words]
                try:
                    answer = ' '.join(e for e in answer if e.isalnum())
                    # doc = lemmatizer(answer)
                    # words = []
                    # for token in doc:
                    #     if token.pos_ in ["NOUN", "VERB"]:
                    #         words.append(token.lemma_)
                    #     else:
                    #         words.append(token.text)
                    # answer = " ".join(words)
                    ensumble_prediction.append(answer.lower())
                except:
                    ensumble_prediction.append('unknown')

            predictions.append(ensumble_prediction)
            predictions_nll.append(vqa['nll'])
            if args.mode == 'confidence':
                predictions_confidence.append(vqa['verify'])
        
        for vqa in vqa_json:
            if 'okvqa' in args.gt_path.split('/'):
                gts.append(vqa['answer'])
            elif 'aokvqa' in args.gt_path.split('/'):
                gts.append(vqa['direct_answers'])

        # for i in range(len(predictions)):
        #     pred = predictions[i]
        #     pred = np.array(predictions[i])
        #     # nlls = torch.tensor(predictions_nll[i]).squeeze(-1)
        #     # max_index = torch.sort(nlls).indices.tolist()#[-7:]
        #     # pred = np.array(pred)[max_index].tolist()
        #     confidence = predictions_confidence[i]
        #     for j in range(len(confidence)):
        #         if confidence[j] == 'A':
        #             confidence[j] = 0
        #         elif confidence[j] == 'B':
        #             confidence[j] = 0.25
        #         elif confidence[j] == 'C':
        #             confidence[j] = 0.5
        #         elif confidence[j] == 'D':
        #             confidence[j] = 0.75
        #     confidence = np.array(confidence)
        #     pred_set = set(pred)
        #     vote = dict()
        #     for ele in pred_set:
        #         vote.update({ele: np.mean(confidence[pred==ele]).item()})
        #     pred = sorted(vote.items(), key=lambda x:x[1])[-1][0]
        #     # pred = max(pred, key=pred.count)        # 选出出现次数最多的元素

        #     # count = Counter(pred)
        #     # confident_pred = [ele for ele in pred if count[ele]>8]
        #     # if len(confident_pred) != 0:
        #     #     pred = confident_pred[0]
        #     # else:
        #     #     continue

        #     ground_truth = gts[i]
        #     num_match = sum([pred == gt for gt in ground_truth])
        #     # for gt in ground_truth:
        #     #     if pred != gt and (pred in gt or gt in pred):
        #     #         import ipdb
        #     #         ipdb.set_trace()
        #     vqa_acc = min(1.0, num_match / 3.0)
        #     acc.append(vqa_acc)

        for i in range(len(predictions)):
            if args.mode == 'confidence':
                pred = np.array(predictions[i])
                confidence = predictions_confidence[i]
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

                # if confidence != ['A']*10:            # 去掉A然后再选出现次数最多的
                #     pred = pred[np.array(confidence)!='A']
                # pred = max(pred.tolist(), key=pred.tolist().count)

                
            elif args.mode == 'vote':
                pred = predictions[i]
                pred = max(pred, key=pred.count)        # 选出出现次数最多的元素
            elif args.mode == 'nll':
                pred = np.array(predictions[i])
                nlls = torch.tensor(predictions_nll[i]).squeeze(-1)
                # pred_set = set(pred)
                # vote = dict()
                # for ele in pred_set:
                #     vote.update({ele: torch.mean(nlls[pred==ele]).item()})
                # pred = sorted(vote.items(), key=lambda x:x[1])[0][0]
                min_index = torch.argmin(nlls)
                pred = predictions[i][min_index.item()]

            ground_truth = gts[i]
            num_match = sum([pred == gt for gt in ground_truth])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        avg_acc.append(accuracy)
    # print(f"smallest: {avg_acc[np.argmin(np.array(avg_acc))]}")
    print(np.mean(avg_acc), np.std(avg_acc))