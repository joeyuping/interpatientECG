import sys
import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(f".\\test_report\\{file_name}.log", "a")
        sys.stdout = self
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

    def stop(self):
        self.log.close()
        sys.stdout = self.terminal

def tsne(features, gt, pred, casebases=None, perplexity=30):
    if torch.is_tensor(features):
        features = features.cpu().detach().numpy()
    else:
        features = np.array(features)
    if torch.is_tensor(gt):
        gt = gt.cpu().detach().numpy()
    else:
        gt = np.array(gt)
    if torch.is_tensor(pred):
        pred = pred.cpu().detach().numpy()
    else:
        pred = np.array(pred)

    if casebases is not None:
        if torch.is_tensor(casebases):
            casebases = casebases.cpu().detach().numpy()
        else:
            casebases = np.array(casebases)
        
        num_classes = len(casebases)
        # from casebases extract case label and case parameters as case features
        case_features = []
        case_labels = []
        case_case = []
        for i, casebase in enumerate(casebases):
            for case in casebase:
                case_features.append(list(case.parameters()))
                case_labels.append(i)
                case_case = num_classes
        case_features = np.array(case_features)
        case_labels = np.array(case_labels)
        case_case = np.array(case_case)

        # concat case features, case labels and case_case to features, pred and gt respectively
        features = np.concatenate([features, case_features], axis=0)
        pred = np.concatenate([pred, case_labels], axis=0)
        gt = np.concatenate([gt, case_case], axis=0)

    # tsne
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, verbose=1)
    X_tsne = tsne.fit_transform(features)

    # tsne(z, gt, predicts, perplexity=30)

    # subplots
    classes = ["N", "S", "V"]
    COLORS = ["#24abd1", "#d64f4f", "#66de80"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, color in enumerate(COLORS):
        idx = np.where(gt == i)
        ax[0].scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=color, s=5, label=classes[i])
        idx = np.where(pred == i)
        ax[1].scatter(X_tsne[idx, 0], X_tsne[idx, 1], c=color, s=5, label=classes[i]) 
    ax[0].set_title("Ground Truth")
    ax[0].legend()
    ax[1].set_title("Predictions")
    ax[1].legend()
    plt.show()

def print_report(report, acc, desc):
    print(desc, end="")
    print(f"[{int(report['N']['support']):^5d},{int(report['S']['support']):^4d},{int(report['V']['support']):^4d}]", end=" ")
    print(f"N:{report['N']['recall']:.4f},{report['N']['precision']:.4f},{report['N']['f1-score']:.4f};", end=" ")
    print(f"S:{report['S']['recall']:.4f},{report['S']['precision']:.4f},{report['S']['f1-score']:.4f};", end=" ")
    print(f"V:{report['V']['recall']:.4f},{report['V']['precision']:.4f},{report['V']['f1-score']:.4f};", end=" ") 
    print(f"Acc:{acc:.4f},F1sv:{((report['S']['f1-score'] + report['V']['f1-score'])/2):.4f}")

def ApplySampler(apply, dataset, list, num_samples=None, replacement=True):
    if not apply:
        return None
    else:
        from torch.utils.data import WeightedRandomSampler

        print("creating sampler weights...")
        sampler_weights = torch.zeros(len(dataset))
        for i in range(len(dataset)):
            _, label = dataset[i]
            sampler_weights[i] = list[label]
        sampler_weights = 1/sampler_weights
        sampler_weights = sampler_weights / sampler_weights.sum()
        sampler = WeightedRandomSampler(sampler_weights, num_samples, replacement=replacement)
        return sampler
    
def output_wrong(test_samples, wrongs):
    
    import os 
        
    # delete all files in D:/wrongbeats and D:/wrongs
    for filename in os.listdir("D:/wrongbeats"):
        os.remove("D:/wrongbeats/"+filename)
    for filename in os.listdir("D:/wrongs"):
        os.remove("D:/wrongs/"+filename)

    with open(test_samples, "rb") as f:
        ds = pickle.load(f)
    # classes_13 = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', 'Q']
    classes = ['N', 'S', 'V']
    count_wrongs = {}
    for i, wrong in enumerate(wrongs):
        inputs, rri_ratios, labels, predicts, record_names, record_id, beat_idx = wrong

        labels = labels.cpu().numpy().squeeze()
        predicts = predicts.cpu().numpy().squeeze()
        # # features = features.cpu().numpy().squeeze()
        data = ds[record_id[0]]
        record_name = record_names[0].item()
        signal = data[0]
        signal = torch.from_numpy(signal).to(torch.float32)
        start = beat_idx[0] - 300
        end = beat_idx[-1] + 300
        if start < 0:
            start = 0
        if end > len(signal):
            end = len(signal)
        signal = signal[start:end]
        beat_idx = beat_idx - start
        if signal.shape[0] == 0:
            continue
        plt.figure(figsize=(10, 3))
        plt.plot(signal.cpu().numpy().squeeze())
        plt.plot(beat_idx, signal[beat_idx].cpu().numpy().squeeze(), 'ko')
        plt.ylim(-1,2)
        for beat_id, label, predict, rri_ratio in zip(beat_idx, labels, predicts, rri_ratios):
            rri_ratio = 0.9 - rri_ratio / 10
            if label != predict:
                plt.text(beat_id, signal[beat_id].cpu().numpy().squeeze()+0.1, f"G:{classes[label]}\nP:{classes[predict]}", color='r', fontsize=10)
            else:
                plt.text(beat_id, signal[beat_id].cpu().numpy().squeeze()+0.1, f"G:{classes[label]}\nP:{classes[predict]}", color='k', fontsize=10)
        plt.title(f"Record: {record_name}")
        plt.savefig(f"D:/wrongs/{i}_{record_name}.png")
        plt.close()

        # plot input
        inputs = inputs.cpu().numpy().squeeze()
        wrong_id = np.where(labels != predicts)[0]
        for j, id in enumerate(wrong_id):
            rri_ratio = rri_ratios[id]
            rri_ratio = 0.9 - rri_ratio / 10
            plt.figure(figsize=(4, 3))
            plt.plot(inputs[id])
            plt.title(f"Record: {record_names[id]}, label: {classes[labels[id]]}, pred: {classes[predicts[id]]}, rri: {rri_ratio:.3f}")
            plt.savefig(f"D:/wrongbeats/{i}-{j}_{record_names[id]}.png")
            plt.close()

            # count wrongs
            count_wrongs[record_names[id].item()] = count_wrongs.get(record_names[id].item(), 0) + 1

    #sort count_wrongs
    count_wrongs = sorted(count_wrongs.items(), key=lambda x: x[1], reverse=True)
    print(count_wrongs)

def create_data_noiseremoval(dataset):
        X = []
        Y = []
        noise_level_list = []
        noise_idx = []
        count = 0

        noises = []
        for sample, label in dataset:
            conversion = [0,0,0,0,0,1,1,1,1,2,2,3,4]
            label = conversion[label]
            if label==3 or label==4:   # ignore F and Q class
                continue
            X.append(sample)
            Y.append(label)
            segment = sample[0]
            # calculate noise_level of segment
            noise_level = torch.std(segment[0])
            
            if noise_level > 0.3:
                noises.append(count)
                noise_level_list.append(noise_level * 10)
            else:
                noise_level_list.append(noise_level)
            count += 1
        keep_list = []
        for k, id in enumerate(noises, leave=False):
            segment = X[id][0]
            for i, id2 in enumerate(noises):
                if i==k:
                    continue
                segment_i = X[id2][0]
                noise_level = torch.std(segment-segment_i)
                if noise_level < 0.3:
                    keep_list.append(id)
                    break
            noise_level_list[id] = noise_level * 10
        # remove from noise_idx those in keep list
        noises2 = [id for id in noises if id not in keep_list]

        noise_idx.extend(noises2)
        print(f"record: {sample[3]}, noises_initial: {len(noises)}, noises_final: {len(noises2)}")

        for i in range(len(X)):
            if i in noise_idx:
                noise_level_list[i] = noise_level_list[i] * 10
        # remove from X and Y those in noise_idx
        X = [X[i] for i in range(len(X)) if i not in noise_idx]
        Y = [Y[i] for i in range(len(Y)) if i not in noise_idx]
        noise_level_list = [noise_level_list[i] for i in range(len(noise_level_list)) if i not in noise_idx]

        print(f"Total noises removed: {len(noise_idx)}")

        # ds = [[X[i], Y[i], noise_level_list[i]] for i in range(len(X))]
        ds = [[X[i], Y[i]] for i in range(len(X))]

        return ds