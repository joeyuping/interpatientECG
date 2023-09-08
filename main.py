import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import wandb
from copy import deepcopy
from tqdm import tqdm
import pickle
from util import *
from prediction_refinement import *


class HiddenAugmentation(nn.Module):
    
    def __init__(self, noise_ratio):
        super(HiddenAugmentation, self).__init__()
        self.noise_ratio = noise_ratio

    def forward(self, x):

        # hidden augmentation
        if self.training:
            noise = 1 + torch.randn_like(x) * self.noise_ratio
            x = x * noise

        return x


class rri_prematureness_thresholding():

    def __init__(self, prematureness_threshold, window_size):
        self.rri_ratios = torch.ones(window_size)
        self.prematureness_threshold = prematureness_threshold

    def __call__(self, rri_ratio):
        self.rri_ratios = torch.cat((self.rri_ratios[1:], rri_ratio.unsqueeze(0)))
        sorted_rri_ratios = torch.sort(self.rri_ratios)[0]
        diff_rri = sorted_rri_ratios[1:] - sorted_rri_ratios[:-1]
        rri_gap = torch.max(diff_rri)
        if rri_gap < self.prematureness_threshold:
            rri_ratio = torch.tensor(1.0)
        return rri_ratio


class ECG_classifier(nn.Module):

    def __init__(self, num_classes, input_length, noise_ratio):
        super(ECG_classifier, self).__init__()
        
        self.convall = nn.Conv1d(1, 20, input_length)
        self.relu = nn.ReLU()

        self.hidden_augmentation = HiddenAugmentation(noise_ratio)

        self.fc1 = nn.Linear(20+1, 16)
        self.fc2 = nn.Linear(16, num_classes)

        # model weight init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
        
    def forward(self, x, rri_ratio):
        
        x = self.convall(x)
        x = x.flatten(1)
        x = self.relu(x)
        
        x = self.hidden_augmentation(x)
        
        x = torch.cat([x, rri_ratio.unsqueeze(1)], dim=1)

        feature_vectors = F.relu(self.fc1(x))  # feature_vectors will be fed to unsupervised clustering
        x = self.fc2(feature_vectors)

        return x, feature_vectors


class custom_dataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


class patient_dataset(Dataset):

    def __init__(self, data, segment_length=200, premature_thres=0.05):
        self.signal = data[0]
        self.anno = data[1]
        self.record_name = self.anno['record_name']
        self.record_id = self.anno['record_id']
        self.sample = self.anno['samples']
        self.labels = self.anno['labels']
        self.padding = 1000
        self.signal = torch.cat([torch.zeros(self.padding), self.signal, torch.zeros(self.padding)])
        beat_idxs = self.sample[:10].to(torch.float32)
        self.rri_profile = (beat_idxs[1:] - beat_idxs[:-1]).mean()
        self.segment_length = segment_length
        self.thresholding = rri_prematureness_thresholding(premature_thres, 16)

        self.rris = torch.ones(16) * self.rri_profile

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        if idx == 0:
            rri = self.sample[idx+1] - self.sample[idx]
        else:
            rri = self.sample[idx] - self.sample[idx-1]
        
        # # constant point beat segmentation
        # window_front = 90
        # window_back = 110

        # # semi-adaptive beat segmentation
        # window_front = (rri - 50).to(torch.long)
        # window_back = 100

        # fully adaptive beat segmentation
        window_front = (rri * 0.9).to(torch.long)
        window_back = (rri * 0.4).to(torch.long)

        s = self.sample[idx] + self.padding
        segment = self.signal[s-window_front:s+window_back].unsqueeze(0)
        segment = F.interpolate(segment.unsqueeze(0), size=self.segment_length, mode='linear', align_corners=True).squeeze(0)

        rri_ratio = rri/self.rris.mean()
        self.rris = torch.cat([self.rris[1:], rri.unsqueeze(0)])

        # RRI prematureness threshold
        rri_ratio = self.thresholding(rri_ratio)

        if rri_ratio > 4:
            rri_ratio = torch.tensor(1.0)

        # scaled rri ratio
        rri_ratio = -(rri_ratio - 0.9)*5

        return (segment, rri_ratio, self.record_name, self.record_id, self.sample[idx]), self.labels[idx]
    

class System():

    def __init__(self, config):
        with open(config['train_samples'], "rb") as f:
            ds1 = pickle.load(f)
        with open(config['test_samples'], "rb") as f:
            ds2 = pickle.load(f)
        self.train_samples = ds1 
        self.test_samples = ds2
        self.classes = config['classes']
        self.num_classes = config['num_classes']
        self.segment_length = config['segment_length']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.criterion_train = config['criterion_train']
        self.criterion_test = config['criterion_test']
        self.pretrained_model_dict = config['pretrained_model_dict']
        self.apply_pred_refinement = config['apply_pred_refinement']
        self.clustering_method = config['clustering_method']
        self.min_samples = config['min_samples']
        self.eps = config['eps']
        self.noise_handling = config['noise_handling']
        self.model_save_path = config['model_save_path']
        self.train_dataloader = None
        self.test_dataloader = None
        
        self.device = torch.device("cuda:0")
        self.model = ECG_classifier( num_classes=config['num_classes'],
                            input_length=config['segment_length'],
                            noise_ratio=config['noise_ratio']).to(self.device).train()
        
        if self.pretrained_model_dict is not None:
            state_dict = torch.load(self.pretrained_model_dict)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained model from {self.pretrained_model_dict}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'], amsgrad=True)
        
        patients_train = range(len(self.train_samples))
        patients_test = range(len(self.test_samples))

        train_dataset_list = [patient_dataset(self.train_samples[p], self.segment_length) for p in patients_train]
        test_dataset_list = [patient_dataset(self.test_samples[p], self.segment_length) for p in patients_test]

        print("creating dataset...")
        train_datasets = [custom_dataset(self.create_data(train_dataset_list[p])) for p in tqdm(patients_train)]
        test_datasets = [custom_dataset(self.create_data(test_dataset_list[p])) for p in tqdm(patients_test)]
        train_dataset = ConcatDataset(train_datasets)

        self.sampler = ApplySampler(config['apply_sampler'], train_dataset, config['train_list'], config['sampler_num_samples'])

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=self.sampler, pin_memory=True)
        self.test_dataloaders = [DataLoader(test_datasets[p], batch_size=self.batch_size, pin_memory=True) for p in patients_test]
        print("done!")
        
    def create_data(self, dataset):
        X = []
        Y = []
        for sample in dataset:
            conversion = [0,0,0,0,0,1,1,1,1,2,2,3,4]
            label = sample[1]
            label = conversion[label]
            if label==3 or label==4:   # ignore F and Q class
                continue
            X.append(sample[0])
            Y.append(label)

        ds = [[X[i], Y[i]] for i in range(len(X))]
        return ds

    def train_loop(self, dataloader, desc=f"train"):
        overall_gt = []
        overall_predicts = []
        running_loss = 0.0

        for data in tqdm(dataloader,desc=desc):
            (inputs, rri_ratio, record_names, record_id, beat_idxs), labels = data   # ds_idx is the index of the sample in the dataset
            inputs = inputs.to(self.device)
            rri_ratio = rri_ratio.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            classes, features = self.model(inputs, rri_ratio)

            classes_loss = self.criterion_train(classes, labels)
            
            _, predict = torch.max(classes.data, 1)

            overall_gt.extend(labels.tolist())
            overall_predicts.extend(predict.tolist())
            running_loss += classes_loss.item() 

            classes_loss.backward()

            self.optimizer.step()
    
        return overall_gt, overall_predicts, running_loss

    def train(self):
        print(f"Training on {self.device}")
        best_test_f1 = {}
        best_state_dict = None

        for epoch in range(self.epochs):

            print("")
            print(f"Epoch {epoch+1}/{self.epochs}")

            overall_gt, overall_predicts, running_loss = self.train_loop(self.train_dataloader)

            classes_loss_train = running_loss / len(self.train_dataloader)
            
            self.model.eval()
            overall_report_test, acc_test, classes_loss_test, _,_,_,_,_ = self.validate(dataloader=self.test_dataloaders, criterion=self.criterion_test, per_patient=True, print_cm=True)
            self.model.train()
            
            overall_report_train = classification_report(overall_gt, overall_predicts, labels=torch.arange(len(self.classes)), target_names=self.classes, output_dict=True, zero_division=0)
            acc_train = accuracy_score(overall_gt, overall_predicts)

            log_all = {
                    'Acc_train': acc_train,
                    'Acc_test': acc_test,
                    'F1n_train': overall_report_train['N']['f1-score'],
                    'F1n_test': overall_report_test['N']['f1-score'],
                    'F1s_train': overall_report_train['S']['f1-score'],
                    'F1s_test': overall_report_test['S']['f1-score'],
                    'F1v_train': overall_report_train['V']['f1-score'],
                    'F1v_test': overall_report_test['V']['f1-score'],
                    'F1_train': (overall_report_train['S']['f1-score'] + overall_report_train['V']['f1-score']) / 2,
                    'F1_test': (overall_report_test['S']['f1-score'] + overall_report_test['V']['f1-score']) / 2,
                    'Loss_train': classes_loss_train,
                    'Loss_test': classes_loss_test,
                }
            
            wandb.log(log_all, step=epoch) 

            print_report(overall_report_train, acc_train, "tra")
                
            if log_all['F1_test'] >= best_test_f1.get('F1_test', 0):
                best_test_f1['F1_test'] = log_all['F1_test']
                best_test_f1['epoch'] = epoch+1
                best_state_dict = deepcopy(self.model.state_dict())
                torch.save(best_state_dict, f"{self.model_save_path}.pth")
        
        wandb.summary['best_test_f1'] = best_test_f1

    def validate_loop(self, dataloader, criterion, per_patient):
        self.model.eval()
        gt = []
        predicts = []
        running_loss = 0.0
        wrongs = []
        features_list = []
        classes_list = []
        
        for data in tqdm(dataloader, desc="test", disable=per_patient):
            (inputs, rri_ratio, record_names, record_id, beat_idxs), labels = data
            inputs = inputs.to(self.device)
            rri_ratio = rri_ratio.to(self.device)
            labels = labels.to(self.device)

            classes, features = self.model(inputs, rri_ratio)

            loss = criterion(classes, labels)

            _, predict = torch.max(classes.data, 1)

            gt.extend(labels.tolist())
            predicts.extend(predict.tolist())
            running_loss += loss.item()
            features_list.append(features)
            classes_list.append(classes)

            wrong_idx = (predict != labels).nonzero().squeeze()
            record_names = record_names.to(self.device)
            if wrong_idx.numel() > 0:
                wrongs.append((inputs, rri_ratio, labels, predict, record_names, record_id, beat_idxs))

        loss = running_loss / len(dataloader)

        if self.apply_pred_refinement:
            predicts = prediction_refinement(features_list, 
                                            predicts, 
                                            clustering_method=self.clustering_method, 
                                            eps=self.eps, 
                                            min_samples=self.min_samples, 
                                            noise_handling=self.noise_handling)

        if per_patient:
            patient_report = classification_report(gt, predicts, labels=torch.arange(len(self.classes)), target_names=self.classes, output_dict=True, zero_division=0)
            patient_acc = accuracy_score(gt, predicts)

            print_report(patient_report, patient_acc, record_names[0].item())

        self.model.train()

        return gt, predicts, loss, wrongs, features_list, classes_list

    def validate(self, dataloader, criterion, per_patient, print_cm=False):
        
        patients = range(len(dataloader))

        self.model.eval()
        with torch.no_grad():
            if per_patient:
                overall_gt = []
                overall_predicts = []
                wrongs = []
                features_list = []
                classes_list = []
                running_loss = 0.0
                for p in patients:
                    gt, predicts, loss, wrong, features, classes = self.validate_loop(dataloader[p], criterion, per_patient=per_patient)

                    overall_gt.extend(gt)
                    overall_predicts.extend(predicts)
                    wrongs.extend(wrong)
                    features_list.append(features)
                    classes_list.append(classes)
                    running_loss += loss
                loss = running_loss/len(dataloader)
            else:
                overall_gt, overall_predicts, loss, wrongs, features_list, classes_list = self.validate_loop(dataloader, criterion, per_patient=per_patient)
            
            overall_report_test = classification_report(overall_gt, overall_predicts, labels=torch.arange(len(self.classes)), target_names=self.classes, output_dict=True, zero_division=0)
            acc_test = accuracy_score(overall_gt, overall_predicts)

            # confusion matrix
            if print_cm:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(overall_gt, overall_predicts)
                print(cm)
            
            print_report(overall_report_test, acc_test, "tes")
        
        self.model.train()
        
        return overall_report_test, acc_test, loss, overall_gt, overall_predicts, wrongs, features_list, classes_list


if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA is not available"

    run_name = f"hdbscan"
    
    config = {}
    config['train'] = True
    config['epochs'] = 20
    config['batch_size'] = 64
    config['optimizer'] = "AdamW"
    config['lr'] = 0.0003
    config['train_list'] = torch.tensor([45866, 944, 3788], dtype=torch.float) # torch.tensor([38102, 3949, 3783, 16, 16, 810, 100, 32, 2, 3683, 105, 415, 8], dtype=torch.float)
    config['test_list'] = torch.tensor([44259, 1837, 3221], dtype=torch.float) # torch.tensor([36444, 4126, 3476, 1, 213, 1736, 50, 51, 1, 3220, 1, 388, 7], dtype=torch.float)
    config['criterion_train'] = nn.CrossEntropyLoss(weight=(config['train_list'].mean() / config['train_list']).to(torch.device("cuda:0"))) # scikit learn class imbalance weight
    config['criterion_test'] = nn.CrossEntropyLoss(weight=(config['test_list'].mean() / config['test_list']).to(torch.device("cuda:0")))
    config['noise_ratio'] = 0.05

    config['pretrained_model_dict'] = None # insert path to pretrained model
    config['model_save_path'] = run_name

    config['classes'] = ["N", "S", "V"]
    config['num_classes'] = len(config['classes'])
    config['train_samples'] = ".\data\ds1.pkl"
    config['test_samples'] = ".\data\ds2.pkl"
    config['segment_length'] = 200

    config['apply_sampler'] = False
    config['sampler_num_samples'] = None

    config['apply_pred_refinement'] = True
    config['clustering_method'] = "DBSCAN"
    config['eps'] = 0.5
    config['min_samples'] = 3
    config['noise_handling'] = "merge_most"

    config['seed'] = 1

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    if config['train']:

        system = System(config)
        
        run = wandb.init(
            mode="disabled",  # delete this line for wandb logging
            project="cluster",  
            name=run_name,
            save_code=True,
            config=config)
        
        system.train()

        wandb.finish()

    else:
        config['pretrained_model_dict'] = config['model_save_path']+".pth"
        config['batch_size'] = 16

        system = System(config)

        report_test, acc_test, loss_test, gt_test, predicts_test, wrongs_test, features_test, predprob_test = system.validate(system.test_dataloaders, config['criterion_test'], per_patient=True, print_cm=True)

        # output_wrong(wrongs_test)