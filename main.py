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

    def __init__(self, num_classes, input_length, num_rri_features, noise_ratio):
        super(ECG_classifier, self).__init__()
        
        self.conv = nn.Conv1d(1, 20, input_length)
        self.relu = nn.ReLU()

        # self.hidden_augmentation = HiddenAugmentation(noise_ratio)

        # self.rri_weight = nn.Parameter(torch.ones(num_rri_features, requires_grad=True))
        # self.rri_bias = nn.Parameter(torch.zeros(num_rri_features, requires_grad=True))

        self.fc1 = nn.Linear(20+num_rri_features, 16)
        self.fc2 = nn.Linear(16, num_classes)

        # model weight init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
        
    def forward(self, x, rri_features):
        
        x = self.conv(x)
        x = x.flatten(1)
        x = self.relu(x)
        
        # x = self.hidden_augmentation(x)

        # rri_features = rri_features * self.rri_weight + self.rri_bias
        x = torch.cat([x, rri_features], dim=1)

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

    def __init__(self, data, segmentation_method="fully_adaptive", segment_length=200, pre=True, pos=False, prepos=False, local=False, premature_thres=0.05):
        self.signal = data[0]
        self.anno = data[1]
        self.record_name = self.anno['record_name']
        self.record_id = self.anno['record_id']
        self.sample = self.anno['samples']
        self.labels = self.anno['labels']
        self.padding = 1000
        self.signal = torch.cat([torch.zeros(self.padding), self.signal, torch.zeros(self.padding)])
        
        self.segmentation_method = segmentation_method
        self.segment_length = segment_length
        self.thresholding = rri_prematureness_thresholding(premature_thres, 16)

        beat_idxs = self.sample[:10].to(torch.float32)
        self.rri_profile = (beat_idxs[1:] - beat_idxs[:-1]).mean()
        self.rris = torch.ones(16) * self.rri_profile
        self.rris_all = torch.tensor([])

        self.pre = pre
        self.pos = pos
        self.prepos = prepos
        self.local = local

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        # rri feature extraction
        if idx == 0:
            rri = self.sample[idx+1] - self.sample[idx]
        else:
            rri = self.sample[idx] - self.sample[idx-1]

        if idx == len(self.labels) - 1:
            rri_back = self.sample[idx] - self.sample[idx-1]
        else:
            rri_back = self.sample[idx+1] - self.sample[idx]
        
        pre, pos, prepos, local = None, None, None, None
        if self.pre:
            pre = rri/self.rris.mean()
        if self.pos:
            pos = rri_back/self.rris.mean()
        if self.prepos:
            prepos = rri / rri_back
            # prepos = rri_back / rri
        if self.local:
            self.rris_all = torch.cat([self.rris_all, rri.unsqueeze(0)])
            local = self.rris.mean()/self.rris_all.mean()
        
        # RRI prematureness threshold
        # pre = self.thresholding(pre)

        # if self.scaled_rri:
        #     # scaled preceding rri ratio
        #     # pre = -(pre - 0.9)*5
        # pre = -(pre - 0.85)*(2/0.3)

        rri_features = torch.tensor([e for e in [pre, pos, prepos, local] if e is not None])

        self.rris = torch.cat([self.rris[1:], rri.unsqueeze(0)])

        # beat segmentation
        if self.segmentation_method == "constant_point":
            # constant_point beat segmentation
            window_front = 90
            window_back = 110
        elif self.segmentation_method == "semi_adaptive":
            # semi_adaptive beat segmentation
            window_front = (rri - 50).to(torch.long)
            window_back = 100
        elif self.segmentation_method == "fully_adaptive":
            # fully_adaptive beat segmentation
            window_front = (rri * 0.9).to(torch.long)
            window_back = (rri * 0.4).to(torch.long)

        s = self.sample[idx] + self.padding
        segment = self.signal[s-window_front:s+window_back].unsqueeze(0)
        segment = F.interpolate(segment.unsqueeze(0), size=self.segment_length, mode='linear', align_corners=True).squeeze(0)

        return (segment, rri_features, self.record_name, self.record_id, self.sample[idx]), self.labels[idx]
    

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
        self.scaled_rri = config['scaled_rri']
        self.pretrained_model_dict = config['pretrained_model_dict']
        self.apply_pred_refinement = config['apply_pred_refinement']
        self.clustering_method = config['clustering_method']
        self.min_samples = config['min_samples']
        self.min_cluster_size = config['min_cluster_size']
        self.eps = config['eps']
        self.noise_handling = config['noise_handling']
        self.model_save_path = config['model_save_path']
        self.train_dataloader = None
        self.test_dataloader = None
        
        self.device = torch.device("cuda:0")
        self.model = ECG_classifier( num_classes=config['num_classes'],
                            input_length=config['segment_length'],
                            num_rri_features=config['num_rri_features'],
                            noise_ratio=config['noise_ratio']).to(self.device).train()
        
        if self.pretrained_model_dict is not None:
            state_dict = torch.load(self.pretrained_model_dict)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained model from {self.pretrained_model_dict}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'], amsgrad=True)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1, verbose=True)
        
        patients_train = range(len(self.train_samples))
        patients_test = range(len(self.test_samples))

        train_dataset_list = [patient_dataset(self.train_samples[p], 
                                              config['segmentation_method'], 
                                              self.segment_length, 
                                              config['pre'], 
                                              config['pos'], 
                                              config['prepos'], 
                                              config['local'],) for p in patients_train]
        test_dataset_list = [patient_dataset(self.test_samples[p], 
                                             config['segmentation_method'], 
                                             self.segment_length, 
                                             config['pre'], 
                                             config['pos'], 
                                             config['prepos'], 
                                             config['local'],) for p in patients_test]

        print("creating dataset...")
        train_datasets = [custom_dataset(self.create_data(train_dataset_list[p])) for p in tqdm(patients_train)]
        test_datasets = [custom_dataset(self.create_data(test_dataset_list[p])) for p in tqdm(patients_test)]
        train_dataset = ConcatDataset(train_datasets)

        self.sampler = ApplySampler(config['apply_sampler'], train_dataset, config['train_list'], config['sampler_num_samples'])

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=self.sampler, pin_memory=True)
        self.test_dataloaders = [DataLoader(test_datasets[p], batch_size=self.batch_size, pin_memory=True) for p in patients_test]
        
        if self.scaled_rri:

            # calculate rri_ratio mean and std for all training samples
            rri_ratios = torch.tensor([])

            for data in self.train_dataloader:
                (inputs, rri_ratio, record_names, record_id, beat_idxs), labels = data
                rri_ratios = torch.cat((rri_ratios, rri_ratio))

            self.rri_ratio_mean = rri_ratios.mean()
            self.rri_ratio_std = -rri_ratios.std()
            
            # # calculate rri_ratio mean and std for all training samples
            # rri_ratios_N = torch.tensor([])
            # rri_ratios_S = torch.tensor([])
            # rri_ratios_V = torch.tensor([])
            # for data in self.train_dataloader:
            #     (inputs, rri_ratio, record_names, record_id, beat_idxs), labels = data
            #     rri_ratios_N = torch.cat((rri_ratios_N, rri_ratio[labels==0]))
            #     rri_ratios_S = torch.cat((rri_ratios_S, rri_ratio[labels==1]))
            #     rri_ratios_V = torch.cat((rri_ratios_V, rri_ratio[labels==2]))
            # rri_ratio_N_mean = rri_ratios_N.mean()
            # rri_ratio_S_mean = rri_ratios_S.mean()
            # rri_ratio_V_mean = rri_ratios_V.mean()

            # # scale so that largest mean among N, S, V becomes 1, and smallest becomes -1
            # largest_mean = max(rri_ratio_N_mean, rri_ratio_S_mean, rri_ratio_V_mean)
            # smallest_mean = min(rri_ratio_N_mean, rri_ratio_S_mean, rri_ratio_V_mean)
            # self.rri_ratio_mean = (largest_mean + smallest_mean) / 2
            # self.rri_ratio_std = -(largest_mean - smallest_mean) / 2
            # # self.rri_ratio_mean = 0.85
            # # self.rri_ratio_std = -0.3/2

            # print(f"largest_mean: {largest_mean}")
            # print(f"smallest_mean: {smallest_mean}")
            # print(f"rri_ratio_mean: {self.rri_ratio_mean}")
            # print(f"rri_ratio_std: {self.rri_ratio_std}")

            # # calculate rri_ratio mean and std for all training samples
            # rri_ratios_N = torch.tensor([])
            # rri_ratios_SV = torch.tensor([])
            # for data in self.train_dataloader:
            #     (inputs, rri_ratio, record_names, record_id, beat_idxs), labels = data
            #     rri_ratios_N = torch.cat((rri_ratios_N, rri_ratio[labels==0]))
            #     rri_ratios_SV = torch.cat((rri_ratios_SV, rri_ratio[labels!=0]))
            # rri_ratio_N_mean = rri_ratios_N.mean()
            # rri_ratio_SV_mean = rri_ratios_SV.mean()

            # self.rri_ratio_mean = (rri_ratio_N_mean + rri_ratio_SV_mean) / 2
            # self.rri_ratio_std = -(rri_ratio_N_mean - rri_ratio_SV_mean) / 2
            # # self.rri_ratio_mean = 0.85
            # # self.rri_ratio_std = -0.3/2

            # print(f"rri_ratio_mean: {self.rri_ratio_mean}")
            # print(f"rri_ratio_std: {self.rri_ratio_std}")
        
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

    def train_loop(self, dataloader, test_patient):
        overall_gt = []
        overall_predicts = []
        running_loss = 0.0

        for data in dataloader:
            (inputs, rri_ratio, record_names, record_id, beat_idxs), labels = data   # ds_idx is the index of the sample in the dataset
            if test_patient is not None:
                if record_names[0].item() in test_patient:
                    continue
            
            # print(rri_ratio.squeeze()[:5])
            if self.scaled_rri:
                rri_ratio = (rri_ratio - self.rri_ratio_mean) / self.rri_ratio_std
            # print(rri_ratio.squeeze()[:5])

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

    def train(self, test_patient=None):
        print(f"Training on {self.device}")
        best_test_f1 = {}
        best_state_dict = None

        for epoch in range(self.epochs):

            print("")
            print(f"Epoch {epoch+1}/{self.epochs}")

            overall_gt, overall_predicts, running_loss = self.train_loop(self.train_dataloader, test_patient=test_patient)

            classes_loss_train = running_loss / len(self.train_dataloader)
            
            self.model.eval()
            overall_report_test, acc_test, classes_loss_test, _,_,_,_,_,_,_,_ = self.validate(dataloader=self.test_dataloaders, criterion=self.criterion_test, per_patient=True, test_patient=test_patient, print_cm=True)
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

            # self.scheduler.step()

            print_report(overall_report_train, acc_train, "tra")
                
            if log_all['F1_test'] >= best_test_f1.get('F1_test', 0):
                best_test_f1['F1n_test'] = log_all['F1n_test']
                best_test_f1['F1s_test'] = log_all['F1s_test']
                best_test_f1['F1v_test'] = log_all['F1v_test']
                best_test_f1['F1_test'] = log_all['F1_test']
                best_test_f1['Acc_test'] = log_all['Acc_test']
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
        features_tensor = torch.tensor([]).to(self.device)
        classes_tensor = torch.tensor([]).to(self.device)
        record_names_list = []
        
        for data in tqdm(dataloader, desc="test", disable=per_patient):
            (inputs, rri_ratio, record_names, record_id, beat_idxs), labels = data
            inputs = inputs.to(self.device)

            if self.scaled_rri:
                rri_ratio = (rri_ratio - self.rri_ratio_mean) / self.rri_ratio_std

            rri_ratio = rri_ratio.to(self.device)
            labels = labels.to(self.device)

            classes, features = self.model(inputs, rri_ratio)

            loss = criterion(classes, labels)

            _, predict = torch.max(classes.data, 1)

            gt.extend(labels.tolist())
            predicts.extend(predict.tolist())
            running_loss += loss.item()
            features_tensor = torch.cat((features_tensor, features), 0)
            classes_tensor = torch.cat((classes_tensor, classes), 0)

            wrong_idx = (predict != labels).nonzero().squeeze()
            record_names = record_names.to(self.device)
            record_names_list.append(record_names)
            if wrong_idx.numel() > 0:
                wrongs.append((inputs, rri_ratio, labels, predict, record_names, record_id, beat_idxs))

        loss = running_loss / len(dataloader)

        predicts_o = predicts.copy()

        pred_cluster = None
        if self.apply_pred_refinement:
            predicts, pred_cluster = prediction_refinement(features_tensor, 
                                            predicts, 
                                            clustering_method=self.clustering_method, 
                                            eps=self.eps, 
                                            min_samples=self.min_samples, 
                                            min_cluster_size=self.min_cluster_size,
                                            noise_handling=self.noise_handling)

        if per_patient:
            patient_report = classification_report(gt, predicts, labels=torch.arange(len(self.classes)), target_names=self.classes, output_dict=True, zero_division=0)
            patient_acc = accuracy_score(gt, predicts)

            print_report(patient_report, patient_acc, record_names[0].item())

        self.model.train()

        return gt, predicts, loss, wrongs, features_tensor, classes_tensor, predicts_o, pred_cluster, record_names_list

    def validate(self, dataloader, criterion, per_patient, test_patient=None, print_cm=False):
        
        patients = range(len(dataloader))

        self.model.eval()
        with torch.no_grad():
            if per_patient:
                overall_gt = []
                overall_predicts = []
                overall_predicts_o = []
                overall_pred_cluster = []
                overall_record_names = []
                wrongs = []
                features_list = []
                classes_list = []
                running_loss = 0.0
                for p in patients:
                    if test_patient is not None:
                        if next(iter(dataloader[p]))[0][2][0].item() not in test_patient:
                            continue
                    gt, predicts, loss, wrong, features, classes, predicts_o, pred_cluster, record_names = self.validate_loop(dataloader[p], criterion, per_patient=per_patient)

                    overall_gt.append(gt)
                    overall_predicts.append(predicts)
                    overall_predicts_o.append(predicts_o)
                    overall_pred_cluster.append(pred_cluster)
                    overall_record_names.append(record_names)
                    wrongs.extend(wrong)
                    features_list.append(features)
                    classes_list.append(classes)
                    running_loss += loss
                loss = running_loss/len(dataloader)
            else:
                overall_gt, overall_predicts, loss, wrongs, features_list, classes_list, overall_predicts_o, overall_pred_cluster, overall_record_names = self.validate_loop(dataloader, criterion, per_patient=per_patient)
            
            overall_gt_flatten = np.concatenate(overall_gt)
            overall_predicts_flatten = np.concatenate(overall_predicts)
            overall_report_test = classification_report(overall_gt_flatten, overall_predicts_flatten, labels=torch.arange(len(self.classes)), target_names=self.classes, output_dict=True, zero_division=0)
            acc_test = accuracy_score(overall_gt_flatten, overall_predicts_flatten)

            # confusion matrix
            if print_cm:
                from sklearn.metrics import confusion_matrix
                # cm_13 = confusion_matrix(gt_13, predicts_13)
                # print(cm_13)
                cm = confusion_matrix(overall_gt_flatten, overall_predicts_flatten)
                print(cm)
            
            print_report(overall_report_test, acc_test, "tes")
        
        self.model.train()
        
        return overall_report_test, acc_test, loss, overall_gt, overall_predicts, wrongs, features_list, classes_list, overall_predicts_o, overall_pred_cluster, overall_record_names

def run (mode, config, sweep_config=None, project="default", run_name="default"):

    if sweep_config is None or len(sweep_config) == 0:
        sweep_config = [{}]

    for sweep in sweep_config:
        config_copy = deepcopy(config)
        
        # alert if sweep config contains invalid keys
        for k in sweep.keys():
            if k != '_' and k not in config_copy.keys():
                print(f"Invalid key {k} in sweep config")
                raise ValueError(f"Invalid key {k} in sweep config")

        config_copy.update(sweep)

        torch.manual_seed(config_copy['seed'])
        torch.cuda.manual_seed(config_copy['seed'])
        np.random.seed(config_copy['seed'])
        config_copy['num_rri_features'] = config_copy['pre'] + config_copy['pos'] + config_copy['prepos'] + config_copy['local']

        if sweep:
            new_run_name = ' '.join([f"{k}={v}" for k,v in sweep.items()])
        else:
            new_run_name = "default_" + run_name
        
        if config_copy['save_model']:
            config_copy['model_save_path'] = new_run_name
        else:
            config_copy['model_save_path'] = None

        if mode == 'train':
                
            system = System(config_copy)
            
            run = wandb.init(
                # mode="disabled",  # delete this line for wandb logging
                project=project,  
                name=new_run_name,
                save_code=True,
                config=config_copy)
            
            system.train()

            wandb.finish()

        elif mode == 'leave_k_patient_out':
            DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
            # randomize DS1 
            np.random.shuffle(DS1)
            config_copy['test_samples'] = config_copy['train_samples']
            k = 11
            for i in range(0, len(DS1), k):
                
                torch.manual_seed(config_copy['seed'])
                torch.cuda.manual_seed(config_copy['seed'])
                np.random.seed(config_copy['seed'])

                test_patient = DS1[i:i+k]

                system = System(config_copy)
                
                wandb.init(
                    # mode="disabled",  # delete this line for wandb logging
                    project=project,  
                    name=f"l4porandom_{test_patient}_{new_run_name}",
                    save_code=True,
                    config=config_copy)
                
                system.train(test_patient=test_patient)

                wandb.finish()
        
        elif mode == 'validate':
            config_copy['pretrained_model_dict'] = config_copy['model_save_path']+".pth"

            system = System(config_copy)

            report, acc, loss, gt, predicts, wrongs, features, predprob, predicts_o, pred_cluster, record_names = system.validate(system.test_dataloaders, config_copy['criterion_test'], per_patient=True, print_cm=True)

            out = {
                'pt_features_list': features,
                'pt_gt': gt,
                'pt_predicts': predicts,
                'pt_predicts_o': predicts_o,
                'pt_cluster': pred_cluster,
                'pt_record_names': record_names,
            }
            
            with open('out_test.pkl', 'wb') as f:
                pickle.dump(out, f)

            # output_wrong(wrongs_test)

        else: 
            raise ValueError("Invalid mode. Should be either 'train', 'validate' or 'leave_k_patient_out'")

if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA is not available"

    config = {}
    config['epochs'] = 20
    config['batch_size'] = 64
    config['optimizer'] = "AdamW"
    config['lr'] = 0.0001
    config['train_list'] = torch.tensor([45866, 944, 3788], dtype=torch.float) # torch.tensor([38102, 3949, 3783, 16, 16, 810, 100, 32, 2, 3683, 105, 415, 8], dtype=torch.float)
    config['test_list'] = torch.tensor([44259, 1837, 3221], dtype=torch.float) # torch.tensor([36444, 4126, 3476, 1, 213, 1736, 50, 51, 1, 3220, 1, 388, 7], dtype=torch.float)
    config['criterion_train'] = nn.CrossEntropyLoss(weight=(config['train_list'].mean() / config['train_list']).to(torch.device("cuda:0"))) # scikit learn class imbalance weight
    config['criterion_test'] = nn.CrossEntropyLoss(weight=(config['test_list'].mean() / config['test_list']).to(torch.device("cuda:0")))
    
    config['noise_ratio'] = 0

    config['classes'] = ["N", "S", "V"]
    config['num_classes'] = len(config['classes'])
    config['train_samples'] = ".\data\ds1.pkl"
    config['test_samples'] = ".\data\ds2.pkl"

    config['segmentation_method'] = "fully_adaptive"
    config['segment_length'] = 200

    config['pre'] = 1
    config['pos'] = 0
    config['prepos'] = 0
    config['local'] = 0
    config['num_rri_features'] = config['pre'] + config['pos'] + config['prepos'] + config['local']
    config['scaled_rri'] = 0

    config['apply_sampler'] = False
    config['sampler_num_samples'] = None

    config['apply_pred_refinement'] = 1
    config['clustering_method'] = "DBSCAN"
    config['eps'] = 0.4
    config['min_samples'] = 5
    config['min_cluster_size'] = 5
    config['noise_handling'] = "merge_most"

    config['seed'] = 1

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    config['pretrained_model_dict'] = None # insert path to pretrained model
    run_name = (f"{config['segmentation_method']}{config['clustering_method']}{config['eps']}{config['noise_handling']}" 
    f"pre{config['pre']}pos{config['pos']}prepos{config['prepos']}local{config['local']}scaled{config['scaled_rri']}")
    config['save_model'] = False

    project = "final_noscale"
    mode = 'train' # 'train', 'validate', 'leave_k_patient_out'
    sweep_config = [
        # {},
        # {'seed':2},
        # {'seed':3},
        # {'eps': 0.2},
        # {'eps': 0.3},
        # {'eps': 0.4},
        # {'eps': 0.5},
        # {'min_samples': 2},
        # {'min_samples': 3},
        # {'min_samples': 4},
        # {'min_samples': 5},
        # {'min_samples': 6},
        # {'min_samples': 7},
        # {},
        # {'clustering_method': 'HDBSCAN'},
        # {'clustering_method': 'HDBSCAN', 'min_samples': 3},
        # {'clustering_method': 'HDBSCAN', 'min_samples': 4},
        # {'clustering_method': 'HDBSCAN', 'min_samples': 5},
        # {'clustering_method': 'HDBSCAN', 'min_samples': 6},
        {'clustering_method': 'HDBSCAN', 'min_samples': 4, 'min_cluster_size': 3},
        {'clustering_method': 'HDBSCAN', 'min_samples': 4, 'min_cluster_size': 4},
        {'clustering_method': 'HDBSCAN', 'min_samples': 4, 'min_cluster_size': 5},
        {'clustering_method': 'HDBSCAN', 'min_samples': 4, 'min_cluster_size': 6},
        # {'_':'ablation', 'pre':0, 'apply_pred_refinement':0},
        # {'_':'ablation', 'pre':1, 'apply_pred_refinement':0},
        # {'_':'ablation', 'pre':1, 'apply_pred_refinement':1},
        # {'_':'ablation', 'pre':1, 'scaled_rri':1, 'apply_pred_refinement':1},
        # {'segmentation_method': 'constant_point'},
        # {'segmentation_method': 'semi_adaptive'},
        # {'_':'rri', 'pre':1, 'pos':0, 'prepos':0, 'local':0},
        # {'_':'rri', 'pre':0, 'pos':1, 'prepos':0, 'local':0},
        # {'_':'rri', 'pre':0, 'pos':0, 'prepos':1, 'local':0},
        # {'_':'rri', 'pre':0, 'pos':0, 'prepos':0, 'local':1},
        # {'_':'rri', 'pre':1, 'pos':1, 'prepos':0, 'local':0},
        # {'_':'rri', 'pre':1, 'pos':0, 'prepos':1, 'local':0},
        # {'_':'rri', 'pre':0, 'pos':0, 'prepos':1, 'local':1},
        # {'_':'rri', 'pre':1, 'pos':1, 'prepos':1, 'local':1},
        # {'noise_handling': 'retain_all'},
        # {'noise_handling': 'merge_all'},
    ]
    
    run(mode, config, sweep_config, project, run_name)

