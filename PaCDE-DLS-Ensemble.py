import sys

import hpelm
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
from pokemon0 import Pokemon
from torchvision.models import resnet18
import os
from utils import Flatten
from mealpy import FloatVar, MPA
import pickle
from torchvision import transforms, models
from    PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    total_val =0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        total_val += y.size(0)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def get_evaluate_acc_pred(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    total_val = 0
    predictions = []  

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits
            predictions.extend(pred.cpu().numpy())  
         
    return predictions
def get_evaluate_acc_pred0(model, loader):
    global device
    model.eval()

    correct = 0
    total = 0  
    predictions = []  

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)            
            pred = logits.argmax(dim=1)  
            predictions.extend(pred.cpu().numpy())  
        
        total += y.size(0)        
        correct += (pred == y).sum().item()
    
    if total == 0:
        raise ValueError("数据加载器没有提供任何样本，请检查数据加载器的配置。")

    accuracy = correct / total
    return accuracy, predictions


def getevaluteY(model, loader):
    pre_Y = []
    Y = []
    model.eval()

    correct = 0
    total = len(loader.dataset)
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)            
            pre_Y.extend(pred.cpu().numpy())
            Y.extend(y.cpu().numpy())
        correct += torch.eq(pred, y).sum().float().item()

    return pre_Y, Y

def set_seed(seed):
    random.seed(seed)                       # 设置Python的随机种子
    np.random.seed(seed)                    # 设置NumPy的随机种子
    torch.manual_seed(seed)                 # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)            # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)        # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False


class Mobilenet_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
            
        self.block = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()
        
    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x
        
        
    
class Alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        modelAlexNet = models.alexnet(pretrained=True)

        for param in modelAlexNet.parameters():
            param.requires_grad = False
        
        self.base_model = modelAlexNet
        self.base_model.classifier = nn.Sequential(
                
                nn.Linear(256 * 6 * 6, 128),  # AlexNet 原始分类器的输入尺寸
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),                
                nn.Linear(128, 3)
            )

    def forward(self, x):
        x = self.base_model(x)
        return x


class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerModel, self).__init__()
        
        self.base = models.swin_t(weights='IMAGENET1K_V1')  

        for param in list(self.base.parameters())[:-15]: 
            param.requires_grad = False

        self.base.head = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(self.base.head.in_features, 128), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes) 
        )
    def forward(self, x):
        x = self.base(x)
        return x

# 训练新的模型
def generativeModel():
    global device, x_train, y_train

    
    
    batchsz =64
    lr = 1e-3
    epochs =20
    num_cuda_devices = torch.cuda.device_count()
    print(f"当前系统上有 {num_cuda_devices} 个可用的CUDA设备。")

    # 指定要使用的CUDA设备
    desired_device_id = 0  # 选择要使用的设备的ID
    if desired_device_id < num_cuda_devices:
        torch.cuda.set_device(desired_device_id)
        print(f"已将CUDA设备切换到设备ID为 {desired_device_id} 的设备。")
    else:
        print(f"指定的设备ID {desired_device_id} 超出可用的CUDA设备数量。")
    device = torch.device('cuda:0')
    parent_dir = os.path.dirname(os.getcwd())
    # 获取当前脚本文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取当前脚本文件的父文件夹
    cwd_dir = os.path.dirname(script_path)

    

   
    model_name = ["SwinTransformerModel", "alexnet_model","mobilenet_v2_model"]

    for index  in range(0,3):                    
        val_acc_Trial = np.zeros((5, epochs))
        train_acc_Trial = np.zeros((5, epochs))
        val_loss_Trial = np.zeros((5, epochs))
        train_loss_Trial = np.zeros((5, epochs))
        test_acc_list=np.zeros((5, 1))
        for ii in range(0, 5):  
            set_seed(43+index+ii )

            if model_name[index]=="mobilenet_v2_model":                
                model=Mobilenet_v2().to(device)
            elif model_name[index]== "alexnet_model":             
                model=Alexnet().to(device)
            elif model_name[index]=="SwinTransformerModel":
                model=SwinTransformerModel().to(device)
            else:
                pass
            
            print(f"执行模型{model_name[index]}:第{ii}次 -------------")
             
            filemame = f"images{0}.csv"            

            train_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='train')
            val_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='val')
            test_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='test')                      

            # Create indices and random sampler
            indices = np.arange(len(train_db))
            indices1 = np.arange(len(val_db))
            indices2 = np.arange(len(test_db))
            # Use SubsetRandomSampler with a random seed
            sampler = SubsetRandomSampler(indices )
            sampler1 = SubsetRandomSampler(indices1)
            sampler2 = SubsetRandomSampler(indices2)
           
            train_loader = DataLoader(train_db, batch_size=batchsz, sampler=sampler)
            val_loader = DataLoader(val_db, batch_size=batchsz, shuffle=False)
            test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=False)

            optimizer = optim.Adam(model.parameters(), lr=lr )
            criteon = nn.CrossEntropyLoss()

            best_acc, best_epoch = 0, 0
            global_step = 0
            
            for epoch in range(epochs):
                correct_train = 0  
                total_train = 0  
                train_loss = 0  

                for step, (x, y) in enumerate(train_loader):
                    
                    x, y = x.to(device), y.to(device)

                    model.train()
                    logits = model(x,)

                     
                    loss = criteon(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    
                    train_loss += loss.item()                    
                    _, preds = torch.max(logits, 1)  
                    correct_train += (preds == y).sum().item()
                    total_train += y.size(0)  

                    global_step += 1

                
                train_acc = correct_train / total_train
                avg_train_loss = train_loss / len(train_loader)

                # 验证阶段
                model.eval()
                val_loss = 0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        logits = model(val_x)
                        loss = criteon(logits, val_y)
                        val_loss += loss.item()
                        _, val_preds = torch.max(logits, 1)
                        correct_val += (val_preds == val_y).sum().item()
                        total_val += val_y.size(0)

                # 计算验证集的平均损失和准确率
                avg_val_loss = val_loss / len(val_loader)
                val_acc = correct_val / total_val

                # 存储准确率和损失
                val_acc_Trial[ii, epoch] = val_acc
                train_acc_Trial[ii, epoch] = train_acc
                val_loss_Trial[ii, epoch] = avg_val_loss
                train_loss_Trial[ii, epoch] = avg_train_loss

                if epoch % 1 == 0:

                    if val_acc > best_acc:
                        best_epoch = epoch
                        best_acc = val_acc
                        dirp = cwd_dir
                        if os.path.exists(os.path.join(dirp,model_name[index],str(epochs),"dim")) == False:
                            os.makedirs(os.path.join(dirp,model_name[index],str(epochs),"dim"))
                        torch.save(model.state_dict(), f'{dirp}/{model_name[index]}/{str(epochs)}/dim/best{ii}.mdl')
  
                print("epoch:", {epoch}, ":best_acc", {best_acc})
                print(f"Epoch [{epoch}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}") 
            
            
                
                
               


from collections import defaultdict


def default_dict_factory():
    return defaultdict(dict)


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import cauchy
from copy import deepcopy
from sklearn.cluster import KMeans
  
PopSize = 50
DimSize = 3
LB = [0] * DimSize
UB = [1] * DimSize
TrialRuns = 1
curFEs = 0
MaxFEs = DimSize * 1000

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
FuncNum = 0
BestPop = None
BestFit = float("inf")

H = 1
muF, muCr = [0.5] * H, [0.5] * H
elms=[]

# initialize the M randomly
def Initialization(func):
    global Pop, FitPop, curFEs, DimSize, BestPop, BestFit, H, muF, muCr
    H = 10
    muF, muCr = [0.5] * H, [0.5] * H
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] ,t= func(Pop[i])
        # curFEs += 1
    BestFit = min(FitPop)
    BestPop = deepcopy(Pop[np.argmin(FitPop)])

def meanWL(SF, Delta):
    numer = 0
    denom = 0
    sumDelta = sum(Delta)
    for i in range(len(SF)):
        numer += Delta[i] / sumDelta * SF[i] ** 2
        denom += Delta[i] / sumDelta * SF[i]
    return numer / denom


def meanWA(SCr, Delta):
    result = 0
    sumDelta = sum(Delta)
    for i in range(len(SCr)):
        result += Delta[i] / sumDelta * SCr[i]
    return result

def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi

def PaCDE(func):
    global Pop, FitPop, LB, UB, PopSize, DimSize, curFEs, BestPop, BestFit, curFEs, muF, muCr, H

    F_idx, Cr_idx = np.random.randint(H), np.random.randint(H)
    SF, SCr, Delta = [], [], []

    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    for i in range(PopSize):
        IDX = np.random.randint(0, PopSize)
        while IDX == i:
            IDX = np.random.randint(0, PopSize)
        candi = list(range(0, PopSize))
        candi.remove(i)
        candi.remove(IDX)
        r1, r2 = np.random.choice(candi, 2, replace=False)

        F = np.clip(np.random.normal(muF[F_idx], 0.1), 0, 1)
        Cr = np.clip(np.random.normal(muCr[Cr_idx], 0.1), 0, 1)
        if FitPop[IDX] < FitPop[i]:  # DE/winner-to-best/1
            Off[i] = Pop[IDX] + F * (BestPop - Pop[IDX]) + F * (Pop[r1] - Pop[r2])
        else:
            Off[i] = Pop[i] + F * (BestPop - Pop[i]) + F * (Pop[r1] - Pop[r2])
        jrand = np.random.randint(0, DimSize)  # bin crossover
        for j in range(DimSize):
            if np.random.rand() < Cr or j == jrand:
                pass
            else:
                Off[i][j] = Pop[i][j]

        for j in range(DimSize):
            if Off[i][j] < LB[j] or Off[i][j] > UB[j]:
                Off[i][j] = np.random.uniform(LB[j], UB[j])

        FitOff[i] ,t= func(Off[i])
        curFEs += 1
        if FitOff[i] < FitPop[i]:
            SF.append(F)
            SCr.append(Cr)
            Delta.append(FitPop[i] - FitOff[i])
            Pop[i] = deepcopy(Off[i])
            FitPop[i] = FitOff[i]
            if FitOff[i] < BestFit:
                BestFit = FitOff[i]
                BestPop = deepcopy(Off[i])

    if len(SF) == 0:
        pass
    else:
        muF[F_idx] = 0.9 * muF[F_idx] + 0.1 * meanWL(SF, Delta)
        muCr[Cr_idx] = 0.9 * muCr[Cr_idx] + 0.1 * meanWA(SCr, Delta)

    if curFEs > 0.8 * MaxFEs:
        tau = 1 / np.sqrt(DimSize)
        sigma = np.exp(tau * np.random.rand())
        for i in range(PopSize):
            Tmp = BestPop + sigma * np.random.uniform(-1, 1, DimSize)
            for j in range(DimSize):
                if Tmp[j] < LB[j] or Tmp[j] > UB[j]:
                    Tmp[j] = np.random.uniform(LB[j], UB[j])
            FitTmp ,t= func(Tmp)
            curFEs += 1
            if FitTmp < BestFit:
                BestFit, BestPop = FitTmp, deepcopy(Tmp)
                sigma *= np.exp(tau * np.random.rand())
            else:
                sigma *= np.exp(-tau * np.random.rand())
    idx = np.argmin(FitPop)
    Pop[idx], FitPop[idx] = deepcopy(BestPop), BestFit
    
    
    

parent_dir = os.path.dirname(os.getcwd())
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本文件的父文件夹
cwd_dir = os.path.dirname(script_path)
device = torch.device('cuda:0')

model1=SwinTransformerModel().to(device)
model2= Alexnet().to(device)
model3=Mobilenet_v2().to(device)
epochs=20

val_loader=[]
p1= np.array([])
p2=np.array([])
p3=np.array([])
def main():
    import numpy as np
    global x_train, y_train, x_val, y_val, device,elms,model1,model2,model3,val_loader
    lr = 1e-3
    batchsz = 64

    parent_dir = os.path.dirname(os.getcwd())
    
    script_path = os.path.abspath(__file__)
    
    cwd_dir = os.path.dirname(script_path)
    
    result = defaultdict(default_dict_factory)
    model_name = ["SwinTransformerModel", "alexnet_model","mobilenet_v2_model"]
 
    epochs=21#加载运行了epochs的模型
    for index  in range(len(model_name)): 
        
        All_Trial_Best = []
        elm_acc = []
        model_acc=[]

        All_test_lable = []
        All_val_lable = []

        All_model_test_lable = []
        All_model_val_lable = []
        tag=0
        for ii in range(0, 5):
            val_loader=[]
            p1= np.array([])
            p2=np.array([])
            p3=np.array([])
            set_seed(42+index+ii )
             
             
            set_seed(42+index+ii )            

            print(f"执行模型{model_name[index]}:第{ii}次 -------------")
            
            filemame = f"images{0}.csv"
            model=SwinTransformerModel().to(device)
            train_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='train')
            val_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='val')
            test_db = Pokemon(cwd_dir + '/data', filemame, 224, mode='test')
            
            indices = np.arange(len(train_db))
            indices1 = np.arange(len(val_db))
            indices2 = np.arange(len(test_db))
           
            sampler = SubsetRandomSampler(indices)
            # 在DataLoader中设置generator参数
            train_loader = DataLoader(train_db, batch_size=batchsz,  sampler=sampler)
            val_loader = DataLoader(val_db, batch_size=batchsz, shuffle=False)
            test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=False)
             
             
            model.load_state_dict(torch.load(f'{cwd_dir}/{model_name[index]}/{str(epochs)}/50dim/best{ii}.mdl'))
            model1.load_state_dict(torch.load(f'{cwd_dir}/SwinTransformerModel/{str(epochs)}/50dim/best{ii}.mdl'))
            model2.load_state_dict(torch.load(f'{cwd_dir}/alexnet_model/{str(epochs)}/50dim/best{ii}.mdl'))
            model3.load_state_dict(torch.load(f'{cwd_dir}/mobilenet_v2_model/{str(epochs)}/50dim/best{ii}.mdl'))

            # 存储生成的特征的路径，下次get_features就直接读取现成的特征文件
            file_path = os.path.join(cwd_dir, f'data/x_train{ii}_{model_name[index]}_50dim.pkl')
            file_path1 = os.path.join(cwd_dir, f'data/y_train{ii}_{model_name[index]}_50dim.pkl')
            file_path2 = os.path.join(cwd_dir, f'data/x_val{ii}_{model_name[index]}_50dim.pkl')
            file_path3 = os.path.join(cwd_dir, f'data/y_val{ii}_{model_name[index]}_50dim.pkl')
            file_path4 = os.path.join(cwd_dir, f'data/test{ii}_{model_name[index]}_50dim.pkl')
            file_path5 = os.path.join(cwd_dir, f'data/test_y{ii}_{model_name[index]}_50dim.pkl')

            train, train_y = get_features(model, train_loader, file_path, file_path1,model_name[index])
            val, val_y = get_features(model, val_loader, file_path2, file_path3,model_name[index])
            test, test_y = get_features(model, test_loader, file_path4, file_path5,model_name[index])

            
            x_train = train
            y_train = train_y
            x_val, y_val = val, val_y
            test, test_y = test, test_y
            
            from keras.utils import to_categorical
            y_train = to_categorical(y_train, 3)
            y_test = to_categorical(test_y, 3)
            y_val = to_categorical(val_y, 3)
            
            dimensions = 3            
            lb = np.ones(dimensions)*0
            ub = np.ones(dimensions)*1
            
            global curFEs, curFEs, TrialRuns, Pop, FitPop, DimSize,LB,UB,elms
            DimSize = dimensions
            LB =lb
            UB =ub
            import sys
            # 确保输出到标准输出而不是关闭的文件
            sys.stdout = sys.__stdout__
            
            All_Trial_Best = []
            MAX = 0
            for i in range(TrialRuns):
                BestList = []
                curFEs = 0
                np.random.seed(2000 + 23 * i)
                Initialization(softvote)
                BestList.append(min(FitPop))
                while curFEs < MaxFEs:
                    PaCDE(softvote)
                    BestList.append(min(FitPop))

                    min_index=np.argmin(FitPop)

                    _,elms=softvote(Pop[min_index])
                    print("bestfit=", min(FitPop))

                MAX = max(len(BestList), MAX)
                All_Trial_Best.append(BestList)
            for i in range(len(All_Trial_Best)):
                for j in range(len(All_Trial_Best[i]), MAX):
                    All_Trial_Best[i].append(All_Trial_Best[i][-1])
            
            val_acc,val_y_pred =getvote(elms,val_loader,model1,model2,model3,y_val)            
            acc ,y_pred  =getvote(elms,test_loader,model1,model2,model3,y_test)           
            elm_acc.append( acc)

            All_test_lable.append(y_pred)
            All_val_lable.append(val_y_pred)
 
            print(f"inter:test_acc={ acc}")                 
            print(f"inter:val_acc={val_acc}")

            
    



def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    else:
        return d






def getvote(X,loader,model1,model2,model3,y_val):


   
    p1 = get_evaluate_acc_pred(model1,loader)
    p2 = get_evaluate_acc_pred(model2, loader)
    p3 = get_evaluate_acc_pred(model3, loader)


    pred_prob1 = torch.tensor(p1, dtype=torch.float32)
    pred_prob2 = torch.tensor(p2, dtype=torch.float32)
    pred_prob3 = torch.tensor(p3, dtype=torch.float32)  
    

    # 设置权重
    weights = [X[0], X[1], X[2]]  # 权重分别为2, 1, 3

    # 计算加权平均预测概率
    weighted_prob = (
        weights[0] * pred_prob1 +
        weights[1] * pred_prob2 +
        weights[2] * pred_prob3
    ) / sum(weights)

   
    y_pred = np.argmax(weighted_prob, axis=1)
    y = np.argmax(y_val,axis=1)
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy,y_pred
def softvote(X):

    global x_train, y_train, x_val, y_val, TZnum, concatenate_x_train, concatenate_y_train, tag, test, test_y,model1,model2,model3,val_loader,p1,p2,p3
    
    if len(p1) == 0:
        p1 = get_evaluate_acc_pred(model1, val_loader)
        p2 = get_evaluate_acc_pred(model2, val_loader)
        p3 = get_evaluate_acc_pred(model3, val_loader)


    pred_prob1 = torch.tensor(p1, dtype=torch.float32)
    pred_prob2 = torch.tensor(p2, dtype=torch.float32)
    pred_prob3 = torch.tensor(p3, dtype=torch.float32)  
    
    # 设置权重
    weights = [X[0], X[1], X[2]]  # 权重分别为2, 1, 3

    # 计算加权平均预测概率
    weighted_prob = (
        weights[0] * pred_prob1 +
        weights[1] * pred_prob2 +
        weights[2] * pred_prob3
    ) / sum(weights)

    
    y_pred = np.argmax(weighted_prob, axis=1)
    y =np.argmax(y_val, axis=1)
    accuracy = accuracy_score(y, y_pred)
    # print(accuracy)
    return -1*accuracy,X

def get_features(model, train_loader, x_path, y_path,modelname):
    global device
    if (not os.path.exists(x_path)) or (not os.path.exists(y_path)):
        
        
        if modelname =='SwinTransformerModel':
           model.base.head[-1] =nn.Identity()
           model0 =model
        elif modelname =='ConvNeXtModel':
            model.base.classifier[2][-1]=nn.Identity()
            model0 =model
        else:

           model.block[3]=nn.Identity()
           model0 = model
        model0=model
        model0.eval()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                
            # 计算训练集准确率
                if modelname == 'inception_model':
                    logits = model0(x)
                    features = logits  # 获取每个样本的预测标签
                else:
                    logits = model0(x)
                    features =logits  # 获取每个样本的预测标签
                
                if step == 0:
                    result = features
                    result_y = y;
                else:
                    result = torch.cat([result, features], dim=0)
                    result_y = torch.cat([result_y, y], dim=0)
        result, result_y = result.cpu(), result_y.cpu()
        with open(x_path, 'wb') as file:
            pickle.dump(result, file)
        with open(y_path, 'wb') as file:
            pickle.dump(result_y, file)

        return result.numpy(), result_y.numpy()
    else:
        with open(x_path, 'rb') as file:
            result = pickle.load(file)
        with open(y_path, 'rb') as file:
            result_y = pickle.load(file)

        return result.numpy(), result_y.numpy()



if __name__ == '__main__':
    # generativeModel()
    main()
