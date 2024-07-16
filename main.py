#main_bert_jikken_copus_cls.py の 7/16 1:31のファイルがベース

import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

#----------------
import torch
import numpy as np
import pandas as pd
import transformers
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from transformers import BertTokenizer


from transformers import BertModel
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import json

#-----------



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.question = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            self.question = process_text(question)
            words = self.question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)
        
#        print("question\n",self.idx2question) #add roomae for debug
        answer_copus = pd.read_csv("data_annotations_class_mapping.csv")
        self.answer2idx = dict(zip(answer_copus["answer"], answer_copus["class_id"]))
        self.idx2answer = {v: k for k, v in self.answer2idx.items()}

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)
            
#        print("andser\n",self.answer2idx) #add roomae for debug

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
#        print("check1")
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
#----------------------------------------
#        question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
#        question_words = self.df["question"][idx].split(" ")
#        for word in question_words:
#            try:
#                question[self.question2idx[word]] = 1  # one-hot表現に変換
#            except KeyError:
#                question[-1] = 1  # 未知語
#        
#        print("\n size, question_word, question", question_words,question) #add roomae for debug
#----------------------------------------
#        print("check2")

#----BERT ---- Tokenize the question-------
#        question_words = self.df["question"][idx].split(" ")
#        inputs = self.tokenizer(question_words, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        inputs = self.tokenizer(self.df["question"][idx], return_tensors='pt',padding='max_length', truncation=True, max_length=512)
        #print("\n inputs=",inputs)
        # Ensure the tensors are in the correct format for the DataLoader
        input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = inputs['attention_mask'].squeeze(0)  # Remove batch dimension
#----BERT ---- Tokenize the question-------

#        print("check3", image.size())

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

#            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)
#            print("check4", input_ids.size(),torch.Tensor(answers).size())

            return image, input_ids, attention_mask, torch.Tensor(answers), int(mode_answer_idx) # add roomae
#            print("size check1=",	image.shape,  torch.Tensor(answers).shape, int(mode_answer_idx))
#            return image, inputs, torch.Tensor(answers), int(mode_answer_idx) # add roomae

        else:
#            return image, torch.Tensor(question) 
#            print("check5")

#            return image, input_ids, attention_mask # add roomae
#           print("size check1=",	inputs.shape)
            return image, input_ids, attention_mask # add roomae

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
#                print("\n i,j,pred,ans", i,j,pred,answers[j]) #add roomae for debug
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
        
#        print("\n pred,ans", pred,answers) #add roomae for debug

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

#modelをついか---------
import timm

class EfficientNet_V2(nn.Module):
    def __init__(self, n_out):
        super(EfficientNet_V2, self).__init__()
        
#        model_name = "mobilenetv3_large_100.miil_in21k"
#        model_name = "tf_efficientnetv2_m_in21ft1k"
        model_name = "tf_efficientnetv2_s_in21ft1k"
#        model_name = "tf_efficientnetv2_b3.in1k"
        n_classes = 512

        #モデルの定義
        self.effnet = timm.create_model(model_name, pretrained=True)
        
        #最終層の再定義
        self.effnet.classifier = nn.Linear(self.effnet.conv_head.out_channels, n_out)

    def forward(self, x):
        return self.effnet(x)

#-----------------------



class VQAModel(nn.Module):
#    def __init__(self, vocab_size: int, n_answer: int):
    def __init__(self, n_answer: int): #add roomae
        super().__init__()
#        self.resnet = ResNet18()
        self.resnet = EfficientNet_V2(512) #add roomae
#        self.text_encoder = nn.Linear(vocab_size, 512)
        self.bert = BertModel.from_pretrained('bert-base-uncased',torch_dtype=torch.float32, attn_implementation="sdpa") #add roomae

#        self.fc = nn.Sequential(
#            nn.Linear(512 + 768, 512),
#            nn.ReLU(inplace=True),
#            nn.Linear(512, n_answer)
#        )
#   OR

        # Classifier 
        # add roomae------------------------------------------------------↓
        self.fc = nn.Sequential(
            #nn.Linear(512 + 768, 512),
            nn.Linear(512 + 768, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_answer)
        )
        # add roomae------------------------------------------------------↑


#    def forward(self, image, question):
    def forward(self, image, input_ids, attention_mask): #add roomae
#    def forward(self, image, question): #add roomae
        #image_feature = self.resnet(image)  # 画像の特徴量

#        image.requires_grad_(True)  # 画像データに対してrequires_gradをTrueに設定
#        image_feature = checkpoint(self.resnet,image,use_reentrant=True)  # 画像の特徴量
        image_feature = checkpoint(self.resnet,image)  # 画像の特徴量

#        question_feature = self.text_encoder(question)  # テキストの特徴量
#------ Bert-------
#        input_ids = question['input_ids'].squeeze(0)  # Remove batch dimension
#        attention_mask = question['attention_mask'].squeeze(0)  # Remove batch dimension
#        print("input_ids shape:", input_ids.shape) #add roomae for debug
#        print("attention_mask shape:", attention_mask.shape) #add roomae for debug

        outputs = checkpoint(self.bert,input_ids,attention_mask) # add roomae
#        outputs = self.bert(input_ids,attention_mask=attention_mask) # add roomae
#        outputs = self.bert(**question) # add roomae
        question_features = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, 768) add roomae
#------ Bert-------
        
        x = torch.cat([image_feature, question_features], dim=1)
#        print("type=", image_feature.shape,question_features.shape, x.shape)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    print("step-6-6")
#    for image, question, answers, mode_answer in dataloader:
#        print("step-5")
#        image, question, answers, mode_answer = \
#            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
#
#        pred = model(image, question)

    for image, input_ids, attention_mask, answers, mode_answer in dataloader: #add roomae
#    for image, question, answers, mode_answer in dataloader: #add roomae
#        print("step-5")
#        print("size check2=",	image.shape, answers.shape, mode_answer)
#        print("size check2-2", question)

        image, input_ids, attention_mask, answers, mode_answer = image.to(device), input_ids.to(device), attention_mask.to(device), answers.to(device), mode_answer.to(device) #add roomae
#        image, question, answers, mode_answer = image.to(device), question.to(device), answers.to(device), mode_answer.to(device) #add roomae

        with torch.cuda.amp.autocast():
        	pred = model(image, input_ids, attention_mask)
        	loss = criterion(pred, mode_answer.squeeze())
#        	print("train-pred",pred)
#        for jj in range(len(pred[0])):
#            print(",",len(pred[0]),jj,pred[0][jj])

#        pred = model(image, question) #add roomae
#       loss = criterion(pred, mode_answer.squeeze())
#        print("\n pred=",pred.size(), pred)
#        print("\n mode_answer.squeeze()=", mode_answer.squeeze())

#        print("step-5-2")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
#        print("pred, answers",pred.argmax(1),answers)

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    print("into eval")
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    # dataloader / model
    transformX = transforms.Compose([
        transforms.Resize((224, 224)),
#        transforms.RandomHorizontalFlip(p=0.30),
#        transforms.RandomVerticalFlip(p=0.30),
#        transforms.RandomCrop((224,224)),
        #transforms.RandomRotation(degrees=(-180, 180)),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor()
    ])
    
    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
#    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
#    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    train_dataset = VQADataset(df_path="/content/data/train.json", image_dir="/content/data/train", transform=transformX)
    test_dataset = VQADataset(df_path="/content/data/valid.json", image_dir="/content/data/valid", transform=transform, answer=False)

    test_dataset.update_dict(train_dataset)

#    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
#    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    #高速化対応で、num_workers=2, pin_memory=Trueを入れる。
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=2, pin_memory=True)

    print("step-1")
#    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)
    model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device) #add roomae
    print("n_out",len(train_dataset.answer2idx))
    print("step-2")
#    print(model) # add roomae for debug

    # optimizer / criterion
    num_epoch = 30
#    num_epoch = 14
    
    print(num_epoch)
    criterion = nn.CrossEntropyLoss()
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) #add roomae
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) #add roomae
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) #add roomae
#    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # train model
    print("step-3")
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    print("step-4")
    model.eval()
    submission = []
#    for image, question in test_loader:
#        image, question = image.to(device), question.to(device)
#        pred = model(image, question)
#        pred = pred.argmax(1).cpu().item()
#        submission.append(pred)
    for image, input_ids, attention_mask in test_loader: #add roomae
        image, input_ids, attention_mask = image.to(device), input_ids.to(device), attention_mask.to(device)
        pred = model(image, input_ids, attention_mask)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)
#        print(pred)
    print("step-7")
    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    print(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
