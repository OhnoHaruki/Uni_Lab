#ライブラリのインポート
from random import shuffle
import numpy as np
import os
from PIL import Image
import glob#塊でファイルを所得できる
import torch
import  torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1#顔検出を行うMTCNNと顔画像のベクトル化を行うInceptionResnetV1
from torchvision import transforms

from Sort_asian_face_01 import validation

#GPUの使用
USE_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#AFAD-liteがあるディレクトリ
INPUT_DIR = '/Users/oonoharuki/PyTorchで始めるAI開発/AFAD-Lite'

#Pytorchの内部を決定論的に設定する
torch.backends.cudnn.deterministic = True #Pytorchでlossが毎回変わらないように再現性をありにする
torch.backends.cudnn.benchmark = False#最適化による実行の高速化の恩恵は得られませんが、テストやデバッグ等に費やす時間を考えると結果としてトータルの時間は節約できる

#乱数の初期化
np.random.seed(0)
torch.manual_seed(0)

#今回の学習で使用する拡張ニューラルネットワーク
class FaceParameterNet(nn.Module):
    def __init__(self):
        super(FaceParameterNet, self).__init__()
        #モデルのロード
        self.base_model = InceptionResnetV1(pretrained='vggface2')
        
        #追加の全結合層
        self.age = nn.Linear(512,1)#年齢の出力用(512人いる)
        self.sex = nn.Linear(512,1)#性別の出力用
        
    def forward(self,input):
        x = self.base_model(input)#モデルを実行する
        a = F.hardtanh(self.age(x),15,75)#年齢を15~75の範囲で出力(データの中に15~75の人しかいないため)
        b = self.sex(x)#性別を出力
        return {'feature':x,'age':a,'sex':b}#モデルの結果、年齢、性別の出力を返す
    
#データセットの定義
class MyDataset(object):
    def __init__(self, valid=False):
        self.files = []#ファイル名
        self.ages = []#年齢
        self.sexs = []#性別
        self.trans = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.lambda(lambda x:x*2-1)
        ])
        #データセット内のファイルを羅列する
        for f in glob.glob(f'{INPUT_DIR},/[0-9][0-9]/*/*.jpg'):
            p = f.split('/')#ディレクトリ
            a = int(p[1])#最初のサブディレクトリ名が年齢
            s = int(p[2])-111#2番目が「111」であれば男性、「222」であれば女性
            self.files.append(f)#ファイル名
            self.ages.append(a)#年齢(15~75)
            self.sexs.append(s)#0であれば男性、1であれば女性
            
    def __getitem__(self,idx):
        img = Image.open(self.files[idx])#ファイルを読み込む
        img = img.convert('RGB')#カラー画像に変換する
        img = self.trans(img)#-1~1の範囲にTensorする
        age = torch.tensor([self.ages[idx]],dtype=torch.float32) #年齢をtensorにする
        sex = np.int64(self.sexs[idx])#性別をtensorにする
        return img,age,sex
    
    def __len__(self):
        return len(self.files)
    
#モデルの作成
model = FaceParameterNet()
model.to(USE_DEVICE)

#全結合層のみ学習する
'''
facenet_pytorchのモデルの最終層には「last_linear」という名前の全結合層と「last_bn」のBatchNoemalizeからなっているのでそれらのパラメータを所得。
そして、作成したモデルの中の「age」と「sex」から年齢と性別の出力層のパラメータを所得し、併せてSGDアルゴリズムに渡す。
'''
params = list(model.base_model.last_linear.parameters()) + \
    list(model.base_model.list_bn.parameters()) + \
    list(model.age.parameters()) + \
    list(model.sex.parameters())
    
optimizer = torch.optim.SGD(params,lr = 1e-5) #optimizerにSGDを採用する。学習率は1e-5

'''
損失関数の設定。
年齢は差の絶対値を使用するL1Loss
性別にはクラス分類用のCrossEntropyLoss
'''
loss1 = nn.L1Loss()#年齢用の損失関数
loss2 = nn.CrossEntropyLoss()#性別用の損失関数

#学習時と評価時のバッチサイズ
batch_size =32
#データ読み込みスレッド数
num_workers =2
#学習エポック数
num_epoches=1#

#データの読み込みクラス
dataset = MyDataset()
#別スレッドでデータを読み込む
data_loader = torch.utils.data.DataLoader(
    dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers
)

#学習のループ
for epoch in range(num_epoches):
    model.train()#学習モードにする
    for i,(x,age,sex) in enumerate(data_loader):#画像を読み込んでtensorにする
        #GPUを使用しているときはGPUメモリに送信する
        x = x.to(USE_DEVICE)
        age = age.to(USE_DEVICE)
        sex = sex.to(USE_DEVICE)
        
        #ニューラルネットワークを実行
        res = model(x)
        #年齢と性別の合計の損失値を求める
        losses = loss1(res['age'],age)+loss2(res['sex'],sex)
        
        #新しいバッチ分の学習を行う。
        optimizer.zero_grad()#ひとつ前の勾配をクリアする
        losses.backward()#損失値を逆伝播させる
        optimizer.step()#新しい購買からパラメータを更新する
        
        #100回ごとにvalidationを呼び出して評価スコアを表示する
        if (i+1) % 100 ==0:
            result = validation(model.base_model)
            th,sc = result[0]
            print(f'best score:{sc}, threathold{th:.2f}')
            model.train()
            
print('final score')
#最終的な評価値の表示
result = validation(model.base_model)
th,sc = result[0]
print('best score:{sc}, threathold{th:.2f}')

#最終的なモデルの保存
torch.save(model.base_model.state_dict(),'chapt06_model1.pth')
