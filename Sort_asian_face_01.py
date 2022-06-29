'''
検証用にJAFFE Datasetという、日本人女性のみの顔データセットを用いて日本人に対してどのくらい精度の高い顔認証を行えるのかを実験した。
実装にはfacenet_pytorchパッケージから顔検出を行う「MTCNN」と、顔画像のベクトル化を行う「InceptionResnetV1」を用いてコサイン類似度を求める。
その際、閾値を0.01から0.99まで99段階で行う。
その結果を「F1_score」関数で求めたスコアに従って同一人物であるか否かを判定する。
そして最も良いスコアが出た閾値とそのスコアを出力する。
'''

# 各パッケージのインポート
import numpy as np
import os
from PIL import Image
import glob#塊でファイルを所得できる
import argparse#コマンドライン引数を扱いやすくする
from sklearn.metrics.pairwise import cosine_similarity #cos類似度の計算
from sklearn.metrics import f1_score #F値を求める
import torch
from torch import utils
from facenet_pytorch import MTCNN, InceptionResnetV1#顔検出を行うMTCNNと顔画像のベクトル化を行うInceptionResnetV1

#GPUの使用
use_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#バッチサイズの指定
batch_size = 16
#データの読み込みのスレッド数
num_workers=2
#検証用データがあるディクショナリ
valid_dir = '/Users/oonoharuki/PyTorchで始めるAI開発'

# JAFFEデータで検証を行う
def validation(model):
    #JAFFEデータセットの読み込み
    '''
    JAFFEデータセットは256ピクセル四方のtiffフォーマットの画像ファイルである
    ファイル名は「<人物ID>.<感情ID>.<通し番号>.tiff」となっている。
    また、顔写真は広い範囲を写しているので顔の部分のみを切り出すため、「facenet-pytorch」に含まれる 「MTCCN」を用いる
    '''
    persons = []#人物ID
    faces = []#顔のデータ
    
    mtccn = MTCNN()
    
    #全ての画像を読み込んでTensor化する
    for f in glob.glob(f'{valid_dir}/*.tiff'):
        p = os.path.basename(f)#ファイル名
        p = p[:2]#ファイル名から人物IDを所得する
        img = Image.open(f)#ファイルの読み込み
        img = img.convert('RGB')#カラー画像にする
        img_cropped = mtccn(img)#顔部分を切り出して-1~1までのtensor型にする
        persons.append(p)#人物IDを追加
        faces.append(img_cropped)#画像のtensorを追加
        
    #モデルを検証用に変更する
    if USE_GPU:
         model.cuda()
    model.eval()
        
        #データセットのベクトル化
        
    #JAFFEデータセットの画像ベクトル一覧
    face_vectors = []
        
    #全ての画像を実行
    data_loader = utils.data.DataLoader(
        faces, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for x in data_loader:#画像を読み込んでtensorにする
        x = x.to(use_device)#GPUを使用するときはGPUに送信
            
        #1バッチを実行する
        pred = model(x)
        pred = pred.detach().cpu().numpy()
            
        #結果を保存
        for p in pred:
            face_vectors.append(p)
        
        #ここでスコアを求める
        
        #全組み合わせのcos類似度を求める
        sims = cosine_similarity(face_vectors,face_vectors)
        
        scores = {}
        #閾値を変えながらテストを行う
        for t in range(1,100,1):
            threashould = t/100#コサイン類似度の閾値
            match_pred = []#類似度から認識した結果
            match_true = []#実際の正解
            
            #全組み合わせ
            for i in range(len(face_vectors)):
                for j in range(len(face_vectors)):
                    #閾値以下なら別人、以上なら同じ人であると判定する
                    if sims[i,j] < threashould:# 閾値以下であれば
                        match_pred.append(0)
                    else:#閾値以上であれば
                        match_pred.append(1)
                        
                    #実際に同じ人であるかを確認する
                    if persons[i] != persons[j]:
                        match_true.append(0)
                    else:
                        match_true.append(1)
            scores[threashould] = f1_score(match_true,match_pred)
            
        #最も良いスコアが出た閾値を求める
        result = sorted(list(scores.items()),key=lambda x:x[1])[::-1]
        return result
    
#プログラムとして実行
if __name__ == '__main__':
    #指定されたモデルの検証を行う
    model = InceptionResnetV1(pretrained='vggface2')
    result = validation(model)
    th, sc = result[0]
    print(f'best score:{sc},threathold:{th:.2f}')