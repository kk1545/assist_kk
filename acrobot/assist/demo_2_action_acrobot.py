"""
2?
CartPoleのデモデータからアシストNNを作成するためのプログラム
模倣学習の部分？
input : ./output_mountaincar/demo/132/acrobot_demo.npz
        ./output_mountaincar/demo/147/acrobot_demo.npz
        ./output_mountaincar/demo/177/acrobot_demo.npz

output : ./output_mountaincar/demo_assist_data_25.net
"""
#フォルダ作成用
import os

#入力データ編集
import numpy as np

#学習用データ作成用
from sklearn.model_selection import train_test_split

#モデル作成用
import chainer

import chainer.links as L   #パラメータを持つリンク層　Denceなど
import chainer.functions as F   #パラメータを持たないファンクション層　シグモイドやReLu
from chainer import Sequential

#利便性用
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', action='store_true', help='No 引数')
args = parser.parse_args()

# パラメータ
n_input = 6 #入力
n_hidden = 48   #任意
n_output = 3    #出力
lr = 0.01   #学習率
n_epoch = 1000    #学習回数
n_batchsize = 32 #計算回数

#データ格納用の変数の作成
    # ログの保存用
results_train = {#学習時
    'loss': [],
    'accuracy': []
}

results_valid = {#評価時
    'loss': [],
    'accuracy': []
}

def Make_Data():
    '''ファイルを読み込んで入力データと正解ラベルを作成する関数 戻り値としてcahinerに渡せるXとTが渡される。'''
    #(1)データの読み込み
        #読む込むファイルの選択
    FILE_NAME = './output_acrobot/demo/132/acrobot_demo.npz'
    FILE_NAME2 = './output_acrobot/demo/147/acrobot_demo.npz'
    FILE_NAME3 = './output_acrobot/demo/177/acrobot_demo.npz'
    #ちょっと精度が悪いアシストデータを読み込む
    #./output/demo/30 ← アシストデータのステップ数
    #FILE_NAME = './output_mountaincar/demo/104/mountaincar_demo.npz'
    #FILE_NAME2 = './output_mountaincar/demo/154/mountaincar_demo.npz'
    #FILE_NAME3 = './output_mountaincar/demo/167/mountaincar_demo.npz'

        #アシストデータを読み込む
    demo = np.load(FILE_NAME)
    demo2 = np.load(FILE_NAME2)
    demo3 = np.load(FILE_NAME3)


    #(2)データの編集　学習に使うすべてのデータを作成する
        #必要な部分だけ取り出す
    X1 = np.array(list(demo.get('obs')))   #入力データ
    T1 = np.array(list(demo.get('actions')))   #正解ラベル

        #別のデータを追加する1
    X2 = np.array(list(demo2.get('obs'))) #入力データ
    T2 = np.array(list(demo2.get('actions')))   #正解ラベル

        #別のデータを追加する2
    X3 = np.array(list(demo3.get('obs'))) #入力データ
    T3 = np.array(list(demo3.get('actions')))   #正解ラベル

        #合体させる
    X = np.concatenate([X1, X2, X3])
    T = np.concatenate([T1, T2, T3])


        #変形する
    T = np.reshape(T, (len(T), ))
    ''' 変形後の出力
    print('x:', X.shape)
    print('t:', Y.shape)
    -------------------------
    x: (296, 4)
    t: (296,)
    '''
        #chainerに形を合わせる
    x = X.astype('float32')
    t = T.astype('int32')

    return x, t

def Make_Model():

    # (4)　ネットワークの作成　全結合層が3つ、活性化関数にReLU関数を持つネットワークを作成する
    # 入力次元数が 3、出力次元数が 2 の全結合層の場合の書き方
    #l = L.Linear(3, 2)

    # net としてインスタンス化
    net = Sequential(
        L.Linear(n_input, n_hidden), F.relu, F.dropout,
        L.Linear(n_hidden, n_hidden), F.relu, F.dropout,
        L.Linear(n_hidden, n_output)
    )

        #最適化法の定義
    #optimizer = chainer.optimizers.SGD(lr=lr)
    optimizer = chainer.optimizers.Adam()

        #Adamによって更新されるようにする　ネットワークに最適化法をセットアップ
    optimizer.setup(net)

    return net, optimizer

def Change_Mode(checker, input_data ,label):

    if checker:
        print('全データを使って学習のみ行います。')
        
        #入力データ
        x = input_data  #すべての入力データを学習に用いる
        t = label   #すべての正解ラベルを学習用に用いる

        #評価データの作成
        x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.3, random_state=0)


         #モデルの呼び出し
        net, optimizer = Make_Model()  #モデルの作成

        iteration = 0  #イテレーション回数

                #学習のループ
        for epoch in range(n_epoch):#全体の学習会回数

            # データセット並べ替えた順番を取得
            order = np.random.permutation(range(len(x)))#学習用

            # 各バッチ毎の目的関数の出力と分類精度の保存用　学習時
            loss_list = []
            accuracy_list = []

            for i in range(0, len(order), n_batchsize):#一回の学習 
                # バッチを準備
                index = order[i:i+n_batchsize]
                x_train_batch = x[index,:]
                t_train_batch = t[index]

                # 予測値を出力
                y_train_batch = net(x_train_batch)

                
                # 目的関数を適用し、分類精度を計算
                loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
                accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

                loss_list.append(loss_train_batch.array)
                accuracy_list.append(accuracy_train_batch.array)

                # 勾配のリセットと勾配の計算
                net.cleargrads()
                loss_train_batch.backward()

                # パラメータの更新
                optimizer.update()

                # カウントアップ
                iteration += 1

            # 訓練データに対する目的関数の出力と分類精度を集計 全体の７割を使って行った学習に対する評価
            loss_train = np.mean(loss_list)
            accuracy_train = np.mean(accuracy_list)


            # 1エポック終えたら、検証データで評価
            # 検証データで予測値　val＿loss
            #with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                #y_val = net(x_val)

            # 目的関数を適用し、分類精度を計算
            #loss_val = F.softmax_cross_entropy(y_val, t_val)
            #accuracy_val = F.accuracy(y_val, t_val)

            # 結果の表示
            #print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
                #epoch, iteration, loss_train, loss_val.array))

            # ログを保存
            results_train['loss'] .append(loss_train)
            results_train['accuracy'] .append(accuracy_train)
            #results_valid['loss'].append(loss_val.array)
            #results_valid['accuracy'].append(accuracy_val.array)

        #(6)テストデータを用いて評価をする
        # テストデータで予測値を計算
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y_test = net(x_test)

        accuracy_test = F.accuracy(y_test, t_test)
        #print(accuracy_test.array)

        #(7)モデルを保存する
        chainer.serializers.save_npz('./output_acrobot/demo_assist_data_25.net', net)
        print('モデルを保存しました。')

    else:
        print('全データを三分割して学習します。')
        #(3)データの分割
            #学習用と評価用に分ける
        x = input_data
        t = label
        x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

            #学習用を学習用と学習時評価用に分ける
        x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0) 

        #モデルの呼び出し
        net, optimizer = Make_Model()  #モデルの作成

        iteration = 0  #イテレーション回数

                #学習のループ
        for epoch in range(n_epoch):#全体の学習会回数

            # データセット並べ替えた順番を取得
            order = np.random.permutation(range(len(x_train)))#学習用

            # 各バッチ毎の目的関数の出力と分類精度の保存用　学習時
            loss_list = []
            accuracy_list = []

            for i in range(0, len(order), n_batchsize):#一回の学習 
                # バッチを準備
                index = order[i:i+n_batchsize]
                x_train_batch = x_train[index,:]
                t_train_batch = t_train[index]

               
                # 予測値を出力
                y_train_batch = net(x_train_batch)

                # 目的関数を適用し、分類精度を計算
                loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
                accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

                loss_list.append(loss_train_batch.array)
                accuracy_list.append(accuracy_train_batch.array)

                # 勾配のリセットと勾配の計算
                net.cleargrads()
                loss_train_batch.backward()

                # パラメータの更新
                optimizer.update()

                # カウントアップ
                iteration += 1

            # 訓練データに対する目的関数の出力と分類精度を集計 全体の７割を使って行った学習に対する評価
            loss_train = np.mean(loss_list)
            accuracy_train = np.mean(accuracy_list)


            # 1エポック終えたら、検証データで評価
            # 検証データで予測値　val＿loss
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y_val = net(x_val)

            # 目的関数を適用し、分類精度を計算
            loss_val = F.softmax_cross_entropy(y_val, t_val)
            accuracy_val = F.accuracy(y_val, t_val)

            # 結果の表示
            #print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
            #    epoch, iteration, loss_train, loss_val.array))

            # ログを保存
            results_train['loss'] .append(loss_train)
            results_train['accuracy'] .append(accuracy_train)
            results_valid['loss'].append(loss_val.array)
            results_valid['accuracy'].append(accuracy_val.array)

        #(6)テストデータを用いて評価をする
        # テストデータで予測値を計算
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            #print(x_test)
            #print(x_test.shape)
            y_test = net(x_test)

        accuracy_test = F.accuracy(y_test, t_test)
        #print(accuracy_test.array)

        #(7)モデルを保存する
        chainer.serializers.save_npz('./output_acrobot/demo_assist_data_25.net', net)
        print('モデルを保存しました。')

if __name__ == "__main__":
    
    x, t = Make_Data()  #入力データと正解ラベルの導出
    check_mode = args.mode  #フル学習モードか通常学習モードか切り替え用変数　Trueが全データ　Falseがバリデーション
    Change_Mode(checker=check_mode, input_data=x, label=t)  #データを利用して学習を行う