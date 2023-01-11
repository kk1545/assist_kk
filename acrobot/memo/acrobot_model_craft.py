# -*- coding: utf-8 -*-
from collections import namedtuple

#パッケージのimport
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import gym

from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

#定数の設定
ENV = 'Acrobot-v1' #使用する課題名
GAMMA = 0.99 #時間割引率 可変 疎0.97
MAX_STEPS = 500 #1試行のステップ数
NUM_EPISODES = 1000 #最大試行回数
LR = 0.0001 #可変 疎0.0023


#経験を保存するメモリクラスを定義

class ReplayMemory:
    
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY #メモリの最大長さ
        self.memory = [] #経験を保存する変数
        self.index = 0 #保存するindexを示す変数
        
    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''
        
        if len(self.memory) < self.capacity:
            self.memory.append(None) #メモリが満タンでないときは足す
        
        #namedtuple の Transition を使用して、値とフィールド名をペアにして保存する
        self.memory[self.index] = Transition(state, action, state_next, reward)
        
        self.index = (self.index + 1) % self.capacity #保存するindexを1つずらす
        
    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)

# エージェントが持つ脳となるクラスです、DQNを実行します
# Q関数をディープラーニングのネットワークをクラスとして定義

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000
middle = 48 #48

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 行動数を取得

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, middle))
        self.model.add_module('relu1', nn.ReLU())
        #self.model.add_module('dropout', nn.Dropout())
        self.model.add_module('fc2', nn.Linear(middle, middle))
        self.model.add_module('relu2', nn.ReLU())
        #self.model.add_module('dropout', nn.Dropout())
        self.model.add_module('fc3', nn.Linear(middle, num_actions))

        #print(self.model)  # ネットワークの形を出力

        # 最適化手法の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR) #lr 学習率 0.0001

    def replay(self):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # -----------------------------------------
        # 1. メモリサイズの確認
        # -----------------------------------------
        # 1.1 メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < BATCH_SIZE:
            return

        # -----------------------------------------
        # 2. ミニバッチの作成
        # -----------------------------------------
        # 2.1 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 各変数をミニバッチに対応する形に変形
        # transitionsは1stepごとの(state, action, state_next, reward)が、BATCH_SIZE分格納されている
        # つまり、(state, action, state_next, reward)×BATCH_SIZE
        # これをミニバッチにしたい。つまり
        # (state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))

        # 2.3 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がBATCH_SIZE分並んでいるのですが、
        # それを torch.FloatTensor of size BATCH_SIZEx4 に変換します
        # 状態、行動、報酬、non_finalの状態のミニバッチのVariableを作成
        # catはConcatenates（結合）のことです。
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        # -----------------------------------------
        # 3. 教師信号となるQ(s_t, a_t)値を求める
        # -----------------------------------------
        # 3.1 ネットワークを推論モードに切り替える
        self.model.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t)を求める
        # self.model(state_batch)は、右左の両方のQ値を出力しており
        # [torch.FloatTensor of size BATCH_SIZEx2]になっている。
        # ここから実行したアクションa_tに対応するQ値を求めるため、action_batchで行った行動a_tが右か左かのindexを求め
        # それに対応するQ値をgatherでひっぱり出す。
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意。

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        # まずは全部0にしておく
        next_state_values = torch.zeros(BATCH_SIZE)

        # 次の状態があるindexの最大Q値を求める
        # 出力にアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてそのQ値（index=0）を出力します
        # detachでその値を取り出します
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values #可変

        # -----------------------------------------
        # 4. 結合パラメータの更新
        # -----------------------------------------
        # 4.1 ネットワークを訓練モードに切り替える
        self.model.train()

        # 4.2 損失関数を計算する（smooth_l1_lossはHuberloss）
        # expected_state_action_valuesは
        # sizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # 4.3 結合パラメータを更新する
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]　を size 1x1 に変換します

        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 0,1,2の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action
    
    def save_model(self, model_num):
        #torch.save(self.model, "./cartpole_model/test.pth")
        torch.save(self.model.state_dict(), "./acrobot_model/test" + str(model_num) + ".pth")
        #print(self.model.state_dict())

# CartPoleで動くエージェントクラスです、棒付き台車そのものになります


class Agent:
    def __init__(self, num_states, num_actions):
        '''課題の状態と行動の数を設定する'''
        self.brain = Brain(num_states, num_actions)  # エージェントが行動を決定するための頭脳を生成

    def update_q_function(self):
        '''Q関数を更新する'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''行動を決定する'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)
        
    def save(self):
        self.brain.save_model(model_n)

# 実行する環境のクラスです


class Environment:

    def __init__(self):
        self.env = gym.make(ENV)  # 実行する課題を設定
        num_states = self.env.observation_space.shape[0]  # 課題の状態数を取得
        num_actions = self.env.action_space.n  # CartPoleの行動数を取得
        self.agent = Agent(num_states, num_actions)  # 環境内で行動するAgentを生成
        
    def run(self):
        '''実行'''
        episode_10_list = np.zeros(10)  # 10試行分の立ち続けたstep数を格納し、平均ステップ数を出力に利用
        complete_episodes = 0  # 195step以上連続で立ち続けた試行数
        episode_final = False  # 最後の試行フラグ
        best_point = -500
        best_step = 500

        sparce = False #疎か密か　疎:True
        

        for episode in range(NUM_EPISODES):  # 最大試行数分繰り返す
            observation = self.env.reset()  # 環境の初期化
            #reward = torch.FloatTensor([0.0])
            point = 0
            last_step = 0

            state = observation  # 観測をそのまま状態sとして使用
            state = torch.from_numpy(state).type(torch.FloatTensor)  # NumPy変数をPyTorchのテンソルに変換
            state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

            for step in range(MAX_STEPS):  # 1エピソードのループ

                #if episode_final is True:  # 最終試行ではframesに各時刻の画像を追加していく
                    #frames.append(self.env.render(mode='rgb_array'))
                last_step += 1

                action = self.agent.get_action(state, episode)  # 行動を求める


                """
                報酬決め
                """

                # 行動a_tの実行により、s_{t+1}とdoneフラグを求める
                # actionから.item()を指定して、中身を取り出す
                observation_next, _, done, _ = self.env.step(
                    action.item())  # rewardとinfoは使わないので_にする
                
                #sparce 疎な報酬環境:True　密な報酬環境:False
                if done:
                    state_next = None

                    if step < 498:
                        point += 0
                        reward = torch.FloatTensor([0.0])
                        complete_episodes = complete_episodes + 1
                    else:
                        point += -1
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                else:
                    if sparce:
                        point += 0
                        reward = torch.FloatTensor([0.0])
                    else:
                        theta = observation_next[0]

                        if theta < 0:
                            point += 1
                            reward = torch.FloatTensor([1.0])
                        else:
                            point += -1
                            reward = torch.FloatTensor([-1.0])
                    
                    state_next = observation_next  # 観測をそのまま状態とする
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
                    state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換
                

                
                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)

                # Experience ReplayでQ関数を更新する
                self.agent.update_q_function()

                # 観測の更新
                state = state_next

                # 終了時の処理
                if done:
                    #報酬による変化を見る
                    #print('%d Episode: Finished after %d ' % (episode, step + 1))
                    #print(point)
                    if best_point < point:
                        best_point = point

                    if best_step > step:
                        best_step = step
                    break

            if episode_final is True:
                # 動画を保存と描画
                #display_frames_as_gif(frames)
                print(str(model_n+1) + "モデル : " + str(best_point) + "ポイント : ステップ数 " + str(best_step) + " : 最終ステップ " + str(last_step) + " : episode " + str(episode))
                #print(best_step)
                self.agent.save()
                break

            # 10連続で200step経ち続けたら成功
            """
            if complete_episodes >= 10 or episode == (NUM_EPISODES-2):
                #print('10回連続成功')
                episode_final = True  # この試行を描画を行う最終試行とする
            """
            if episode == (NUM_EPISODES-2):
                episode_final = True

# main クラス
model_number = 100 #作成するモデル数

for model_n in range(model_number):
    env = Environment()
    env.run()

