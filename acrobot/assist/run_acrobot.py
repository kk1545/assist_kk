
"""ディレクトリを指定して、その指定されたディレクトリの中にあるモデルすべてを一度動かしてその実行結果を保存するプログラム
【input】：model.npz, optimizer.npz, target_model.npz
【output】:result_$ASSIST_STATUS.npy

mountaincar ステップ数が少ないほうがいい
成績の評価方法 考える必要あり？
"""

import os
import fnmatch
import gym
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import argparse #引数のためのライブラリ
import csv


#ｇｙｍがバグらないためのおまじない
gym.logger.set_level(40)

#-----------------------引数の定義---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', '-T', type=str, default='None/400', help=' example: 400ep/Assist')
parser.add_argument('--save', action='store_true')
parser.add_argument('--end_steps', type=int, default=500) #タスク変更時 ※ 最大step数

args = parser.parse_args()
#-----------------------引数の定義---------------------------------------

#--------------------モデルをロードするための箱作り----------------------------
##定数
env =   gym.make('Acrobot-v1') #タスク変更時 ※ タスク名
env._max_episode_steps = args.end_steps
gamma = 0.98
##モデル構造
class QFunction(chainer.Chain):
    def __init__(self, obs_size, hidden, n_actions):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, hidden)                                   # 入力数,中間層のノード
            self.l1 = L.Linear(hidden, hidden)                                     # 中間層,中間層のノード        
            self.l2 = L.Linear(hidden, n_actions)                                  # 中間層,出力層のノード
    def __call__(self, x, test=False):
        h0 = F.tanh(self.l0(x))                                                    # 中間層の活性化関数
        h1 = F.tanh(self.l1(h0))                                                   # 中間層の活性化関数
        h2 = self.l2(h1)                                                           # 出力層の活性化関数(恒等関数)
        return chainerrl.action_value.DiscreteActionValue(h2)                      # 深層強化学習関数
# DQN設定
q_func     = QFunction(env.observation_space.shape[0], 50, env.action_space.n)     # 入力数2, 中間層50, 出力層3
opt        = chainer.optimizers.Adam(eps=1e-2)                                     # 最適化関数
opt.setup(q_func)
explorer   = chainerrl.explorers.LinearDecayEpsilonGreedy \
            (start_epsilon=1, end_epsilon=0.01, decay_steps=300, random_action_func=env.action_space.sample) # ε-greedy法
ex_rep     = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)                # 経験再生(experience replay)
phi        = lambda x: x.astype(np.float32, copy=False)
agent      = chainerrl.agents.DQN(q_func, opt, ex_rep, gamma, explorer, phi=phi)   # 深層強化学習
#--------------------モデルをロードするための箱作り----------------------------



#全体のループを決めるために、ターゲットとなるディレクトリに何ファイルあるのかを確認する。

if __name__ == "__main__":
    #全体のループを決めるために、ターゲットとなるディレクトリに何ファイルあるのかを確認する。
    catch_dir_name = args.target_dir    #catch_dir_name example:400ep/Assist/
    target_dir = './output_acrobot/'+catch_dir_name+'/'
    dir_in_count = len(fnmatch.filter(os.listdir(target_dir), "acrobot_agent*"))

    #このモデル郡のプレイ結果を保存する配列1
    result = []
    vares = []
    stdes = []
    covs = []
    
clear_model_count = 0 #クリアしたモデル数

for i in range(dir_in_count):
    #何ステップ倒立できたかをカウントする
    step_count = 0  #１ループの最後にresultにappendする
    
    #モデルのロード
    AGENT_NAME = target_dir+'acrobot_agent-n'+str(i)
    agent.load(AGENT_NAME)

    l2_list = agent.q_function['l2'].W._data#l2の重みについて
    vares.append(np.var(l2_list))
    stdes.append(np.std(l2_list))
    covs.append(np.cov(l2_list[0][0],l2_list[0][1])[0][1])

    #タスクのリセット
    observ = env.reset()

    while True:
        action = agent.act(observ)
        observ, reward, done, _ = env.step(action)
        step_count += 1

        if done == True:
            if step_count < 500.0:
                clear_model_count += 1
            break
    #今回のプレイのスコアを保存
    print(f'{i}世代では、{step_count}でした')
    result.append(step_count)

#１ディレクトリの総評
#clear_count = np.count_nonzero(result >= 200)
#clear_count = result.count(200)
print(f"100step以内にゴールできたモデルは{sum(x<100 for x in result)}モデルでした")
print(f"150step以内にゴールできたモデルは{sum(100<=x<150 for x in result)}モデルでした")
print(f"200step以内にゴールできたモデルは{sum(150<=x<200 for x in result)}モデルでした")
print(f"250step以内にゴールできたモデルは{sum(200<=x<250 for x in result)}モデルでした")
print(f"300step以内にゴールできたモデルは{sum(250<=x<300 for x in result)}モデルでした")
print(f"350step以内にゴールできたモデルは{sum(300<=x<350 for x in result)}モデルでした")
print(f"400step以内にゴールできたモデルは{sum(350<=x<400 for x in result)}モデルでした")
print(f"450step以内にゴールできたモデルは{sum(400<=x<450 for x in result)}モデルでした")
print(f"499step以内にゴールできたモデルは{sum(450<=x<499 for x in result)}モデルでした")
print(f"500step以内にゴールできなかったモデルは{sum(499<=x<=500 for x in result)}モデルでした")


ave = sum(result) / len(result)
print(f'平均値:{ave}')
print(f'標準偏差：{np.std(result)}')

#result.append(clear_count)

#セーブについて
if args.save:    #saveするならば
    split_name = catch_dir_name.split('/')
    np.save('./output_acrobot/'+split_name[0]+'/'+split_name[1], result)
else:
    print(result)
    
print(f'分散の平均値:{np.mean(vares)}')
print(f'標準偏差の平均値:{np.mean(stdes)}')
print(f'共分散の平均値:{np.mean(covs)}')

#csvファイルに結果を保存
with open('run_acrobot_result.csv', 'w') as f:
    writer = csv.writer(f)
    for cnt in range(24):
        print(f"{(cnt+1)*20}step未満 : {sum(cnt*20 <= x < (cnt+1)*20 for x in result)}")
        writer.writerow([(cnt+1)*20, sum(cnt*20 <= x < (cnt+1)*20 for x in result)])
    print(f"499step未満 : {sum(480 <= x < 499 for x in result)}")
    writer.writerow([499, sum(480 <= x < 499 for x in result)])
    print(f"500step : {sum(x >= 499 for x in result)}")
    writer.writerow([500, sum(x >= 499 for x in result)])
    writer.writerow(["クリアしたモデル数 : ", clear_model_count])
