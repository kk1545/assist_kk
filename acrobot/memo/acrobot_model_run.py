#パッケージのimport
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import gym

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import csv

middle = 32

model = nn.Sequential()
model.add_module('fc1', nn.Linear(2, middle))
model.add_module('relu1', nn.ReLU())
model.add_module('dropout', nn.Dropout())
model.add_module('fc2', nn.Linear(middle, middle))
model.add_module('relu2', nn.ReLU())
model.add_module('dropout', nn.Dropout())
model.add_module('fc3', nn.Linear(middle, 3))

#model = torch.load('./cartpole_model/test.pth')
#model.load_state_dict(torch.load("./cartpole_model/test.pth"))

#print(model.state_dict())

# 環境の作成
env = gym.make('MountainCar-v0')

# モデルのテスト
model_count = 100
clear_model_count = 0

result = []

for num in range(model_count):
    model.load_state_dict(torch.load("./mountaincar_model/test" + str(num) + ".pth"))
    state = env.reset()
    model.eval()
    point = 0

    for i in range(200):
        
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0)
        action = model(state).max(1)[1].view(1, 1)

        # １ステップ実行
        state, rewards, done, _ = env.step(action.item())
        #state, rewards, done, _ = env.step(0)
        point += rewards 
        #frames.append(env.render(mode='rgb_array'))

        # エピソード完了判定
        if done:
            print(str(num) + "モデル : " + str(point) + "ポイント : ステップ数 " + str(i))
            if point != -200.0:
                clear_model_count += 1

            result.append(i)

            break

    # 環境のクローズ
    #display_frames_as_gif(frames)
    env.close()

#print("クリアしたモデル数 : " + str(clear_model_count))

"""
print(f"20step未満  : {sum(0 <= x < 20 for x in result)}")
print(f"40step未満  : {sum(20 <= x < 40 for x in result)}")
print(f"60step未満  : {sum(40 <= x < 60 for x in result)}")
print(f"80step未満  : {sum(60 <= x < 80 for x in result)}")
print(f"100step未満 : {sum(80 <= x < 100 for x in result)}")
print(f"120step未満 : {sum(100 <= x < 120 for x in result)}")
print(f"140step未満 : {sum(120 <= x < 140 for x in result)}")
print(f"160step未満 : {sum(140 <= x < 160 for x in result)}")
print(f"180step未満 : {sum(160 <= x < 180 for x in result)}")
print(f"200step未満 : {sum(180 <= x < 200 for x in result)}")
"""

with open('mountaincar_result.csv', 'w') as f:
    writer = csv.writer(f)
    for cnt in range(9):
        print(f"{(cnt+1)*20}step未満 : {sum(cnt*20 <= x < (cnt+1)*20 for x in result)}")
        writer.writerow([(cnt+1)*20, sum(cnt*20 <= x < (cnt+1)*20 for x in result)])
    print(f"199step未満 : {sum(180 <= x < 199 for x in result)}")
    writer.writerow([199, sum(180 <= x < 199 for x in result)])
    print(f"200step : {sum(x == 199 for x in result)}")
    writer.writerow([200, sum(x == 199 for x in result)])
    writer.writerow(["クリアしたモデル数 : ", clear_model_count])
