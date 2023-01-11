"""
1
タスクを人間が手動で操作するファイル
output : ./output/cartpole_demo
"""

import random
import pyglet
import gym
import time
from pyglet.window import key
from stable_baselines.gail import generate_expert_traj
#from baselines.common.atari_wrappers import *
#ｇｙｍがバグらないためのおまじない
gym.logger.set_level(40)

ENV = 'Acrobot-v1'




#環境の生成
env = gym.make(ENV)
#env.MaxAndSkipEnv(env, skip=4)  #4フレーム毎に行動を選択
#env = WarpFrame(env)    # 画像を84*84のグレースケールに変換
env.render()

#キーイベント用のウィンドウの生成
win = pyglet.window.Window(width=300, height=100, vsync=False)
key_handler = pyglet.window.key.KeyStateHandler()

win.push_handlers(key_handler)
pyglet.app.platform_event_loop.start()

#キー状態の取得
def get_key_state():
    key_state = set()
    win.dispatch_events()
    for key_code, pressed in key_handler.items():
        if pressed:
            key_state.add(key_code)
    return key_state

#キー入力待ち
while len(get_key_state()) == 0:
    time.sleep(1.0/30.0)

#人間のデモを収集するコールバック
def human_expert(_state):

    #キー状態の取得
    key_state = get_key_state()

    #行動の選択
    if key.RIGHT in key_state:
        action = 2
    elif key.LEFT in key_state:
        action = 0
    else:
        action = 1
    
    #スリープ
    time.sleep(1/4.5)

    #環境の描画
    env.render()

    #行動の選択
    return action

#人間のデモの収集
generate_expert_traj(human_expert, './output/acrobot_demo', env, n_episodes=1)
