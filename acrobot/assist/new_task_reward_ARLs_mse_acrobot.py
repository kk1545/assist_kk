"""
3?

input : ./output_acrobot/demo_assist_data_25.net
output : ./output_acrobot/ + str(args.out_file_name) + '/' + str(args.episode)/acrobot_agent
"""

#引数用のライブラリ
import argparse
#アシストNN用のライブラリ
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Sequential
#DQN用のライブラリ
import chainerrl
#タスクライブラリ
import gym
gym.logger.set_level(40)
#その他
import numpy as np
import os
import logging
from logging import getLogger, StreamHandler, Formatter

#----------定数の定義----------------
ENV_NAME = 'Acrobot-v1'
env        = gym.make(ENV_NAME)
gamma = 0.99

#-----------------------引数の定義---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--episode', '-E',type=int, default=400)
parser.add_argument('--out_file_name',type=str)
parser.add_argument('--logger', action='store_true')
#parser.add_argument('--ann',type=str, default='./output_acrobot/demo_assist_alldata.net')#デフォルトが全データを使って学習させたモデルをロード 50%
parser.add_argument('--ann',type=str, default='./output_acrobot/demo_assist_data_25.net')#デフォルトが全データを使って学習させたモデルをロード 50%
#parser.add_argument('--assist', action='store_true')
parser.add_argument('--random', action='store_true')
#parser.add_argument('--arl', action='store_true')
parser.add_argument('--sparce', '-S', action='store_true')
parser.add_argument('--end_steps', type=int, default=500)

parser.add_argument('--fixed_ep', action='store_true')

parser.add_argument('--assist_steps',type=int, default=np.inf)#これによってDQNとARLとARL2&ARL2randomを区別する。
args = parser.parse_args()
#-----------------------引数の定義---------------------------------------

#------logger --------------
# logging snippet start #log levels: debug, info, warning, error, critical
_logdir = 'output_acrobot'
_logfile = os.path.join(_logdir, os.path.splitext(os.path.basename(__file__))[0]+'.log')
if os.path.exists(_logdir) == False: os.mkdir(_logdir)
_fmt_l = '%(asctime)s %(levelname)s:%(funcName)s: %(message)s'
_fmt_s = '%(message)s'
logging.basicConfig(level=logging.WARNING, format=None)
_logger = logging.getLogger(__name__)
_logger.setLevel(level=logging.INFO)
_handler1 = logging.StreamHandler()
_handler1.setFormatter(logging.Formatter(_fmt_s))
_logger.addHandler(_handler1)
_handler2 = logging.FileHandler(filename=_logfile)  #handler2はファイル出力
_handler2.setFormatter(logging.Formatter(_fmt_l)) # 通常は_fmt_l
_logger.addHandler(_handler2)
_logger.propagate = False
# ログの初期設定
logger = _logger
logging.basicConfig(level=logging.DEBUG, format=None)
# logging snippet end
#------logger ---------------
if args.logger:
    print('ログを出力します。')
    logger.setLevel(level=logging.DEBUG)
else:
    print('ログは出力されません。')

##出力するフォルダを作成するプログラム
#モデルの出力のためのディレクトリを作成する
OUT_DIR_NAME = "./output_acrobot/" + str(args.out_file_name) + '/' + str(args.episode)
os.makedirs(OUT_DIR_NAME, exist_ok=True)
print(f'{OUT_DIR_NAME}にモデルを保存します。')


def make_net(file_name):
    n_input = 6 #入力
    n_hidden = 48   #任意 derault 48
    n_output = 3    #0or１or2
    #重みをロードするための入れ物を作る
    net = Sequential(
            L.Linear(n_input, n_hidden), F.relu, F.dropout,
            L.Linear(n_hidden, n_hidden), F.relu, F.dropout,
            L.Linear(n_hidden, n_output)
        )
    FILE_NAME = file_name
    #定義したネットワークに重みを読み込む
    chainer.serializers.load_npz(FILE_NAME, net)
    return net

def check_l2_weight(q_func):
    '''q_funcを受け取ってmseを計算して出力する関数'''
    #比較用の基準となる配列を用意
    all_zero = np.zeros((1,3,50)).astype(np.float32)    #error タスク変更時 zerosの真ん中の値を変更した

    #q_funcのl2を取り出して比較しやすいように整形
    l2_list = np.array(q_func.l2.W._data).astype(np.float32)

    #mseの計算
    mse = np.array(F.mean_squared_error(all_zero, l2_list)._data)[0]

    return mse

#ネットワークを定義して、ロードした重みを読み込む
assist_net = make_net(args.ann)

#---------------------------------学習用のNNの定義---------------------------------------------------------------
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
#---------------------------------学習用のNNの定義---------------------------------------------------------------
#------------------------------DQNの設定-----------------------------------------------------------------
##基準点の作成

q_func     = QFunction(env.observation_space.shape[0], 50, env.action_space.n)


mse = check_l2_weight(q_func)#初回のmseチェック

print(f'first_mse:{mse}')

while mse <= 0.02:
    ran_seed = int(np.random.random_sample() * 10) #整数のランダム値を作り、seedを書き換えることでl2の重みの初期化を行う
    np.random.seed(ran_seed)    #seedの変更
    q_func     = QFunction(env.observation_space.shape[0], 50, env.action_space.n)#新しいネットワークの定義
    mse = check_l2_weight(q_func)#初回のmseチェック

print(f'decision_mse:{mse}')

#------------------------基準点からの誤差を初期値から計測して、指定以下ならば再度作り直す。------------

opt        = chainer.optimizers.Adam(eps=1e-2)                                     # 最適化関数
opt.setup(q_func)

#epsilonを固定する場合
if args.fixed_ep:
    explorer   = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.5, random_action_func=env.action_space.sample)
    print(explorer)
else:
    explorer   = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1, end_epsilon=0.01, decay_steps=300, random_action_func=env.action_space.sample) # ε-greedy法
    print(explorer)

ex_rep     = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)                # 経験再生(experience replay)
phi        = lambda x: x.astype(np.float32, copy=False)

if args.logger:#ログのありなし
    agent      = chainerrl.agents.DQN(q_func, opt, ex_rep, gamma, explorer, replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi, logger=logger)   # 深層強化学習ログあり
else:
    agent      = chainerrl.agents.DQN(q_func, opt, ex_rep, gamma, explorer, replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi)   # 深層強化学習ログあり
#------------------------------DQNの設定-----------------------------------------------------------------

#------擬似疎な報酬環境としてタスクの報酬を書き換える関数--------
def in_task_reward_out_new_reward(reward, args, steps, done):
    '''デフォルトで擬似疎な報酬環境なので、stepsが条件に該当したら報酬を返す'''
    logger.debug(f'現在{steps}ステップで{reward}を獲得')
    check = steps + 1

    if check < args.end_steps and done:#500step未満でタスクから報酬
        task_reward = 1.0
        logger.debug(f'---------------------タスクからの報酬を獲得しました。--------------------------')
        print('タスクから報酬を獲得しました！！！！！！')
    else:
        task_reward = 0.0
    logger.debug(f'擬似疎な報酬環境なので{steps}ステップで{task_reward}として出力')
    return task_reward

#-------ランダムにアシスト報酬を出力する関数---------
def random_assist():
    myflag = np.random.choice(['on', 'off'], p=[0.5, 0.5])
    if myflag == 'on': # onの場合、アシスト報酬を与える
        reward = 0.01
        logger.debug(f'ランダムな報酬が発動したので、アシスト報酬は{reward}になりました。')
    else: # offの場合、報酬は０になる
        reward = 0.0
    return reward

#---------アシスト報酬を発生させ、合成報酬にする関数------------
def out_new_reward(observ, action, reward):
    #リストにして入れられる形にする　状態の整形
    x = observ.astype('float32')
    #変形する   
    x = np.reshape(x, (1, 6))   #error タスク変更時 解決方法不明
    #ネットワークのモードを変更し予測する
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        assist_action = assist_net(x)#予測
    #予測値の行動に変換する
    assist_action = int(np.argmax(assist_action[0,:].array))

    #行動が同じかどうかを判断する。
    if assist_action == action:#エージェントの行動とアシストの行動が類似しているならば
        reward = 0.01
        logger.debug(f'エージェントの行動とアシスト行動は一致しました。{reward}')
    else:
        reward = reward

    return reward



for episode in range(args.episode):
    #学習ループに利用する定数の定義
    reward      = 0#前回エピソードの報酬をリセット
    reward_sum  = 0#前回エピソードの収益をリセット
    steps       = args.end_steps #1エピソードのstep上限値を設定
    observ      = env.reset()        #環境のリセット
    done        = False

    for step in range(steps):
        assist_reward = 0.0
        #状態を受け取り行動を出力する
        logger.debug(f'過去{step-1}ステップでは{reward}が報酬でした。')
        action = agent.act_and_train(observ, reward)
        observ, reward, done, _  = env.step(action)# アクション後の観測値


        #--------タスクの難易度を変更し、タスク報酬(内部報酬の出力)-----------
        if args.sparce:#擬似疎な報酬環境
            #print('疎な報酬環境で学習を行います。！！！！！！！！！！！！！')
            task_reward = in_task_reward_out_new_reward(reward=reward, args=args, steps=step, done=done)#stepのループの中で
        else:#密な報酬環境
            task_reward = reward

        #----------アシスト報酬を出力する。
        #args.assist_steps
        if args.random:#ランダムが与えられるか
            assist_reward += random_assist()
        else:
            if step < args.assist_steps:#ランダムじゃなくて、アシストができたら
                assist_reward = out_new_reward(observ, action, task_reward)
            else:
                assist_reward = 0.0
        '''
        if args.assist:#アシストがあるならば
            if args.arl and step < 100:#ARL(途中切り離しで100step未満ならば)
                assist_reward += out_new_reward(observ, action, task_reward)
            elif args.arl and step > 100:#100stepより多いならばアシスト報酬はなし
                assist_reward = 0.0
            elif args.arl != True:#step数に関係なくアシスト報酬が入る。
                assist_reward += out_new_reward(observ, action, task_reward)
        else:
            pass#ただのDQNを行う'''



                
        
        #内部報酬と外部報酬を和算する。
        logger.debug(f'task_reward:{task_reward}でassist_reward:{assist_reward}')
        reward = task_reward + assist_reward 
        logger.debug(f'現在{step}ステップでは{reward}を最終的な報酬として出力します。')

        #確定した報酬を収益に加える。
        reward_sum += reward

        if args.logger:
            logger.debug(f'現在{step}ステップで')
            logger.debug(f'このステップでの報酬は{reward}')
            logger.debug(f'収益は{reward_sum}')
    
        if done: break#タスクが失敗してdoneがTrueになったらゲームオーバー
    agent.stop_episode_and_train(observ, reward, done) # DQNの重み更新
    if episode % 50 == 0:
        print(f'これまでのエピソードでの経過{agent.get_statistics()}')

#モデルを保存する。
SAVE_NAME = OUT_DIR_NAME + '/acrobot_agent'
agent.save(SAVE_NAME)
print('model_saved')