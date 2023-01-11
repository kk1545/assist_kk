import os

import subprocess

for cnt in range(100):
    command = ["python", "demo_2_action_acrobot.py"]
    proc = subprocess.Popen(command)
    result = proc.communicate()

    #--sparce 疎  --random ランダムアシスト報酬 
    #command = ["python", "new_task_reward_ARLs_mse_acrobot.py", "--sparce"]
    command = ["python", "new_task_reward_ARLs_mse_acrobot.py", "--sparce", "--random"]

    proc = subprocess.Popen(command)
    result = proc.communicate()

    before_re_name = './output_acrobot/None/400/acrobot_agent'
    os.rename(before_re_name, before_re_name + '-n' + str(cnt))

command = ["python", "run_acrobot.py"]
proc = subprocess.Popen(command)
result = proc.communicate()