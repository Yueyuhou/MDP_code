import time
from my_env import MyYuanYangEnv
import numpy as np
import pandas as pd


class MyDPValueIter:
    def __init__(self, yuanYang):
        self.states = yuanYang.state_space
        self.actions = yuanYang.action_space
        self.value = np.ones(len(self.states))
        self.yuan_yang = yuanYang
        self.gamma = self.yuan_yang.gamma
        self.state_trans_pro_mat = yuanYang.state_trans_pro_mat
        self.policy_eva_num = 10
        self.policy_ite_num = 10
        self.evaluate_threshold = 1e-3
        self.improve_method = 'greedy'

        self.optimal_choice = dict()
        for s in self.states:
            self.optimal_choice[s] = self.actions[0]

    def policy_evaluate(self):
        for i in range(self.policy_eva_num):
            delta = 0
            for s in self.states:
                flag_collision = self.yuan_yang.collision(self.yuan_yang.state_to_position(s))
                flag_find = self.yuan_yang.find(self.yuan_yang.state_to_position(s))
                if flag_find or flag_collision:
                    continue

                temp = 0
                for a in self.actions:
                    new_s, reward, _ = self.yuan_yang.step(s, a)
                    temp = temp + self.state_trans_pro_mat.at[s, a] * (reward + self.gamma * self.value[new_s])

                delta = np.amax([delta, np.abs(self.value[s] - temp)])
                self.value[s] = temp
            # if delta < self.evaluate_threshold:
            #     print("迭代收敛: ", i)
            #     break

    def policy_improve(self):
        for s in self.states:
            flag1 = self.yuan_yang.collision(self.yuan_yang.state_to_position(s))
            flag2 = self.yuan_yang.find(self.yuan_yang.state_to_position(s))
            if flag1 or flag2:
                continue

            if self.improve_method == "greedy":
                max_value_action = None
                max_value = -1e-7
                for a in self.actions:
                    new_s, _, _ = self.yuan_yang.step(s, a)
                    if new_s == s:
                        continue
                    if max_value < self.value[new_s]:
                        max_value = self.value[new_s]
                        max_value_action = a
                self.state_trans_pro_mat.loc[s] = np.zeros((1, len(self.actions)))
                self.state_trans_pro_mat.loc[s][max_value_action] = 1
                self.optimal_choice[s] = max_value_action

    def policy_iterate(self):
        for i in range(self.policy_ite_num):
            self.policy_evaluate()
            self.policy_improve()


if __name__ == "__main__":
    yuan_yang = MyYuanYangEnv()
    policy_ite = MyDPValueIter(yuan_yang)
    policy_ite.policy_iterate()
    flag = 1
    start_point = 0
    step_num = 0
    max_step_num = 1
    optimal_path = [start_point, ]

    # 将v值打印出来
    for state in range(100):
        i = int(state / 10)
        j = state % 10
        yuan_yang.value[j, i] = policy_ite.value[state]

    while flag:
        yuan_yang.path = optimal_path
        optimal_action = policy_ite.optimal_choice[start_point]
        yuan_yang.bird_male_position = yuan_yang.state_to_position(start_point)
        yuan_yang.render()
        time.sleep(0.2)
        step_num += 1

        new_s, _, flag_c_f = yuan_yang.step(start_point, optimal_action)
        if flag_c_f or step_num > max_step_num:
            flag = 0
        start_point = new_s

        yuan_yang.bird_male_position = yuan_yang.state_to_position(start_point)
        optimal_path.append(start_point)
        yuan_yang.render()

    while True:
        yuan_yang.render()
