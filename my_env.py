import pygame
from load import *
import math
import time
import numpy as np
import pandas as pd

class MyYuanYangEnv:
    def __init__(self):
        self.state_space = np.arange(0, 100)
        self.action_space = ['e', 's', 'w', 'n']
        self.gamma = 0.8
        self.maze_shape = (10, 10)
        self.value = np.zeros(self.maze_shape)
        self.viewer = None
        self.FPSCLOCK = pygame.time.Clock()

        self.screen_size = (1200, 900)
        self.bird_position = (0, 0)
        self.step_size_x = 120
        self.step_size_y = 90
        self.obstacle_size = [120, 90]
        self.maze_boundary = {"x": [0, self.screen_size[0]],
                              "y": [0, self.screen_size[1]]}

        self.path = []
        self.bird_male_init_position = [0, 0]
        self.bird_male_position = [0, 0]
        self.bird_female_init_position = [1080, 0]

        self.obstacle_1, self.obstacle_2 = self.set_obstacle()

        # 各向同性初始化概率矩阵
        self.state_trans_pro_mat = pd.DataFrame(data=np.ones((len(self.state_space),
                                                              len(self.action_space)))/len(self.action_space),
                                                index=self.state_space,
                                                columns=self.action_space)

    def set_obstacle(self):
        obstacle_1 = {"x": [], 'y': []}
        obstacle_2 = {"x": [], 'y': []}

        obstacle_1["x"] = obstacle_1["x"] + [360] * 8
        obstacle_1["y"] = obstacle_1["y"] + list(np.arange(10) * 90)
        obstacle_1["y"] = obstacle_1["y"][:4] + obstacle_1["y"][6:10]

        obstacle_2["x"] = obstacle_2["x"] + [720] * 8
        obstacle_2["y"] = obstacle_2["y"] + list(np.arange(10) * 90)
        obstacle_2["y"] = obstacle_2["y"][:3] + obstacle_2["y"][5:10]

        return obstacle_1, obstacle_2

    def collision(self, state_position):
        # 判断是否与两堵墙或者边界相撞, 0 :不相撞
        flag = 0
        flag_ob_1, flag_ob_2 = 0, 0

        dx1 = np.abs(np.asarray(state_position[0]) - np.asarray(self.obstacle_1["x"]))
        dy1 = np.abs(np.asarray(state_position[1]) - np.asarray(self.obstacle_1["y"]))
        if np.amin(dx1) < self.obstacle_size[0] or np.amin(dy1) < self.obstacle_size[1]:
            flag_ob_1 = 1

        dx2 = np.abs(np.asarray(state_position[0]) - np.asarray(self.obstacle_2["x"]))
        dy2 = np.abs(np.asarray(state_position[1]) - np.asarray(self.obstacle_2["y"]))
        if np.amin(dx2) < self.obstacle_size[0] or np.amin(dy2) < self.obstacle_size[1]:
            flag_ob_2 = 1

        if self.maze_boundary["x"][0] > state_position[0] or self.maze_boundary["x"][1] <= state_position[0]:
            flag = 1
        if self.maze_boundary["y"][0] > state_position[1] or self.maze_boundary["y"][1] <= state_position[1]:
            flag = 1

        if flag_ob_1 or flag_ob_2:
            flag = 1

        return flag

    def find(self, state_position):
        # 1 找到
        flag = 0
        if abs(state_position[0] - self.bird_female_init_position[0]) < self.obstacle_size[0] and \
                abs(state_position[1] - self.bird_female_init_position[1]) < self.obstacle_size[1]:
            flag = 1
        return flag

    def state_to_position(self, state):
        i = int(state / 10)
        j = state % 10
        position = [0, 0]
        position[0] = 120 * j
        position[1] = 90 * i
        return position

    def position_to_state(self, position):
        i = position[0] / 120
        j = position[1] / 90
        return int(i + 10 * j)

    def reset(self):
        # 随机产生初始状态
        flag1 = 1
        flag2 = 1
        while flag1 or flag2:
            # 随机产生初始状态，0~99，randoom.random() 产生一个0~1的随机数
            state = self.states[int(np.random.random() * len(self.states))]
            state_position = self.state_to_position(state)
            flag1 = self.collision(state_position)
            flag2 = self.find(state_position)
        return state

    def step(self, current_state, action):
        current_position = self.state_to_position(current_state)
        next_position = [0, 0]

        flag_collision = self.collision(current_position)
        flag_find = self.find(current_position)

        if flag_collision or flag_find:
            return current_state, 0, True

        if action == 'e':
            next_position[0] = current_position[0] + 120
            next_position[1] = current_position[1]
        if action == 's':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] + 90
        if action == 'w':
            next_position[0] = current_position[0] - 120
            next_position[1] = current_position[1]
        if action == 'n':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] - 90

        flag_collision = self.collision(next_position)
        flag_find = self.find(next_position)

        if flag_collision:
            return self.position_to_state(current_position), -1, True

        if flag_find:
            return self.position_to_state(next_position), 1, True

        return self.position_to_state(next_position), 0, False

    def gameover(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

    def render(self):
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
            pygame.display.set_caption("yuanYangGame")

            self.bird_male = load_bird_male()
            self.bird_female = load_bird_female()
            self.background = load_background()
            self.obstacle = load_obstacle()

            self.viewer.blit(self.bird_female, self.bird_female_init_position)
            self.viewer.blit(self.background, (0, 0))
            self.font = pygame.font.SysFont('times', 15)

            # 画直线
            for i in range(11):
                pygame.draw.lines(self.viewer, (255, 255, 255), True, ((120 * i, 0), (120 * i, 900)), 1)
                pygame.draw.lines(self.viewer, (255, 255, 255), True, ((0, 90 * i), (1200, 90 * i)), 1)
            self.viewer.blit(self.bird_female, self.bird_female_init_position)
            # 画障碍物
            for i in range(8):
                self.viewer.blit(self.obstacle, (self.obstacle_1["x"][i], self.obstacle_1["y"][i]))
                self.viewer.blit(self.obstacle, (self.obstacle_2["x"][i], self.obstacle_2["y"][i]))
            # 画小鸟
            self.viewer.blit(self.bird_male, self.bird_male_position)

            # 画值函数
            for i in range(10):
                for j in range(10):
                    surface = self.font.render(str(round(float(self.value[i, j]), 3)), True, (0, 0, 0))
                    self.viewer.blit(surface, (120 * i + 5, 90 * j + 70))
            # 画路径点
            for i in range(len(self.path)):
                rec_position = self.state_to_position(self.path[i])
                pygame.draw.rect(self.viewer, [255, 0, 0], [rec_position[0], rec_position[1], 120, 90], 3)
                surface = self.font.render(str(i), True, (255, 0, 0))
                self.viewer.blit(surface, (rec_position[0] + 5, rec_position[1] + 5))
            pygame.display.update()
            self.gameover()
            # time.sleep(0.1)
            self.FPSCLOCK.tick(30)







