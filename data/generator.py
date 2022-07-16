from PIL import Image
from util.agent import Agent
import numpy as np
import tensorflow as tf
import time


class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, item):
        self.position = (self.position + 1) % self.capacity
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        replay = np.array(self.memory)[idx]
        return (np.concatenate(replay[:, 0]), 
                np.array(replay[:, 1], np.uint8), 
                np.array(replay[:, 2], np.float32), 
                np.concatenate(replay[:, 3]))

    def __len__(self):
        return len(self.memory)


class Dataset(object):
    def __init__(self, epsilon_start=0.9, epsilon_end=0.005, epsilon_decay=200) -> None:
        self.agent = Agent('127.0.0.1', 9090)
        self.short_replays = ReplayMemory(2000)
        self.long_replays = list()
        self.state = list()
        self.model = None
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end 
        self.decay = epsilon_decay

    def prepare_state(self, frame):
        roi_height, roi_width = frame.shape[0], int(frame.shape[1] * .68)
        roi = frame[:, :roi_width, 0]
        image = np.zeros_like(roi)
        obstacle = roi > 50
        image[obstacle] = 1
        unharmful = roi > 200
        image[unharmful] = 0
        image = np.array(Image.fromarray(image).resize((80, 80))) / 255.0
        # image = image.resize((80, 80))
        # image = np.array(image, np.float32) / 255
        self.state.append(image)
        while len(self.state) < 4:
            self.state.append(image)
        self.state = self.state[-4:]
        return np.expand_dims(np.stack(self.state, axis=-1), axis=0)

    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(3), True
        return tf.argmax(self.model(state, training=False), axis=-1)[0].numpy(), False

    def generator(self, batch_size, cnt=4):
        if self.model is None:
            raise ValueError("must set target model first")
        terminal = False
        # if self.agent.client is None:
        curr_frame, terminal, _, _ = self.agent.start_game()
        curr_state = self.prepare_state(curr_frame)
        # else:
        #     curr_frame, terminal, _, _ = self.agent.do_action(0)
        #     time.sleep(3)
        #     # curr_frame, terminal, _, _ = self.agent.do_action(1)
        #     curr_state = self.prepare_state(curr_frame)
        step = 0
        replay = ReplayMemory(500)
        actions = list()
        while not terminal:
            # start_frame, terminal, start_reward = self.agent.start_game()
            # state = self.prepare_state(curr_frame)
            action, random = self.select_action(curr_state)
            actions.append(str(action) + ("*" if random else ""))
            next_frame, terminal, reward, distance = self.agent.do_action(action)
            next_state = self.prepare_state(next_frame)
            replay.push((curr_state, action, reward, next_state, distance))
            self.short_replays.push((curr_state, action, reward, next_state, distance))
            curr_state = next_state
            step += 1
            yield self.short_replays.sample(batch_size)
        # print(step, '...' + ','.join(actions[-10:]))
        # if step > 50:
        #     self.long_replays.append(replay)
        # if len(self.long_replays) < 5:
        #     # for _ in range(cnt):
        #         yield self.short_replays.sample(batch_size)
        # else:
        #     for _ in range(cnt):
        #         prob = np.array([len(e) for e in self.long_replays])
        #         prob = prob / np.sum(prob)
        #         replay = np.random.choice(self.long_replays, size=1, p=prob)[0]
        #         yield replay.sample(batch_size)

    def update_model(self, model):
        self.model = model

    def update_epsilon(self, step):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * step / self.decay)


if __name__ == "__main__":
    from nets.model import DinoModel
    model = DinoModel()
    datasets = Dataset()
    datasets.update_model(model)
    datasets.update_epsilon(0)
    for e in datasets.generator(model):
        pass
