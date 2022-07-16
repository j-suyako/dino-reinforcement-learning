from argparse import Action
from .websocket import WebsocketServer
from io import BytesIO
from PIL import Image
import base64
import json
import multiprocessing
import numpy as np
import time
import threading

UP, FORWARD, DOWN = 0, 1, 2

class Agent(object):
    actions = {UP: 'UP', FORWARD: 'FORWARD', DOWN: 'DOWN'}

    def __init__(self, host, port, debug=False) -> None:
        self.debug = debug
        self.queue = multiprocessing.Queue()
        self.client = None
        self.server = WebsocketServer(port, host)
        self.server.set_fn_new_client(self.new_client)
        self.server.set_fn_message_received(self.new_message)
        print('\nGame can be connected(press F5 in Browser)')
        thread = threading.Thread(target=self.server.run_forever)
        thread.daemon = True
        thread.start()

    def new_client(self, client, server):
        if self.debug:
            print('Game just connected')
        self.client = client
        self.server.send_message(self.client, 'Connection to game agent established.')

    def new_message(self, client, server, message):
        if self.debug:
            print('Incoming data from game')
        data = json.loads(message)
        image, crashed, distance = data['world'], data['crashed'], data['distance']
        prefix = 'data:image/png;base64,'
        image = image[len(prefix):]
        image = np.array(Image.open(BytesIO(base64.b64decode(image))))
        crashed = True if str.lower(crashed) == 'true' else False
        self.queue.put((image, crashed, distance))

    def start_game(self):
        while self.client is None:
            time.sleep(1)
        self.server.send_message(self.client, 'START')
        time.sleep(4)
        return self.get_state(FORWARD)

    def refresh_game(self):
        time.sleep(.5)
        print('refresh game')
        self.server.send_message(self.client, 'REFRESH')
        time.sleep(1)

    def do_action(self, action):
        if action != FORWARD:
            self.server.send_message(self.client, self.actions[action])
        time.sleep(.05)
        return self.get_state(action)

    def get_state(self, action):
        self.server.send_message(self.client, 'STATE')
        image, terminal, distance = self.queue.get()
        if terminal:
            reward = -100.
        elif action == UP:
            reward = 1
        elif action == DOWN:
            reward = 3
        else:
            reward = 7
        return image, terminal, reward, distance


if __name__ == "__main__":
    agent = Agent('127.0.0.1', '14400', debug=True)