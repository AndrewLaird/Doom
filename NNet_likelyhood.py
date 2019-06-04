import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import vizdoomgym
import vizdoom
import sys

from PIL import Image
import numpy as np
import PIL

import signal, os

def resize_and_grayscale(observation):
    #print("observation size",observation.shape)
    observation = observation[:210,:]
    observation = np.mean(observation,axis=2)
    observation_image = Image.fromarray(observation)
    observation_image = observation_image.resize((100,100))
    observation = np.array(observation_image)

    #print("observation size after",observation.shape)

    #image = Image.fromarray(observation)
    #image.show()
    return observation


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=128 ,
                               kernel_size=5 )
        #take size
        self.pool1 = nn.AvgPool2d(kernel_size=20)
        self.pool2 = nn.AvgPool2d(kernel_size=4)


        self.dense1 = nn.Linear(in_features=128,
                                out_features=3)

    def forward(self, x):
        if(len(x.shape) < 4):
            x = [x]
        # first reshape the image
        for i in range(len(x)):
            x[i] = resize_and_grayscale(x[i])
        x = torch.Tensor(x).reshape((-1,1, 100, 100))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.view((-1,128))


        #print("after pool size", x.shape)
        return F.relu(self.dense1(x))

        #return F.relu(self.conv2(x))


def backup_rewards(training_data):
    nn_training_data =[]
    running_reward = 0

    for i in range(len(training_data)-1,0,-1):

        action_prob,action,reward,_ = training_data[i]
        running_reward += reward
        nn_training_data.append((action_prob,action,running_reward))
    return nn_training_data

exploration = .1

def play_game(env,model,render=False):
    observation = env.reset()
    terminal = False
    training_data =[]
    steps = 0

    while not terminal:
        if(render):
            env.render()
        action_probs = model.forward(observation)
        numpy_probs = np.array(action_probs.detach()[0])
        # deal with nans
        nans = np.isnan(numpy_probs)
        for i in range(len(numpy_probs)):
            if(nans[i]):
                numpy_probs[i] = 0
        base_sum = np.sum(numpy_probs)
        #print(base_sum,numpy_probs)
        if(base_sum ==0):
            #make them all equally likely
            numpy_probs = np.array([1.0,1.0,1.0])

            base_sum = 3.0

        numpy_probs = numpy_probs/base_sum
        action = np.random.choice(3,1,p=numpy_probs)
        if(np.random.rand(1) < exploration):
            action = [env.action_space.sample()]
        #print("action",action)
        observation,reward,terminal,info = env.step(action)
        if(terminal or steps>1000):
            break
        else:
            steps += 1
            #print("reward:",reward)
            training_data.append((action_probs[0],action[0],reward,None))

    nn_training_data = backup_rewards(training_data)
    #print(nn_training_data)
    return nn_training_data




def train_on_data(model,optimizer,data):

    action_probs = []
    actions = []
    rewards= []
    loss = 0
    for action_prob,a,r in data:
        #print(action_prob)
        #action_probs.append(action_prob)
        target= [0,0,0]
        target[a] =1.0
        #actions.append(target)
        #rewards.append(r)
        if(r==0):
            continue

        loss += (action_prob - torch.Tensor(target)).pow(2) * -(r/abs(r))
        

    #action_probs = np.array(action_probs)
    #correct_actions = np.array(actions)
    #rewards= np.array(rewards)

    loss = loss.mean()

    #loss = (correct_actions - action_probs).pow(2).mean()
    print("loss:",loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def save_model(model):
    print("SIGINT seen saving the model")
    torch.save(model.state_dict(),"./doom_model.pt")



signal.signal(signal.SIGINT,save_model)


if __name__ == "__main__":
    env = gym.make("VizdoomBasic-v0")
    model = Model()
    optimizer= optim.Adam(model.parameters(),lr=1e-3)

    #observation = env.reset()
    #print(observation.shape)
    #observation = resize_and_grayscale(observation)
    #model_result= model.forward(observation)
    #print(model_result)
    while True:
        try:
            training_data= []
            for game in range(3):
                #print("\r",game)
                game_data =play_game(env,model,render=False)#True)
                training_data.extend(game_data)
            #print("start training")
            loss =train_on_data(model,optimizer,training_data)
            #print('end training')
        except vizdoom.vizdoom.SignalException:
            print()
            print("saving and exiting")
            save_model(model)
            sys.exit()



