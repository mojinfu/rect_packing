import pdb
import json
from tensorboardX import SummaryWriter
# import cv2
import sys
# from PIL import Image
from binmanager import *
# import matplotlib.pyplot as pyplot
import os
# import scipy
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# sys.path.append("game/")
# import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
import torch.nn as nn
writer = SummaryWriter("/Users/mojinfu/PyWorks/github.com/mojinfu/rect_packing/boardx/0820/")
binW =  2
binH  = 2
binV =  4
itemV  = 1
# GAME = 'bird' # the name of the game being played for log files
# ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 50000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
INITIAL_EPSILON = 0.2 # starting value of epsilon
REPLAY_MEMORY = 1000 # number of previous transitions to remember
BATCH_SIZE = 128 # size of minibatch
FRAME_PER_ACTION = 1
UPDATE_TIME = 1000
# width = 80
# height = 80


class DeepNetWork(nn.Module):
    def __init__(self,):
        super(DeepNetWork,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=binV, out_channels=32, kernel_size=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(800*binV+itemV*2,512),
            nn.Linear(64+itemV*2,binV*itemV*4),
            nn.ReLU()
        )
        self.out = nn.Linear(binV*itemV*4,binV*itemV*2)

    def forward(self, binInfo,itemInfo):
        binInfo = self.conv1(binInfo)
        binInfo = self.conv2(binInfo)
        binInfo = self.conv3(binInfo)
        binInfo = binInfo.view(binInfo.size(0),-1)
        itemInfo = itemInfo.view(itemInfo.size(0),-1)
        allInfo = torch.cat((binInfo,itemInfo ),dim = 1 )
        allInfo = self.fc1(allInfo); 
        return self.out(allInfo)
class BrainDQNMain(object):
    def save(self):
        print("save model param")
        torch.save(self.Q_new.state_dict(), str(self.timeStep)+'params3.pth')

    def load(self,model):
        if os.path.exists(model):
            print("load model param")
            self.Q_new.load_state_dict(torch.load(model))
            self.Q_old.load_state_dict(torch.load(model))
    def loadWithPath(self,modelPath):
        if os.path.exists(modelPath):
            print("load model param")
            self.Q_new.load_state_dict(torch.load(modelPath))
            self.Q_old.load_state_dict(torch.load(modelPath))
    def __init__(self,actions,saveAndRun = True, model = ""):
        self.replayMemory = deque() # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.Q_new=DeepNetWork()
        self.Q_old=DeepNetWork()
        if model!="":
            self.load(model)
        else:
            print("init model")

        self.loss_func=nn.MSELoss()
        LR=1e-6
        self.optimizer = torch.optim.Adam(self.Q_new.parameters(), lr=LR)
        if saveAndRun:
            self.saveAndRun(-1)

    def train(self): # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
         # Step 2: calculate y
        y_batch = np.zeros([BATCH_SIZE,1])


        # action_batch=np.array(action_batch)
        # index=action_batch.argmax(axis=0)
        # print("action "+str(index))
        # index=np.reshape(index,[BATCH_SIZE,1])
        # action_batch_tensor=torch.LongTensor(index)


        # nextState_batch = [data[3] for data in minibatch]
        # nextState_batch=np.array(nextState_batch) #print("train next state shape")
        # nextState_batch=torch.Tensor(nextState_batch)
        # QValue_batch = self.Q_old(nextState_batch)
        # QValue_batch=QValue_batch.detach().numpy()

# 实验数据 
        
        currentBinState = torch.Tensor([data[0] for data in  nextState_batch])
        currentItemState = torch.Tensor([data[1] for data in  nextState_batch])
        QValue_batch = self.Q_new(currentBinState ,currentItemState )
        QValue_batch = QValue_batch.detach().numpy()



        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0]=reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=rewaerd[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                y_batch[i][0]=reward_batch[i] + GAMMA * np.max(QValue_batch[i])

        y_batch=np.array(y_batch)
        y_batch=np.reshape(y_batch,[BATCH_SIZE,1])
        fun_value=Variable(torch.Tensor(y_batch))


        # state_batch_tensor=Variable(torch.Tensor(state_batch))
        # print("self.Q_new(state_batch_tensor): ",self.Q_new(state_batch_tensor))
        # model_predict_value=self.Q_new(state_batch_tensor).gather(1,action_batch_tensor)

# 实验数据 
        currentBinState2 = torch.Tensor([data[0] for data in  state_batch])
        currentItemState2 = torch.Tensor([data[1] for data in  state_batch])
        model_predict_value = self.Q_new(currentBinState2 ,currentItemState2 ).gather(1,torch.LongTensor([[ac] for ac in action_batch]))
        # currentQValue_batch=currentQValue_batch.detach().numpy()



        loss=self.loss_func(model_predict_value,fun_value)
        # print(self.timeStep, " loss is ",str(loss))
        if self.timeStep%5==0:
            writer.add_scalar("0811模型损失曲线",loss,self.timeStep)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.timeStep%100 == 0 :
            print("self.timeStep:",self.timeStep,"loss ",loss)
        if self.timeStep % (UPDATE_TIME * 10) == 0:
            self.saveAndRun( 10000 )
        elif self.timeStep % UPDATE_TIME == 0:
            self.saveAndRun( -1 )
    def saveAndRun(self, drawNum ):
        self.Q_old.load_state_dict(self.Q_new.state_dict())
        self.save()
        # 测试装箱效果
        reward , binNum  = self.runModel(drawNum,1)
        writer.add_scalar("0811reward收敛情况",reward,self.timeStep)
        writer.add_scalar("0811效果测试",binNum,self.timeStep)
        print("0811reward收敛情况",reward,self.timeStep )
        print("0811效果测试",binNum,self.timeStep )
    def setPerceptionAndTrain(self,nextObservation,action,reward,terminal): #print(nextObservation.shape)
        # newState = np.append(self.currentState[1:,:,:],nextObservation,axis = 0) # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # 这是寻找连续动作下的状态  bin packing 不需要
        self.replayMemory.append((self.currentState,action,reward,nextObservation,terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE: # Train the network
            self.train()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        # print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
        self.currentState = nextObservation
        self.timeStep += 1

    def getAction(self, bm ,ifRandom = True):
        currentBinState = torch.Tensor([self.currentState[0]])
        currentItemState = torch.Tensor([self.currentState[1]])
        QValue = self.Q_new(currentBinState ,currentItemState )[0]
        action = np.zeros(self.actions)
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon and ifRandom:
                action_index = random.randrange(self.actions)
                
                while True:
                    if bm._algoVItemIndexList[(action_index//2) % itemV ]<0:
                        action_index = (action_index + 2) % self.actions
                    else:
                        break
                       


                # print("choose random action " + str(action_index))
                return action_index
            else:
                Q  =  QValue.detach().numpy()
                for i in range(Q.size):
                    if bm._algoVItemIndexList[(i//2) % itemV ]<0:
                        Q[i]= - np.Inf
                action_index = np.argmax(Q)
                # print("choose qnet value action " + str(action_index))
                return action_index



        # change episilon
        raise("")
        return random.randrange(self.actions)

    def setInitState(self, observation):
        # self.currentState = np.stack((observation, observation, observation, observation),axis=0)
        self.currentState = observation
        # print(self.currentState.shape)
    def runModel(self,drawNum,askNum):
        FINAL_EPSILON = 0
        data = json.loads(open('./testData.json').read() )
        myEnv = env(binW,binH)
        rewardAll = 0
        binAll = 0
        orgDrawNum = drawNum
        for ii in range(len(data)):
            if ii>=askNum:
                break
            ask = data[ii]
            bm= binManager(myEnv ,binV,itemV)
            for wh in ask:
                # bm.AddRandomItem()
                if random.random() < 0.5:
                    bm.AddItem(1,2)
                else:
                    bm.AddItem(2,1) 
                # bm.AddItem(wh[0],wh[1]) 

            self.setInitState(bm.AllStatus())
            if orgDrawNum>0 :
                paper = Paper(myEnv.binWidth,myEnv.binHeight)
            while True:
                action = self.getAction(bm,ifRandom = False) 
                # rotationChose =  action % 2 
                # binChose = (action//2) // itemV
                # itemChose = (action//2) % itemV
                binChose ,itemChose,rotationChose =   getChoseByAction(action)
                terminal,ifSucc ,reward = bm.Action(itemChose,binChose ,rotationChose)
                rewardAll = rewardAll+ reward
                self.setInitState(bm.AllStatus())
                if terminal :
                    binAll = binAll + len(bm.bins)
                    if orgDrawNum>0:
                        paper.Close()
                    break
                if drawNum>0:
                    drawNum = drawNum - 1
                    print("rotationChose :",rotationChose,"binChose :",binChose,"itemChose :",itemChose,"reward :",reward,"terminal :",terminal)
                    # print()
                    # print()
                    # print()
                    # print()
                    # print("---------------------")
                
                    if ifSucc:
                        bin = bm.bins[bm._algoVBinIndexList[binChose]]
                        paper.AddRect(bin.placedRect[-1].X(),bin.placedRect[-1].Y(),bin.placedRect[-1].Width(),bin.placedRect[-1].Height(),binChose)
                        time.sleep(0.5)
                        paper.AddBlackRect(bin.placedRect[-1].X(),bin.placedRect[-1].Y(),bin.placedRect[-1].Width(),bin.placedRect[-1].Height(),binChose)
                    # if reward == -1:
                    #     print("选错item")
                    if not ifSucc:
                        paper.Close()
                        paper = Paper(myEnv.binWidth,myEnv.binHeight)
                        for iii in range(len(bm._algoVBinIndexList)):
                            bin = bm.bins[bm._algoVBinIndexList[iii]]
                            for rect in bin.placedRect:
                                paper.AddBlackRect(rect.X(),rect.Y(),rect.Width(),rect.Height(),iii)
                                # time.sleep(0.01)

                
                
        return rewardAll ,binAll
def getChoseByAction(action):
    rotationChose =  action % 2 
    binChose = (action//2) // itemV
    itemChose = (action//2) % itemV
    return binChose,itemChose, rotationChose
def runModel(modelPath,ifDraw,askNum):
        brain = BrainDQNMain(binV * itemV *2 , False,modelPath)
        
        if ifDraw:
            rewardAll ,binAll  = brain.runModel( 1000000,askNum)
        else:
            rewardAll ,binAll  = brain.runModel( -1 ,askNum)    
        print("rewardAll:",rewardAll)
        print("使用bin数量:",binAll)
def trainModel():
    actions = binV * itemV * 2
    random.seed(0)
    myEnv =env(binW,binH)
    
    while True:
        bm= binManager(myEnv ,binV,itemV)
        for i in range(itemV*5):
            # bm.AddRandomItem()
            if random.random() < 0.5:
                bm.AddItem(1,2)
            else:
                bm.AddItem(2,1) 


        brain = BrainDQNMain(actions,False) # Step 2: init Flappy Bird Game
        # status0 =  # Step 3: play game
        # # Step 3.1: obtain init state
        # terminal,reward0 = bm.Action(0,0)
        # status1 =  bm.AllStatus()

        brain.setInitState(bm.AllStatus())
        # print(brain.currentState.shape) # Step 3.2: run the game

        while 1!= 0:
            action = brain.getAction(bm,ifRandom = True)
            binChose ,itemChose,rotationChose =   getChoseByAction(action)
            terminal,_,reward = bm.Action(itemChose,binChose,rotationChose)
            # if brain.timeStep %100 ==0:
                # print("rotationChose :",rotationChose,"binChose :",binChose,"itemChose :",itemChose,"reward :",reward,"terminal :",terminal)
            
            brain.setPerceptionAndTrain(bm.AllStatus(),action,reward,terminal)
            if terminal:
                bm= binManager(myEnv,binV,itemV)
                for i in range(itemV*5):
                    # bm.AddRandomItem()
                    if random.random() < 0.5:
                        bm.AddItem(1,2)
                    else:
                        bm.AddItem(2,1) 
                brain.setInitState(bm.AllStatus())

            
if __name__ == '__main__': 
    # Step 1: init BrainDQN
    random.seed(0)
    # runModel("./120000params3.pth",False,1)
    # runModel("./2000params3.pth",True,1)
    trainModel()
    
