CharVectorLength = 40
I_SIZE = 8
F_SIZE = 8
O_SIZE = 8
L_SIZE = I_SIZE + F_SIZE

import numpy as np
class Coder:
    def __init__(self, size=32, unit=10):
        self.text2vector = {}
        self.vector2text = {}
        self.size = size
        self.unit = unit
        self.index = 0
        self.null = np.zeros(size)

    def encodeTextToVector(self, text):
        if len(text) == 0 or text=='':
            return self.null
        else:
            if text not in self.text2vector:
                self.index = self.index + 1
                vector = np.zeros(self.size)
                vector[int((self.index-1)/self.unit)] = (self.index - int((self.index-1)/self.unit)*self.unit)/self.unit
                self.text2vector[text] = vector
                self.vector2text[self.index] = text
            return self.text2vector[text]

    def decodeVectorToText(self, vector):
        if len(vector)==0 or vector == self.null:
            return ''
        else:
            argmax = np.argmax(vector)
            index = int(argmax * self.unit + vector[argmax] * self.unit)
            if index not in self.vector2text:
                text = ''
            else:
                text = self.vector2text[index]
            return text

class GraphMemory(object):
    def __init__(self):
        self.nodes = {}

    def rememberTextToGraphBall(self, text, vector):
        self.nodes[text] = self.Ball(text, vector)

    def correlateTextToGraphBall(self, text):
        if text in self.nodes:
            return self.nodes[text]
        else:
            return None

    class Ball(object):
        def __init__(self, ownTensor, ownSymbol):
            self.own_tensor = ownTensor
            self.own_symbol = ownSymbol
            self.cognition_graph_correlation = []
            self.perception_scene_correlation = []
            self.reasonable_function_networks = []
            self.local_temporary_memory = []
            self.particular_case = []

class QueueMemory:
    def __init__(self):
        self.pad = np.zeros(CharVectorLength)
        self.i_size = I_SIZE
        self.i_acts = []
        self.i_txts = []

        self.f_size = F_SIZE
        self.f_list = []

        self.o_size = O_SIZE
        self.o_acts = []
        self.o_txts = []

    def cache_i_act(self, act):
        self.i_acts = self._limited_append(self.i_acts, act, self.i_size)

    def cache_i_txt(self, txt):
        self.i_txts = self._limited_append(self.i_txts, txt, self.i_size)

    def react_o_act(self):
        return self.o_acts[-1] if len(self.o_acts) > 0 else []

    def react_o_txt(self):
        return self.o_txts[-1] if len(self.o_txts) > 0 else []

    def get(self, out=0):
        state = []

        total_i = 0
        for i in range(len(self.i_acts)-1, -1, -1):
            moment_data_i = np.concatenate((self.i_acts[i], self.i_txts[i]))
            moment_size_i = int((len(self.i_acts[i])+len(self.i_txts[i]))/CharVectorLength)
            if total_i + moment_size_i <= I_SIZE:
                state = np.concatenate((moment_data_i,state))
                total_i = total_i + moment_size_i
        for pad in range(I_SIZE-total_i):
            state = np.concatenate((self.pad, state))
            total_i = total_i + int(len(self.pad)/CharVectorLength)

        total_f = 0
        for f in range(len(self.f_list)-1, -1, -1):
            moment_data_f = self.f_list[f]
            moment_size_f = int(len(self.f_list[f])/CharVectorLength)
            if total_f + moment_size_f <= F_SIZE:
                state = np.concatenate((moment_data_f,state))
                total_f = total_f + moment_size_f
        for pad in range(F_SIZE-total_f):
            state = np.concatenate((self.pad, state))
            total_f = total_f + int(len(self.pad)/CharVectorLength)

        for pad in range(L_SIZE-(total_i+total_f)):
            state = np.concatenate((self.pad, state))

        if out:
            total_o = 0
            for o in range(len(self.o_acts)-1, -1, -1):
                moment_data_o = np.concatenate((self.o_acts[o], self.o_txts[o]))
                moment_size_o = int((len(self.o_acts[o])+len(self.o_txts[o]))/CharVectorLength)
                if total_o + moment_size_o <= O_SIZE:
                    state = np.concatenate((moment_data_o,state))
                    total_o = total_o + moment_size_o
            for pad in range(O_SIZE-total_o):
                state = np.concatenate((self.pad, state))
                total_o = total_o + int(len(self.pad)/CharVectorLength)

        return state

    def put(self, fun, act, txt):
        self.f_list = self._limited_append(self.f_list, fun, self.f_size)
        self.o_acts = self._limited_append(self.o_acts, act, self.o_size)
        self.o_txts = self._limited_append(self.o_txts, txt, self.o_size)

    def _limited_append(self, all, one, size):
        if len(all) >= size:
            all = all[1:]
        all.append(one)
        return all

class Function_L2_DL_Nops:
    def __init__(self, myFunctionIdEmbedding, workingMemroy, coder):
        self.myFunctionIdEmbedding = myFunctionIdEmbedding
        self.workingMemroy = workingMemroy
        self.coder = coder
        self.nops = np.zeros(CharVectorLength)

    def interact_txt(self, txt):
        print('Function_L2_DL_Nops: interact_txt', self.coder.decodeVectorToText(txt))
        self.workingMemroy.put(self.myFunctionIdEmbedding, self.nops, self.nops)
        
class Function_L2_DL_SubnetworkAllocator:
    def __init__(self, myFunctionIdEmbedding, workingMemroy, coder, upperFunctions):
        self.myFunctionIdEmbedding = myFunctionIdEmbedding
        self.workingMemroy = workingMemroy
        self.coder = coder
        self.upperFunctions = upperFunctions
        self.nops = np.zeros(CharVectorLength)

    def interact_txt(self, txt):
        print('Function_L2_DL_SubnetworkAllocator: interact_txt', self.coder.decodeVectorToText(txt))
        self.upperFunctions.append(Function_L2_DL_Nops(self.coder.encodeTextToVector(str(len(self.upperFunctions))), self.workingMemroy, self.coder))
        #TODO 手工的特定网络分配逻辑： subnetwork(by task:out=function_id; by task:out=input_segment; by data:out=born_char&word; by data:out=long_term_memory_graph) & calling
        
class RL:
    def __init__(self, max_functions=2):
        from rl_sac import SAC
        self.rl = SAC(state_space=CharVectorLength*L_SIZE, action_space=max_functions)
        
        self.steps = 0
        
    def react(self, learn=1, rl_solveing_iterations=2, exploration_steps=100):   #TODO 这里每次记录的old new state长度不一样，RL没法处理
        self.steps += 1        
        for iterator in range(rl_solveing_iterations if learn else 1):
            self.old_state = np.float32(self.workingMemroy.get())
            computed_action, log_prob = self.rl.exploration_action(self.old_state, self.steps<exploration_steps) if learn else self.rl.exploitation_action(self.old_state)
            self.action = computed_action
            self.log_prob = log_prob
            action_index = int(np.argmax(computed_action))
            print('RL: action_index=',action_index)
            if action_index < len(self.functions):
                self.functions[action_index].interact_txt(self.workingMemroy.get())
        self.new_state = np.float32(self.workingMemroy.get())    
        
    def reward(self, reward):
        self.rl.remember(self.old_state, self.action, reward, self.new_state, self.log_prob)

    def rethink(self):
        self.rl.rethink()  

class Function_L1_RL_Universal(RL):
    def __init__(self, myFunctionIdEmbedding, workingMemroy, coder):
        super().__init__()
        self.myFunctionIdEmbedding = myFunctionIdEmbedding
        self.workingMemroy = workingMemroy
        self.coder = coder
        self.nops = np.zeros(CharVectorLength)
        
        self.functions = []
        self.functions.append(Function_L2_DL_SubnetworkAllocator(self.coder.encodeTextToVector(str(len(self.functions))), workingMemroy, self.coder, self.functions))

    def interact_txt(self, txt):  #TODO 学习并调度子功能区 
        print('Function_L1_RL_Universal: interact_txt', self.coder.decodeVectorToText(txt))
        self.react()       

class Function_L1_RL_SubnetworkAllocator:
    def __init__(self, myFunctionIdEmbedding, workingMemroy, coder, upperFunctions):
        self.myFunctionIdEmbedding = myFunctionIdEmbedding
        self.workingMemroy = workingMemroy
        self.coder = coder
        self.upperFunctions = upperFunctions
        self.nops = np.zeros(CharVectorLength)

    def interact_txt(self, txt):  #TODO 手工的特定网络分配逻辑： functional-policy
        print('Function_L1_RL_SubnetworkAllocator: interact_txt', self.coder.decodeVectorToText(txt))       
        self.upperFunctions.append(Function_L1_RL_Universal(self.coder.encodeTextToVector(str(len(self.upperFunctions))), self.workingMemroy, self.coder))

class Brain(RL):
    def __init__(self):
        super().__init__()
        
        self.coder = Coder()

        self.workingMemroy = QueueMemory()
        self.longtermMemroy = GraphMemory()

        self.functions = []
        self.functions.append(Function_L1_RL_SubnetworkAllocator(self.coder.encodeTextToVector(str(len(self.functions))), self.workingMemroy, self.coder, self.functions))

        self.rewards = 0

    def interact_act(self, act):
        print('Brain: interact_act', act)
        self.rewards += 1
        self.reward(act)
        if self.rewards > 16:
            self.rewards = 0
            self.rethink()

    def interact_txt(self, txt):
        print('Brain: interact_txt', txt)
        emb = self.coder.encodeTextToVector(txt)

        self.workingMemroy.o_txts = []
        self.workingMemroy.cache_i_txt(emb)
        
        self.react()

        out = self.workingMemroy.react_o_txt()
        gen = self.coder.decodeVectorToText(out)
        print('Brain: interact_gen', gen)
        return gen       

import random
class World:
    def __init__(self):
        self._data = []
        self.__load_born('./data/data_born.txt')
        self.__load_task('./data/data_task.txt')
        self.curriculum = None
        self.states = None
        self.action = None
        self.step = None

    def interact_txt(self, txt):
        print('World: interact_txt', txt)
        if self.step is None or self.step==len(self.states):
            self.curriculum = 0 if txt is None or self.curriculum == len(self.data) else self.curriculum + 1
            self.states = self._data[self.curriculum][0]['i_txt']
            self.action = self._data[self.curriculum][1]['o_txt']
            self.step = 0
            
            print()
            import time
            time.sleep(0.1)
            
        state = self.states[self.step]
        if self.step < len(self.states)-1 and (txt is not None and len(txt) <=0):  #一句未完，不答有奖
            reward = '1'
        elif self.step == len(self.states)-1 and (txt is not None and txt in self.action):  #一句完成，答对有奖
            reward = '1'
        else:
            reward = '0'
        self.step += 1
        return state, reward

    def interact_act(self, act):
        print('World: interact_act', act)

    def __load_born(self, filename):
        born = []
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(len(lines)):
               line = lines[i].rstrip('\n')
               if len(line)>0 and line[0] != '#':
                   char_or_word = '字' if len(line)==1 else '词'
                   i1 = {}
                   o1 = {}
                   i1['i_txt']='学' + char_or_word+ ':' + line
                   o1['o_txt']=''
                   i2 = {}
                   o2 = {}
                   i2['i_txt']='背' + char_or_word
                   o2['o_txt']=line
                   born.append( ((i1,o1),(i2,o2)) )
        random.shuffle(born)
        for b in born:
            self._data.append(b[0])
            self._data.append(b[1])

    def __load_task(self, filename):
        task = []
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
               line = line.rstrip('\n')
               if len(line)>0 and line[0] != '#':
                   if line.startswith('i'):
                       i = {}
                       i['i_txt']=line[len('i: '):]
                   elif line.startswith('o'):
                       o = {}
                       o['o_txt']=line[len('o: '):].split('##')
                       task.append((i,o))

        random.shuffle(task)
        for t in task:
            self._data.append(t)

def tpj():
    world = World()
    brain = Brain()
    action = None
    while 1:
        state, reward = world.interact_txt(action)
        brain.interact_txt(state)
        brain.interact_act(reward)

if __name__ == '__main__':
    tpj()
