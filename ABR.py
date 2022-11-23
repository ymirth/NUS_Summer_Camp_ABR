''' Simulator for SWS3021'''
# import the env
import fixed_env as fixed_env
import load_trace as load_trace
import time as tm
import os
# import libs team4 use
import random
import math
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
# import tensorflow_probability as tfp
import numpy as np
# import matplotlib.pyplot as plt

# env
MODEL_PATH = './model'
BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # kpbs
TARGET_BUFFER = [0.5, 1.0]  # seconds
BATCH_SIZE = 64
# Action Product
def get_cart_prd(pools):
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    return result

Action_Space = [BIT_RATE, TARGET_BUFFER]

Action_Space = get_cart_prd(Action_Space)


def random_float(length, a, b):
    list = []
    count = 0
    while (count < length):
        number = random.randint(a, b)
        list.append(float(number))
        count = count + 1
    return list


def get_state():
    state = [0] * 10
    for i in range(10):
        state[i] = float(random.randint(0, 1))

    return state

def state_transform(states, batch_size):
    states = tf.reshape(states, [batch_size, len(states[0])])
    return states

def get_batch_states(batch_size=1):
    s = []
    for i in range(batch_size):
        state = get_state()
        s.append(state)
    return s

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        # print(f"buffer_size = ",batch_size)
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SoftQNetwork(keras.Model):
    def __init__(self, num_actions, batch_size=1, hidden_size=64):
        super(SoftQNetwork, self).__init__()
        self.unit = num_actions
        self.batch_size = batch_size

        # Concat_State_Input to The Actor:
        self.linear1 = keras.layers.Dense(hidden_size, activation='relu',
                                          kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros')
        self.drop1 = keras.layers.Dropout(rate=0.2)
        self.linear2 = keras.layers.Dense(math.floor(hidden_size / 1.5), activation='relu',
                                          kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros')
        self.drop2 = keras.layers.Dropout(rate=0.2)
        self.Q = keras.layers.Dense(num_actions, activation=None,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros')


    def call(self,states, batch_size=None, softmax_dim=-1):
        x = self.linear1(states)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        output = self.Q(x)
        return output



class PolicyNetwork(keras.Model):
    def __init__(self, num_actions, batch_size=1, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        print(f"-------", hidden_size)
        self.noise = 1e-7
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.unit = num_actions

        # Concat_State_Input to The Actor:
        self.linear1 = keras.layers.Dense(
            hidden_size, activation='relu')
        self.drop1 = keras.layers.Dropout(rate=0.2)
        self.linear2 = keras.layers.Dense(
            math.floor(hidden_size // 1.5), activation='relu')
        self.drop2 = keras.layers.Dropout(rate=0.1)
        self.output_layer = keras.layers.Dense(num_actions, activation=None)


    def call(self,states, batch_size=None,training = None):
        x = self.linear1(states)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        probs = keras.layers.Softmax(axis=-1)(self.output_layer(x))
        # print(f"prob : = ",x)
        return probs

    def evaluate(self, states,batch_size):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        probs = self.call(states,batch_size)

        # Avoid numerical instability.
        # Ref: https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/model.py#L98
        z = ((probs.numpy().all()).astype(float) == 0.0) * \
            self.noise  # avoid log(0)
        log_probs = tf.math.log(probs + z)

        return log_probs


    def get_action(self, states, deterministic=False,batch_size=1):
        probs = self.call(states,batch_size)
        # dist = tfp.distributions.Categorical(probs=probs)
        dist = tf.compat.v1.distributions.Categorical(probs=probs)

        if deterministic:
            # print(1)
            action = tf.argmax(probs, axis=1)
        else:
            action = dist.sample()
        return action.numpy()

class SAC():
    def __init__(self,s,replay_buffer, action_dim, hidden_dim, batch_size = 1 ):
        self.replay_buffer = replay_buffer
        # print(hidden_dim)
        self.soft_q_net1 = SoftQNetwork(
            action_dim, hidden_size=hidden_dim,batch_size=batch_size)

        self.soft_q_net2 = SoftQNetwork(
            action_dim, hidden_size=hidden_dim,batch_size=batch_size)

        self.target_soft_q_net1 = SoftQNetwork(
            action_dim, hidden_size=hidden_dim,batch_size=batch_size)

        self.target_soft_q_net2 = SoftQNetwork(
            action_dim, hidden_size=hidden_dim,batch_size=batch_size)

        self.policy_net = PolicyNetwork(
            action_dim, hidden_size=hidden_dim,batch_size=batch_size)

        self.batch_size = batch_size

        self.soft_q_criterion1 = tf.keras.losses.MeanSquaredError()
        self.soft_q_criterion2 = tf.keras.losses.MeanSquaredError()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.log_alpha = tf.Variable([0], trainable=True, validate_shape=True, dtype=tf.float32)

        self.soft_q_net1(s,batch_size= 1)
        self.soft_q_net2(s, batch_size= 1)
        self.target_soft_q_net1(s, batch_size= 1)
        self.target_soft_q_net2(s, batch_size= 1)
        self.policy_net(s, batch_size= 1)

        self.soft_q_net1.optimizer = tf.keras.optimizers.Adam(
            self.soft_q_net1.variables, lr=soft_q_lr)
        self.soft_q_net2.optimizer = tf.keras.optimizers.Adam(
            self.soft_q_net2.variables, lr=soft_q_lr)

        self.target_soft_q_net1.optimizer = tf.keras.optimizers.Adam(
            self.target_soft_q_net1.variables, lr=soft_q_lr)
        self.target_soft_q_net2.optimizer = tf.keras.optimizers.Adam(
            self.target_soft_q_net2.variables, lr=soft_q_lr)

        self.policy_net.optimizer= tf.keras.optimizers.Adam(
            self.policy_net.variables, lr=policy_lr)
        self.alpha_optimizer = tf.keras.optimizers.Adam([self.log_alpha], lr=alpha_lr)

        ################
        self.load_model(MODEL_PATH)



    def save_models(self):
        print('... saving models ...')
        self.soft_q_net1.save_weights(MODEL_PATH + "q1")
        self.soft_q_net2.save_weights(MODEL_PATH + "q2")
        self.policy_net.save_weights(MODEL_PATH + 'policy')

    def load_model(self, path):
        print('... loading models ...')
        # self.soft_q_net1.load_weights(MODEL_PATH + "q1")
        # self.soft_q_net2.load_weights(MODEL_PATH + "q2")
        self.policy_net.load_weights(MODEL_PATH + 'policy')

    def sampleAndLearn(self, batch_size= None, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,
                       soft_tau=1e-2):
        if(batch_size == None):
            batch_size = self.batch_size

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        if(batch_size == 1):
            state = state[0]
            next_state = next_state[0]
            states = state_transform([state], 1)
            next_states = state_transform([next_state], 1)
        else :
            states = state_transform(state, batch_size)
            next_states = state_transform(next_state, batch_size)

        actions = np.squeeze(action,axis=1)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)

        # print(states)
        # print(actions)
        predicted_q_value1 = self.soft_q_net1(states,batch_size)
        predicted_q_value2 = self.soft_q_net2(states,batch_size)
        # zzz = [[i, a] for i, a in enumerate(actions)]
        # print(zzz)
        indices = tf.constant([[i, a] for i, a in enumerate(actions)])

        predicted_q_value1 = tf.gather_nd(predicted_q_value1, indices)
        predicted_q_value2 = tf.gather_nd(predicted_q_value2, indices)
        # log_prob = self.policy_net.evaluate(states,batch_size)

        # with torch.no_grad():
        next_log_prob = self.policy_net.evaluate(next_states,batch_size)
        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            self.alpha = tf.math.exp(self.log_alpha)
            target_q_min = tf.math.exp(next_log_prob) * (tf.minimum
                                                             (self.target_soft_q_net1(next_states,batch_size=batch_size),
                                                              self.target_soft_q_net2(next_states,batch_size=batch_size))
                                                         - self.alpha * next_log_prob)
            target_q_min = tf.math.reduce_sum(target_q_min,axis = -1)
            target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value)
            q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value)

        target_soft_q_net1_gradient = tape.gradient(q_value_loss1, self.target_soft_q_net1.trainable_variables)
        target_soft_q_net2_gradient = tape.gradient(q_value_loss2, self.target_soft_q_net2.trainable_variables)

        self.target_soft_q_net1.optimizer.apply_gradients(
            zip(target_soft_q_net1_gradient, self.target_soft_q_net1.trainable_variables))
        self.target_soft_q_net2.optimizer.apply_gradients(
            zip(target_soft_q_net2_gradient, self.target_soft_q_net2.trainable_variables))

        # self.update_network_parameters()

        '''
        self.alpha = self.log_alpha.exp()
        target_q_min = (next_log_prob.exp() * (torch.min(self.target_soft_q_net1(next_state),self.target_soft_q_net2(next_state)) - self.alpha * next_log_prob)).sum(dim=-1).unsqueeze(-1)
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  
        '''

        # Training Policy Function
        predicted_new_q_value = tf.minimum(
            self.soft_q_net1(states, batch_size=batch_size),
            self.soft_q_net2(states, batch_size=batch_size))
        # print(f"actions = ", actions)
        # print(f"Predicted_new_q1 =  ", self.soft_q_net1(states, batch_size=batch_size))
        # print(f"Predicted_new_q2 =  ", self.soft_q_net2(states, batch_size=batch_size))
        with tf.GradientTape() as tape:
            log_prob = self.policy_net.evaluate(states, batch_size)
            policy_loss = tf.math.exp(log_prob) * (self.alpha * log_prob - predicted_new_q_value)
            policy_loss = tf.math.reduce_mean(tf.math.reduce_sum(policy_loss,axis = -1))

        # print(f"policy_loss : ",policy_loss)
        policy_network_gradient = tape.gradient(policy_loss, self.policy_net.trainable_variables)
        # print(f"policy_gradient: ",policy_network_gradient)
        self.policy_net.optimizer.apply_gradients(
            zip(policy_network_gradient, self.policy_net.trainable_variables))

        '''
        with torch.no_grad():
            predicted_new_q_value = torch.min(self.soft_q_net1(state),self.soft_q_net2(state))
        policy_loss = (log_prob.exp()*(self.alpha * log_prob - predicted_new_q_value)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        '''

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            with tf.GradientTape() as tape:
                log_prob = self.policy_net.evaluate(states, batch_size)   # log_prob placed at this place to caculate gradient
                alpha_loss = -(self.log_alpha * (log_prob + target_entropy))
                alpha_loss = tf.math.reduce_mean(alpha_loss)
                # print('alpha loss: ',alpha_loss)

            # self.alpha_optimizer.apply_gradients(None)
            alpha_gradient = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(
                zip(alpha_gradient, [self.log_alpha]))
            '''
            # self.alpha_optimizer.zero_grad()
            # alpha_loss.backward()
            # self.alpha_optimizer.step()
            '''
        else:
            self.alpha = 1.
            alpha_loss = 0

        # print('q loss: ', q_value_loss1.item(), q_value_loss2.item())
        # print('policy loss: ', policy_loss.item() )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.variables, self.soft_q_net1.variables):
            target_param.assign(  # copy data value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.variables, self.soft_q_net2.variables):
            target_param.assign(  # copy data value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )

        return tf.math.reduce_mean(predicted_new_q_value)

# Setting HyperParameter

AUTO_ENTROPY = True
DETERMINISTIC = True
rewards = []
action_dim = len(Action_Space)
target_entropy = -1 * action_dim
hidden_dim = 64
replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)


batch_size = BATCH_SIZE
update_itr = 1
# state1=tf.keras.layers.Input(shape=[5,batch_size,25])
# state2=tf.keras.layers.Input(shape=[batch_size,4])

# Get Initial Inputs
s_init = get_batch_states(batch_size)
# print(state)
s_init = state_transform(s_init,batch_size)
# print(state)

state = get_state()
state_init = state_transform([state],batch_size=1)
sac_trainer = SAC(state_init, action_dim=action_dim,
                  replay_buffer=replay_buffer, hidden_dim=hidden_dim,
                  batch_size=batch_size)

class envDASH():
    def __init__(self):
        self.bestScore = 0
        # self.abr = ABR.Algorithm()
        # self.abr_init = self.abr.Initial()
        # bandWidthEstimate
        self.magicWeights = [0.031497799353385, 0.030552865372783, 0.0296362794116, 0.028747191029252, 0.027884775298374, 0.027048232039423, 0.02623678507824, 0.025449681525893, 0.024686191080116, 0.023945605347713, 0.023227237187281, 0.022530420071663, 0.021854507469513, 0.021198872245428, 0.020562906078065, 0.019946018895723, 0.019347638328851, 0.018767209178986, 0.018204192903616, 0.017658067116508, 0.017128325103012, 0.016614475349922, 0.016116041089424, 0.015632559856742, 0.015163583061039, 0.014708675569208, 0.014267415302132, 0.013839392843068, 0.013424211057776, 0.013021484726043, 0.012630840184261, 0.012251914978734, 0.011884357529372, 0.01152782680349, 0.011181991999386, 0.010846532239404, 0.010521136272222, 0.010205502184055, 0.009899337118534, 0.009602357004978, 0.009314286294828, 0.009034857705983, 0.008763811974804, 0.00850089761556, 0.008245870687093, 0.00799849456648, 0.007758539729486, 0.007525783537601, 0.007300010031473, 0.007081009730529,
                             0.006868579438613, 0.006662522055455, 0.006462646393791, 0.006268767001977, 0.006080703991918, 0.005898282872161, 0.005721334385996, 0.005549694354416, 0.005383203523783, 0.00522170741807, 0.005065056195528, 0.004913104509662, 0.004765711374372, 0.004622740033141, 0.004484057832147, 0.004349536097182, 0.004219050014267, 0.004092478513839, 0.003969704158424, 0.003850613033671, 0.003735094642661, 0.003623041803381, 0.00351435054928, 0.003408920032801, 0.003306652431817, 0.003207452858863, 0.003111229273097, 0.003017892394904, 0.002927355623057, 0.002839534954365, 0.002754348905734, 0.002671718438562, 0.002591566885405, 0.002513819878843, 0.002438405282478, 0.002365253124003, 0.002294295530283, 0.002225466664375, 0.002158702664444, 0.00209394158451, 0.002031123336975, 0.001970189636866, 0.00191108394776, 0.001853751429327, 0.001798138886447, 0.001744194719854, 0.001691868878258, 0.00164111281191, 0.001591879427553, 0.001544123044726]
        self.queueLen = len(self.magicWeights)



    def reset(self, i, j):
        # -- Configuration variables --
        # Edit these variables to configure the simulator

        # Change which set of network trace to use: 'fixed' 'low' 'medium' 'high'
        netTraceList = os.listdir('./dataset/network_trace')
        self.NETWORK_TRACE = netTraceList[i % (len(netTraceList)-1)]
        ''' self.NETWORK_TRACE = 'fixed' '''

        # Change which set of video trace to use.
        videoTraceList = os.listdir('./dataset/video_trace')
        self.VIDEO_TRACE = videoTraceList[j % len(videoTraceList)] # j -> [0,5)
        ''' self.VIDEO_TRACE = 'AsianCup_China_Uzbekistan' '''

        # Turn on and off logging.  Set to 'True' to create log files.
        # Set to 'False' would speed up the simulator.
        self.DEBUG = False

        # Control the subdirectory where log files will be stored.
        self.LOG_FILE_PATH = './log/'

        # create result directory
        if not os.path.exists(self.LOG_FILE_PATH):
            os.makedirs(self.LOG_FILE_PATH)

        # -- End Configuration --
        # You shouldn't need to change the rest of the code here.

        self.network_trace_dir = './dataset/network_trace/' + self.NETWORK_TRACE + '/'
        self.video_trace_prefix = './dataset/video_trace/' + \
            self.VIDEO_TRACE + '/frame_trace_'

        # load the trace
        self.all_cooked_time, self.all_cooked_bw, self.all_file_names = load_trace.load_trace(
            self.network_trace_dir)
        # random_seed
        self.random_seed = 2
        self.count = 0
        self.trace_count = 1
        self.FPS = 25
        self.frame_time_len = 0.04
        self.reward_all_sum = 0
        self.run_time = 0
        # init
        # setting one:
        #     1,all_cooked_time : timestamp
        #     2,all_cooked_bw   : throughput
        #     3,all_cooked_rtt  : rtt
        #     4,agent_id        : random_seed
        #     5,logfile_path    : logfile_path
        #     6,VIDEO_SIZE_FILE : Video Size File Path
        #     7,Debug Setting   : Debug
        self.net_env = fixed_env.Environment(all_cooked_time=self.all_cooked_time,
                                             all_cooked_bw=self.all_cooked_bw,
                                             random_seed=self.random_seed,
                                             logfile_path=self.LOG_FILE_PATH,
                                             VIDEO_SIZE_FILE=self.video_trace_prefix,
                                             Debug=self.DEBUG)

        # BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # kpbs
        # TARGET_BUFFER = [0.5, 1.0]   # seconds

        self.cnt = 0
        # defalut setting
        self.last_bit_rate = 0
        self.bit_rate = 0
        self.target_buffer = 0
        self.latency_limit = 4

        # QOE setting
        self.reward_frame = 0
        self.reward_all = 0
        self.SMOOTH_PENALTY = 0.02
        self.REBUF_PENALTY = 1.85
        self.LANTENCY_PENALTY = 0.005
        self.SKIP_PENALTY = 0.5
        # past_info setting
        self.past_frame_num = 7500
        self.S_time_interval = [0] * self.past_frame_num
        self.S_send_data_size = [0] * self.past_frame_num
        self.S_chunk_len = [0] * self.past_frame_num
        self.S_rebuf = [0] * self.past_frame_num
        self.S_buffer_size = [0] * self.past_frame_num
        self.S_end_delay = [0] * self.past_frame_num
        self.S_chunk_size = [0] * self.past_frame_num
        self.S_play_time_len = [0] * self.past_frame_num
        self.S_decision_flag = [0] * self.past_frame_num
        self.S_buffer_flag = [0] * self.past_frame_num
        self.S_cdn_flag = [0] * self.past_frame_num
        self.S_skip_time = [0] * self.past_frame_num
        # params setting
        self.call_time_sum = 0

    def trainStep(self,state):
        while True:
            self.reward_frame = 0
            '''
            # input the train steps
            # if cnt > 5000:
            # plt.ioff()
            #    break
            # actions bit_rate  target_buffer
            # every steps to call the environment
            # time           : physical time
            # time_interval  : time duration in self.step
            # send_data_size : download frame data size in self.step
            # chunk_len      : frame time len
            # rebuf          : rebuf time in self.step
            # buffer_size    : current client buffer_size in self.step
            # rtt            : current buffer  in self.step
            # play_time_len  : played time len  in self.step
            # end_delay      : end to end latency which means the (upload end timestamp - play end timestamp)
            # decision_flag  : Only in decision_flag is True ,you can choose the new actions, other time can't Becasuse the Gop is consist by the I frame and P frame. Only in I frame you can skip your frame
            # buffer_flag    : If the True which means the video is rebuffing , client buffer is rebuffing, no play the video
            # cdn_flag       : If the True cdn has no frame to get
            # end_of_video   : If the True ,which means the video is over.
            '''
            time, time_interval, send_data_size, chunk_len,\
                rebuf, buffer_size, play_time_len, end_delay,\
                cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag,\
                buffer_flag, cdn_flag, skip_flag, end_of_video = self.net_env.get_video_frame(
                    self.bit_rate, self.target_buffer, self.latency_limit)
            # S_info is sequential order
            self.S_time_interval.pop(0)
            self.S_send_data_size.pop(0)
            self.S_chunk_len.pop(0)
            self.S_buffer_size.pop(0)
            self.S_rebuf.pop(0)
            self.S_end_delay.pop(0)
            self.S_play_time_len.pop(0)
            self.S_decision_flag.pop(0)
            self.S_buffer_flag.pop(0)
            self.S_cdn_flag.pop(0)
            self.S_skip_time.pop(0)

            self.S_time_interval.append(time_interval)
            self.S_send_data_size.append(send_data_size)
            self.S_chunk_len.append(chunk_len)
            self.S_buffer_size.append(buffer_size)
            self.S_rebuf.append(rebuf)
            self.S_end_delay.append(end_delay)
            self.S_play_time_len.append(play_time_len)
            self.S_decision_flag.append(decision_flag)
            self.S_buffer_flag.append(buffer_flag)
            self.S_cdn_flag.append(cdn_flag)
            self.S_skip_time.append(skip_frame_time_len)

            # QOE setting
            if end_delay <= 1.0:
                self.LANTENCY_PENALTY = 0.005
            else:
                self.LANTENCY_PENALTY = 0.01

            if not cdn_flag:
                self.reward_frame = self.frame_time_len * \
                    float(BIT_RATE[self.bit_rate]) / 1000 - self.REBUF_PENALTY * rebuf - \
                    self.LANTENCY_PENALTY * end_delay - self.SKIP_PENALTY * skip_frame_time_len
            else:
                self.reward_frame = -(self.REBUF_PENALTY * rebuf)
            if decision_flag or end_of_video:
                # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
                self.reward_frame += -1 * self.SMOOTH_PENALTY * \
                    (abs(BIT_RATE[self.bit_rate] -
                     BIT_RATE[self.last_bit_rate]) / 1000)
                # last_bit_rate
                self.last_bit_rate = self.bit_rate

                # ----------------- Your Algorithm ---------------------
                # which part is the algorithm part ,the buffer based ,
                # if the buffer is enough ,choose the high quality
                # if the buffer is danger, choose the low  quality
                # if there is no rebuf ,choose the low target_buffer
                self.cnt += 1
                timestamp_start = tm.time()

                ############################################################################################
                ##get next state begin:
                # bandWidth
                next_state = []

                next_state.append(sum(self.S_buffer_size)/len(self.S_buffer_size)/10.0) #1
                next_state.append(sum(self.S_end_delay)/len(self.S_end_delay)/2.0) #2
                next_state.append(sum(self.S_rebuf)/len(self.S_rebuf)/5.0) #3
                next_state.append(sum(self.S_send_data_size)/len(self.S_send_data_size)/3000.0) #4
                next_state.append(sum(self.S_time_interval)/len(self.S_time_interval)/20.0) #5
                next_state.append(sum(self.S_skip_time)/len(self.S_skip_time)/10.0) #6
                next_state.append(sum(self.S_chunk_len)/len(self.S_chunk_len)/2000.0) #7
                # next_state.append(sum()/len())

                next_state.append(BIT_RATE[self.bit_rate]/1000.0)                      # 8
                next_state.append(TARGET_BUFFER[self.target_buffer])         # 9

                bandWidth = 0
                Slen = len(self.S_send_data_size)
                for tmp in range(self.queueLen):
                    bandWidth += self.S_send_data_size[Slen+tmp-self.queueLen] * self.magicWeights[self.queueLen-1-tmp]
                next_state.append(bandWidth/3500.0)                  # 10

                ##get next state end;
                done = end_of_video and self.trace_count >= len(self.all_file_names)
                reward = self.reward_frame

                action = sac_trainer.policy_net.get_action(state_transform([state], 1), deterministic=DETERMINISTIC,batch_size= 1)
                replay_buffer.push(state, action, reward, next_state, done)  # push to buffer
                state = next_state

                if(len(replay_buffer) < batch_size):
                    x_predict = sac_trainer.sampleAndLearn(batch_size=1, reward_scale=1.,
                               auto_entropy=AUTO_ENTROPY, gamma=0.99,
                               target_entropy=target_entropy, soft_tau=1e-2)  # train_for_once
                else:
                    x_predict = sac_trainer.sampleAndLearn(batch_size= batch_size, reward_scale=1.,
                                                           auto_entropy=AUTO_ENTROPY, gamma=0.99,
                                                           target_entropy=target_entropy,
                                                           soft_tau=1e-2)  # train_for_once
                # print(f"x_predict: ",x_predict)


                self.bit_rate, self.target_buffer,self.latency_limit = action % 4, action // 4, 4
                self.bit_rate = self.bit_rate[0]             # get the value in list
                self.target_buffer = self.target_buffer[0]   # get the value in list
                # print(f"bit_rate",self.bit_rate)
                # print(f"target_buffer",self.target_buffer)
                # print(f"latency_buffer",self.latency_limit)
                # print(f"state",state)
                # print(f"reward",reward)
                if len(replay_buffer) > batch_size:
                    # print("-----------Start_Training-----------")
                    for i in range(update_itr):

                        acc =sac_trainer.sampleAndLearn(batch_size,
                                                     reward_scale=1.,
                                                     auto_entropy=AUTO_ENTROPY,
                                                     target_entropy=target_entropy)
                    #     print("Iteration ",i," ---||---acc---",acc)
                    # print("-------End_OneBuffer_Training-------")
                ############################################################################################

                '''self.bit_rate, self.target_buffer, self.latency_limit = self.abr.run(time,
                                                                                self.S_time_interval,
                                                                                self.S_send_data_size,
                                                                                self.S_chunk_len,
                                                                                self.S_rebuf,
                                                                                self.S_buffer_size,
                                                                                self.S_play_time_len,
                                                                                self.S_end_delay,
                                                                                self.S_decision_flag,
                                                                                self.S_buffer_flag,
                                                                                self.S_cdn_flag,
                                                                                self.S_skip_time,
                                                                                end_of_video,
                                                                                cdn_newest_id,
                                                                                download_id,
                                                                                cdn_has_frame,
                                                                                self.abr_init)
                '''
                timestamp_end = tm.time()
                self.call_time_sum += timestamp_end - timestamp_start
                # -------------------- End --------------------------------

            if end_of_video:
                print("network traceID, network_reward, avg_running_time\n",
                      self.trace_count, self.reward_all, self.call_time_sum/self.cnt)
                self.reward_all_sum += self.reward_all
                self.run_time += self.call_time_sum / self.cnt
                if self.trace_count >= len(self.all_file_names): #where we break
                    if(self.reward_all_sum/self.trace_count>self.bestScore):
                        self.bestScore = self.reward_all_sum/self.trace_count
                    sac_trainer.save_models()
                    print("bestScore:",self.bestScore,"; nowScore:",self.reward_all_sum/self.trace_count)
                    break
                self.trace_count += 1
                self.cnt = 0

                self.call_time_sum = 0
                self.last_bit_rate = 0
                self.reward_all = 0
                self.bit_rate = 0
                self.target_buffer = 0

                self.S_time_interval = [0] * self.past_frame_num
                self.S_send_data_size = [0] * self.past_frame_num
                self.S_chunk_len = [0] * self.past_frame_num
                self.S_rebuf = [0] * self.past_frame_num
                self.S_buffer_size = [0] * self.past_frame_num
                self.S_end_delay = [0] * self.past_frame_num
                self.S_chunk_size = [0] * self.past_frame_num
                self.S_play_time_len = [0] * self.past_frame_num
                self.S_decision_flag = [0] * self.past_frame_num
                self.S_buffer_flag = [0] * self.past_frame_num
                self.S_cdn_flag = [0] * self.past_frame_num

            self.reward_all += self.reward_frame

        return [self.reward_all_sum / self.trace_count, self.run_time / self.trace_count]


# Runer = envDASH()
# for j in range(5): # four different videos
#     if (j == 0):
#         continue
#     for i in range(8): # four different network conditions cycle | each (50/4) times
#         print(f"EPOCH : ", 8*(j-1)+i)
#         Runer.reset(i,j)
#         Runer.trainStep(state = state)



class Algorithm:
     def __init__(self):
     # fill your self params
         self.buffer_size = 0
         self.a_dim =4

         self.magicWeights = [0.031497799353385, 0.030552865372783, 0.0296362794116, 0.028747191029252, 0.027884775298374,
                          0.027048232039423, 0.02623678507824, 0.025449681525893, 0.024686191080116, 0.023945605347713,
                          0.023227237187281, 0.022530420071663, 0.021854507469513, 0.021198872245428, 0.020562906078065,
                          0.019946018895723, 0.019347638328851, 0.018767209178986, 0.018204192903616, 0.017658067116508,
                          0.017128325103012, 0.016614475349922, 0.016116041089424, 0.015632559856742, 0.015163583061039,
                          0.014708675569208, 0.014267415302132, 0.013839392843068, 0.013424211057776, 0.013021484726043,
                          0.012630840184261, 0.012251914978734, 0.011884357529372, 0.01152782680349, 0.011181991999386,
                          0.010846532239404, 0.010521136272222, 0.010205502184055, 0.009899337118534, 0.009602357004978,
                          0.009314286294828, 0.009034857705983, 0.008763811974804, 0.00850089761556, 0.008245870687093,
                          0.00799849456648, 0.007758539729486, 0.007525783537601, 0.007300010031473, 0.007081009730529,
                          0.006868579438613, 0.006662522055455, 0.006462646393791, 0.006268767001977, 0.006080703991918,
                          0.005898282872161, 0.005721334385996, 0.005549694354416, 0.005383203523783, 0.00522170741807,
                          0.005065056195528, 0.004913104509662, 0.004765711374372, 0.004622740033141, 0.004484057832147,
                          0.004349536097182, 0.004219050014267, 0.004092478513839, 0.003969704158424, 0.003850613033671,
                          0.003735094642661, 0.003623041803381, 0.00351435054928, 0.003408920032801, 0.003306652431817,
                          0.003207452858863, 0.003111229273097, 0.003017892394904, 0.002927355623057, 0.002839534954365,
                          0.002754348905734, 0.002671718438562, 0.002591566885405, 0.002513819878843, 0.002438405282478,
                          0.002365253124003, 0.002294295530283, 0.002225466664375, 0.002158702664444, 0.00209394158451,
                          0.002031123336975, 0.001970189636866, 0.00191108394776, 0.001853751429327, 0.001798138886447,
                          0.001744194719854, 0.001691868878258, 0.00164111281191, 0.001591879427553, 0.001544123044726]
         self.queueLen = len(self.magicWeights)
         self.bit_rate =0
         self.target_buffer =0

     # Intial
     def Initial(self):
     # Initail your session or something
     # restore neural net parameters
         self.buffer_size = 0
     #Define your al
     def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,S_end_delay, S_decision_flag, S_buffer_flag,S_cdn_flag,S_skip_time, end_of_video, cdn_newest_id,download_id,cdn_has_frame,IntialVars):


         RESEVOIR = 0.5
         CUSHION =  1.5
         next_state = []

         next_state.append(sum(S_buffer_size) / len(S_buffer_size) / 10.0)  # 1
         next_state.append(sum(S_end_delay) / len(S_end_delay) / 2.0)  # 2
         next_state.append(sum(S_rebuf) / len(S_rebuf) / 5.0)  # 3
         next_state.append(sum(S_send_data_size) / len(S_send_data_size) / 3000.0)  # 4
         next_state.append(sum(S_time_interval) / len(S_time_interval) / 20.0)  # 5
         next_state.append(sum(S_skip_time) / len(S_skip_time) / 10.0)  # 6
         next_state.append(sum(S_chunk_len) / len(S_chunk_len) / 2000.0)  # 7
         # next_state.append(sum()/len())

         next_state.append(BIT_RATE[self.bit_rate] / 1000.0)  # 8
         next_state.append(TARGET_BUFFER[self.target_buffer])  # 9

         bandWidth = 0
         Slen = len(S_send_data_size)
         for tmp in range(self.queueLen):
             bandWidth += S_send_data_size[Slen + tmp - self.queueLen] * self.magicWeights[self.queueLen - 1 - tmp]
         next_state.append(bandWidth / 3500.0)

         next_state = state_transform([next_state], 1)
         action = sac_trainer.policy_net.get_action(next_state,
                                                    deterministic=True,
                                                    batch_size=1)
         self.bit_rate, self.target_buffer, self.latency_limit = action % 4, action // 4, 4
         self.target_buffer = self.target_buffer[0]  # get the value in list
         self.bit_rate = self.bit_rate[0]            # get the value in list

         latency_limit = 4
         bit_rate =  self.bit_rate
         target_buffer = self.target_buffer

         return bit_rate, target_buffer, latency_limit


     def get_params(self):
     # get your params
        your_params = []
        return your_params

     def learn(self):
         return 0





