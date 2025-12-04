"""
DC-IDS 已知流量细粒度分类模块 - 完整实现
基于深度强化学习和类条件变分自动编码器的工业物联网入侵检测模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 超参数配置 ====================
class Hyperparameters:
    # DQN参数
    GAMMA = 0.99           # 折扣因子
    LR = 1e-4              # 学习率
    BATCH_SIZE = 64        # 批量大小
    TARGET_UPDATE = 10     # 目标网络更新频率
    MEMORY_SIZE = 10000    # 经验回放缓冲区大小
    EPS_START = 0.9        # ε-greedy起始值
    EPS_END = 0.1          # ε-greedy最小值
    EPS_DECAY = 200        # ε衰减率
    
    # CVAE参数
    LATENT_DIM = 32        # 潜在变量维度
    HIDDEN_DIM = 128       # 隐藏层维度
    KL_WEIGHT = 1.0        # KL散度权重
    
    # 训练参数
    NUM_EPISODES = 500     # 训练回合数（可调整）
    STEPS_PER_EPISODE = 100 # 每回合步数
    NUM_CLASSES = 4        # 类别数（Normal+DDoS+Password+XSS）
    
    # 数据参数
    STATE_DIM = 105        # 状态维度（特征数）
    ACTION_DIM = NUM_CLASSES  # 动作维度

# ==================== CVAE 网络结构 ====================
class PriorNetwork(nn.Module):
    """先验网络 p(z|s)"""
    def __init__(self, state_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, s):
        h = F.softplus(self.fc1(s))
        h = F.softplus(self.fc2(h))
        mu = self.fc3_mu(h)
        logvar = self.fc3_logvar(h)
        return mu, logvar


class RecognitionNetwork(nn.Module):
    """识别网络 q(z|s,a)"""
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, s, a):
        # a需要是one-hot编码
        x = torch.cat([s, a], dim=-1)
        h = F.softplus(self.fc1(x))
        h = F.softplus(self.fc2(h))
        mu = self.fc3_mu(h)
        logvar = self.fc3_logvar(h)
        return mu, logvar


class CVAE(nn.Module):
    """类条件变分自动编码器"""
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.prior_net = PriorNetwork(state_dim, hidden_dim, latent_dim)
        self.recog_net = RecognitionNetwork(state_dim, action_dim, hidden_dim, latent_dim)
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, s, a=None):
        """
        前向传播
        Args:
            s: 状态 [batch_size, state_dim]
            a: 动作(标签) [batch_size, action_dim], one-hot编码
        Returns:
            z: 潜在变量
            prior_mu, prior_logvar: 先验分布参数
            recog_mu, recog_logvar: 后验分布参数
        """
        # 先验网络
        prior_mu, prior_logvar = self.prior_net(s)
        
        if a is not None:
            # 训练时使用识别网络
            recog_mu, recog_logvar = self.recog_net(s, a)
            z = self.reparameterize(recog_mu, recog_logvar)
            return z, prior_mu, prior_logvar, recog_mu, recog_logvar
        else:
            # 测试时使用先验网络
            z = self.reparameterize(prior_mu, prior_logvar)
            return z, prior_mu, prior_logvar, None, None
    
    def compute_kl_loss(self, prior_mu, prior_logvar, recog_mu, recog_logvar):
        """计算KL散度损失"""
        # KL(q(z|s,a) || p(z|s))
        kl_loss = -0.5 * torch.sum(1 + recog_logvar - prior_logvar - 
                                   (recog_logvar.exp() + (recog_mu - prior_mu)**2) / prior_logvar.exp())
        return kl_loss

# ==================== DQN 网络结构 ====================
class QNetwork(nn.Module):
    """Q网络（主网络和目标网络）"""
    def __init__(self, state_dim, latent_dim, action_dim, hidden_dim):
        super().__init__()
        # 输入：状态特征 + 潜在变量
        input_dim = state_dim + latent_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)  # Q值输出
        
    def forward(self, s, z):
        x = torch.cat([s, z], dim=-1)
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


class DCIDSNetwork(nn.Module):
    """DC-IDS完整网络结构"""
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.cvae = CVAE(state_dim, action_dim, hidden_dim, latent_dim)
        self.q_net = QNetwork(state_dim, latent_dim, action_dim, hidden_dim)
        self.target_q_net = QNetwork(state_dim, latent_dim, action_dim, hidden_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
    def forward(self, s, a=None, mode='train'):
        """
        Args:
            mode: 'train' 或 'test'
        """
        if mode == 'train':
            # 训练时使用识别网络
            z, prior_mu, prior_logvar, recog_mu, recog_logvar = self.cvae(s, a)
            q_values = self.q_net(s, z)
            target_q_values = self.target_q_net(s, z)
            return q_values, target_q_values, prior_mu, prior_logvar, recog_mu, recog_logvar
        else:
            # 测试时使用先验网络
            z, prior_mu, prior_logvar, _, _ = self.cvae(s)
            q_values = self.q_net(s, z)
            return q_values, z

# ==================== 经验回放缓冲区 ====================
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done', 'true_action'])

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, true_action):
        """存储经验"""
        experience = Experience(state, action, reward, next_state, done, true_action)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        true_actions = np.array([exp.true_action for exp in batch])
        
        return (states, actions, rewards, next_states, dones, true_actions)
    
    def __len__(self):
        return len(self.buffer)

# ==================== DQN 智能体 ====================
class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_dim, action_dim, device='cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 初始化网络
        self.network = DCIDSNetwork(
            state_dim, action_dim, 
            Hyperparameters.HIDDEN_DIM, 
            Hyperparameters.LATENT_DIM
        ).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=Hyperparameters.LR)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(Hyperparameters.MEMORY_SIZE)
        
        # ε-greedy参数
        self.eps = Hyperparameters.EPS_START
        self.eps_end = Hyperparameters.EPS_END
        self.eps_decay = Hyperparameters.EPS_DECAY
        
        # 训练步数
        self.steps_done = 0
        
    def select_action(self, state, train=True):
        """选择动作（ε-greedy策略）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 计算当前ε值
        if train:
            self.eps = Hyperparameters.EPS_END + (Hyperparameters.EPS_START - Hyperparameters.EPS_END) * \
                      np.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
        
        # ε-greedy选择
        if train and np.random.random() < self.eps:
            # 随机选择动作
            return np.random.randint(self.action_dim)
        else:
            # 选择Q值最大的动作
            with torch.no_grad():
                q_values, _ = self.network(state_tensor, mode='test')
                return q_values.argmax().item()
    
    def train_step(self):
        """执行一次训练步骤"""
        if len(self.memory) < Hyperparameters.BATCH_SIZE:
            return None, None
        
        # 采样一批经验
        states, actions, rewards, next_states, dones, true_actions = \
            self.memory.sample(Hyperparameters.BATCH_SIZE)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 将真实动作转换为one-hot编码
        true_actions_onehot = F.one_hot(
            torch.LongTensor(true_actions), self.action_dim
        ).float().to(self.device)
        
        # 前向传播
        q_values, target_q_values, prior_mu, prior_logvar, recog_mu, recog_logvar = \
            self.network(states, true_actions_onehot, mode='train')
        
        # 获取当前状态的Q值
        current_q = q_values.gather(1, actions).squeeze()
        
        # 使用Double-DQN计算目标Q值
        with torch.no_grad():
            # 使用主网络选择动作
            next_q_values, _ = self.network(next_states, mode='test')
            next_actions = next_q_values.argmax(1, keepdim=True)
            
            # 使用目标网络评估Q值
            next_target_q_values = self.network.target_q_net(
                next_states, 
                self.network.cvae(next_states)[0]  # 只取潜在变量
            )
            next_q = next_target_q_values.gather(1, next_actions).squeeze()
            target_q = rewards + Hyperparameters.GAMMA * next_q * (1 - dones)
        
        # 计算DQN损失
        dqn_loss = F.mse_loss(current_q, target_q)
        
        # 计算CVAE的KL损失
        kl_loss = self.network.cvae.compute_kl_loss(
            prior_mu, prior_logvar, recog_mu, recog_logvar
        ) * Hyperparameters.KL_WEIGHT / Hyperparameters.BATCH_SIZE
        
        # 总损失
        total_loss = dqn_loss + kl_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # 定期更新目标网络
        if self.steps_done % Hyperparameters.TARGET_UPDATE == 0:
            self.network.target_q_net.load_state_dict(self.network.q_net.state_dict())
        
        return dqn_loss.item(), kl_loss.item()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'eps': self.eps
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.eps = checkpoint['eps']

# ==================== 入侵检测环境 ====================
class IntrusionDetectionEnv:
    """入侵检测环境（离散时间的马尔可夫决策过程）"""
    def __init__(self, data, labels, test_size=0.2):
        """
        Args:
            data: 流量特征数据 [n_samples, n_features]
            labels: 流量标签 [n_samples]
        """
        # 划分训练集和验证集
        self.train_data, self.val_data, self.train_labels, self.val_labels = \
            train_test_split(data, labels, test_size=test_size, stratify=labels)
        
        self.current_idx = 0
        self.num_samples = len(self.train_data)
        
        # 随机打乱数据
        self.shuffle_data()
        
    def shuffle_data(self):
        """随机打乱数据"""
        indices = np.random.permutation(self.num_samples)
        self.train_data = self.train_data[indices]
        self.train_labels = self.train_labels[indices]
        self.current_idx = 0
    
    def reset(self):
        """重置环境"""
        self.shuffle_data()
        state = self.train_data[0]
        true_action = self.train_labels[0]
        return state, true_action
    
    def step(self, action_idx):
        """
        执行一步动作
        Args:
            action_idx: 智能体选择的动作索引
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            true_action: 真实动作
        """
        # 计算奖励
        true_action = self.train_labels[self.current_idx]
        reward = 1 if action_idx == true_action else -1
        
        # 获取下一个状态
        self.current_idx += 1
        
        if self.current_idx >= self.num_samples:
            done = True
            next_state = self.train_data[0]
            next_true_action = self.train_labels[0]
        else:
            done = False
            next_state = self.train_data[self.current_idx]
            next_true_action = self.train_labels[self.current_idx]
        
        return next_state, reward, done, next_true_action
    
    def get_val_data(self):
        """获取验证集数据"""
        return self.val_data, self.val_labels

# ==================== 数据预处理 ====================
class DataPreprocessor:
    """数据预处理类"""
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='most_frequent')
        
    def load_and_preprocess(self, filepath):
        """加载并预处理数据 - 修改版"""
        # 1. 加载数据
        df = pd.read_csv(filepath)
        
        print(f"原始数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 2. 合并日期和时间，转换为时间戳
        if 'date' in df.columns and 'time' in df.columns:
            # 将日期和时间合并为一个datetime列
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            
            # 从datetime提取有用的时间特征
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            
            # 删除原始日期时间列
            df = df.drop(['date', 'time', 'datetime'], axis=1)
        
        # 3. 处理分类特征（如temp_condition）
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
        
        print(f"分类特征列: {categorical_cols}")
        
        # 对分类特征进行标签编码或独热编码
        label_encoders = {}
        for col in categorical_cols:
            if col not in ['label', 'type']:  # 跳过标签列
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # 4. 根据type列创建细粒度标签（多分类标签）
        # 注意：这里label已经是0/1，但我们需要基于type创建多分类标签
        if 'type' in df.columns:
            print(f"攻击类型分布:")
            print(df['type'].value_counts())
            
            # 创建细粒度标签编码器
            type_encoder = LabelEncoder()
            df['fine_grained_label'] = type_encoder.fit_transform(df['type'])
            
            # 打印标签映射
            print(f"细粒度标签映射:")
            for i, label in enumerate(type_encoder.classes_):
                print(f"  {label}: {i}")
            
            # 保存攻击类型映射
            self.type_encoder = type_encoder
        else:
            # 如果没有type列，使用原始label作为细粒度标签（只有0/1）
            df['fine_grained_label'] = df['label']
        
        # 5. 分离特征和标签
        # 删除标签列，保留特征
        feature_cols = [col for col in df.columns if col not in ['label', 'type', 'fine_grained_label']]
        features = df[feature_cols].values
        fine_labels = df['fine_grained_label'].values
        binary_labels = df['label'].values if 'label' in df.columns else None
        
        print(f"特征维度: {features.shape}")
        print(f"细粒度标签类别数: {len(np.unique(fine_labels))}")
        print(f"特征列: {feature_cols}")
        
        # 6. 处理缺失值
        if np.isnan(features).any():
            print("处理缺失值...")
            features = SimpleImputer(strategy='mean').fit_transform(features)
        
        # 7. 特征归一化
        features = self.scaler.fit_transform(features)
        
        return features, fine_labels, binary_labels, df['type'].values if 'type' in df.columns else None
    
    def split_known_unknown(self, features, labels, known_classes=None):
        """
        简化版：直接根据标签划分
        Args:
            features: 特征矩阵
            labels: 标签数组
            known_classes: 已知类别的标签列表
        """
        if known_classes is None:
            # 假设标签0,1,2,3是已知的
            known_classes = [0, 1, 2, 3]
        
        known_mask = np.isin(labels, known_classes)
        unknown_mask = ~known_mask
        
        known_features = features[known_mask]
        known_labels = labels[known_mask]
        unknown_features = features[unknown_mask]
        unknown_labels = labels[unknown_mask]
        
        # 重新映射已知标签为0到N-1
        unique_known = np.unique(known_labels)
        label_mapping = {old: new for new, old in enumerate(unique_known)}
        known_labels = np.array([label_mapping[l] for l in known_labels])
        
        return known_features, known_labels, unknown_features, unknown_labels
    
    def create_sample_dataset(self, n_samples=2000):
        """创建示例数据集（返回3个值）"""
        np.random.seed(42)
        n_features = Hyperparameters.STATE_DIM
        
        # 创建随机特征
        features = np.random.randn(n_samples, n_features)
        
        # 创建标签（4个类别）
        labels = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.25, 0.25])
        
        return features, labels, None  # 返回3个值，attack_types设为None

# ==================== 训练函数 ====================
def train_dc_ids(data_path=None, use_sample_data=False):
    """训练DC-IDS模型 - 修改版"""
    # 1. 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2. 加载和预处理数据
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    if use_sample_data or data_path is None:
        print("Using sample dataset...")
        features, labels, _ = preprocessor.create_sample_dataset()  # 接收3个值
        known_features, known_labels, _, _ = preprocessor.split_known_unknown(
            features, labels
        )
    else:
        features, labels, _, _ = preprocessor.load_and_preprocess(data_path)  # 接收4个值
        known_features, known_labels, _, _ = preprocessor.split_known_unknown(
            features, labels
        )
    
    print(f"原始数据形状: {features.shape}")
    print(f"细粒度标签类别数: {len(np.unique(fine_labels))}")
    
    # 3. 划分已知和未知攻击
    known_features, known_labels, known_attack_types, unknown_features, unknown_labels, unknown_attack_types = \
        preprocessor.split_known_unknown(
            features, fine_labels, attack_types,
            known_attack_names=['ddos', 'password', 'xss'],  # 已知攻击
            include_normal=True  # 包含正常流量
        )
    
    # 更新超参数中的类别数
    Hyperparameters.NUM_CLASSES = len(np.unique(known_labels))
    Hyperparameters.ACTION_DIM = Hyperparameters.NUM_CLASSES
    Hyperparameters.STATE_DIM = known_features.shape[1]
    
    print(f"\n训练参数:")
    print(f"已知流量数: {len(known_features)}")
    print(f"特征维度: {known_features.shape[1]}")
    print(f"类别数: {Hyperparameters.NUM_CLASSES}")
    print(f"已知攻击类型: {np.unique(known_attack_types)}")
    print(f"未知攻击类型: {np.unique(unknown_attack_types)}")
    
    # 4. 创建环境（只使用已知流量）
    print("\nCreating environment...")
    env = IntrusionDetectionEnv(known_features, known_labels)
    
    # 5. 创建智能体
    print("Initializing agent...")
    agent = DQNAgent(
        state_dim=known_features.shape[1],
        action_dim=Hyperparameters.NUM_CLASSES,
        device=device
    )
    
    # 6. 训练循环（保持不变）

    print("Starting training...")
    episode_rewards = []
    dqn_losses = []
    kl_losses = []
    
    for episode in tqdm(range(Hyperparameters.NUM_EPISODES)):
        state, true_action = env.reset()
        episode_reward = 0
        
        for step in range(Hyperparameters.STEPS_PER_EPISODE):
            # 选择动作
            action = agent.select_action(state, train=True)
            
            # 执行动作，获取下一个状态
            next_state, reward, done, next_true_action = env.step(action)
            
            # 存储经验
            agent.memory.push(
                state, action, reward, next_state, done, true_action
            )
            
            # 训练一步
            dqn_loss, kl_loss = agent.train_step()
            
            if dqn_loss is not None:
                dqn_losses.append(dqn_loss)
                kl_losses.append(kl_loss)
            
            # 更新状态
            state = next_state
            true_action = next_true_action
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 每50回合输出一次进度
        if episode % 50 == 0:
            val_data, val_labels = env.get_val_data()
            accuracy = evaluate(agent, val_data, val_labels)
            print(f"\nEpisode {episode}, Reward: {episode_reward:.2f}, "
                  f"Val Accuracy: {accuracy:.4f}")
    
    # 7. 保存模型
    agent.save_model('dc_ids_model.pth')
    print(f"Model saved to dc_ids_model.pth")
    
    # 8. 可视化训练结果
    plot_training_results(episode_rewards, dqn_losses, kl_losses)
    
    return agent

# ==================== 评估函数 ====================
def evaluate(agent, val_data, val_labels):
    """评估模型在验证集上的表现"""
    correct = 0
    total = len(val_data)
    
    agent.network.eval()
    with torch.no_grad():
        for i in range(total):
            state = val_data[i]
            true_label = val_labels[i]
            
            # 选择动作（测试模式）
            action = agent.select_action(state, train=False)
            
            if action == true_label:
                correct += 1
    
    agent.network.train()
    return correct / total

def test_model(model_path, data_path=None, use_sample_data=False):
    """测试训练好的模型"""
    # 1. 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 加载和预处理数据
    preprocessor = DataPreprocessor()
    
    if use_sample_data or data_path is None:
        print("Using sample dataset for testing...")
        # 创建示例数据（现在返回3个值）
        features, fine_labels, attack_types = preprocessor.create_sample_dataset()
        known_features, known_labels, unknown_features, unknown_labels = \
            preprocessor.split_known_unknown(features, fine_labels)
    else:
        # 加载真实数据（返回4个值）
        features, fine_labels, binary_labels, attack_types = preprocessor.load_and_preprocess(data_path)
        known_features, known_labels, unknown_features, unknown_labels = \
            preprocessor.split_known_unknown(features, fine_labels)
    
    # 3. 加载模型
    agent = DQNAgent(
        state_dim=known_features.shape[1],
        action_dim=Hyperparameters.NUM_CLASSES,
        device=device
    )
    agent.load_model(model_path)
    
    # 4. 评估模型
    print(f"Testing on {len(known_features)} known traffic samples...")
    accuracy = evaluate(agent, known_features, known_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 5. 生成混淆矩阵
    generate_confusion_matrix(agent, known_features, known_labels)
    
    return agent

# ==================== 可视化函数 ====================
def plot_training_results(episode_rewards, dqn_losses, kl_losses):
    """绘制训练结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 回合奖励
    axes[0].plot(episode_rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True)
    
    # DQN损失
    if dqn_losses:
        axes[1].plot(dqn_losses[:1000])  # 只显示前1000步
        axes[1].set_title('DQN Loss (First 1000 Steps)')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
    
    # KL损失
    if kl_losses:
        axes[2].plot(kl_losses[:1000])  # 只显示前1000步
        axes[2].set_title('KL Loss (First 1000 Steps)')
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()

def generate_confusion_matrix(agent, features, labels):
    """生成混淆矩阵"""
    predictions = []
    
    agent.network.eval()
    with torch.no_grad():
        for i in range(len(features)):
            state = features[i]
            action = agent.select_action(state, train=False)
            predictions.append(action)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

# ==================== 主函数 ====================
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DC-IDS 已知流量分类模块')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'], help='运行模式')
    parser.add_argument('--data_path', type=str, default=None, 
                       help='数据文件路径')
    parser.add_argument('--model_path', type=str, default='dc_ids_model.pth', 
                       help='模型文件路径')
    parser.add_argument('--use_sample', action='store_true', 
                       help='使用示例数据')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("=" * 50)
        print("DC-IDS 训练模式")
        print("=" * 50)
        agent = train_dc_ids(args.data_path, args.use_sample)
        print("训练完成！")
        
        # 测试训练好的模型
        print("\n" + "=" * 50)
        print("测试训练好的模型")
        print("=" * 50)
        test_model('dc_ids_model.pth', args.data_path, args.use_sample)
        
    elif args.mode == 'test':
        print("=" * 50)
        print("DC-IDS 测试模式")
        print("=" * 50)
        agent = test_model(args.model_path, args.data_path, args.use_sample)
        print("测试完成！")
    
    print("\n所有任务完成！")

# ==================== 模型使用示例 ====================
def example_usage():
    """使用示例"""
    print("DC-IDS 模型使用示例")
    print("-" * 30)
    
    # 1. 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 2. 加载数据
    print("1. 加载数据...")
    if False:  # 如果有真实数据
        features, labels = preprocessor.load_and_preprocess('ton_iot.csv')
    else:  # 使用示例数据
        features, labels = preprocessor.create_sample_dataset()
    
    # 3. 划分已知和未知数据
    print("2. 划分已知和未知数据...")
    known_features, known_labels, _, _ = preprocessor.split_known_unknown(
        features, labels, known_classes=[0, 1, 2, 3]
    )
    
    # 4. 创建环境
    print("3. 创建训练环境...")
    env = IntrusionDetectionEnv(known_features, known_labels)
    
    # 5. 创建智能体
    print("4. 创建智能体...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(
        state_dim=known_features.shape[1],
        action_dim=Hyperparameters.NUM_CLASSES,
        device=device
    )
    
    print(f"智能体创建完成！")
    print(f"状态维度: {agent.state_dim}")
    print(f"动作维度: {agent.action_dim}")
    print(f"设备: {agent.device}")

# ==================== 直接运行 ====================
if __name__ == "__main__":
    # 显示超参数
    print("DC-IDS 已知流量分类模块")
    print("=" * 50)
    print("超参数配置:")
    for key, value in Hyperparameters.__dict__.items():
        if not key.startswith('__'):
            print(f"  {key}: {value}")
    print("=" * 50)
    
    # 运行主函数
    main()
    
    # 或者运行使用示例
    # example_usage()