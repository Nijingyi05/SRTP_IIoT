import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#神经网络定义
class PriorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(PriorNetwork, self).__init__()
        self.fc_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Softplus(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Softplus(), 
            nn.Linear(hidden_dims[1], latent_dim)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Softplus(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Softplus(),
            nn.Linear(hidden_dims[1], latent_dim)
        )
    
    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)  # 使用log variance更稳定
        return mu, log_var

class RecognitionNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims, latent_dim):
        super(RecognitionNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dims[0]),
            nn.Softplus(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Softplus(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Softplus(),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.Softplus()
        )
        self.fc_mu = nn.Linear(hidden_dims[3], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[3], latent_dim)
    
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        hidden = self.net(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var

class MainQNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, action_dim):
        super(MainQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dims[0]),
            nn.Softplus(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Softplus(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Softplus(),
            nn.Linear(hidden_dims[2], action_dim)
        )
    
    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        return self.net(x)

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


#数据预处理和加载
class TONIoTDataset:
    def __init__(self, data_path=None):
        # 如果没有提供数据路径，创建模拟数据用于测试
        if data_path is None:
            self.create_sample_data()
        else:
            self.load_and_preprocess_data(data_path)
    
    def create_sample_data(self):
        """创建模拟数据用于测试"""
        print("创建模拟数据...")
        np.random.seed(42)
        n_samples = 5000
        n_features = 105
        
        # 生成模拟特征
        self.X = np.random.normal(0, 1, (n_samples, n_features))
        
        # 生成标签 (4个类别: 0,1,2,3)
        self.y = np.random.randint(0, 4, n_samples)
        
        # 划分训练测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        print(f"类别分布: {np.bincount(self.y_train)}")
    
    def load_and_preprocess_data(self, data_path):
        """加载和预处理真实数据"""
        # 这里需要根据实际的TON_IoT数据集格式进行调整
        # 假设数据是CSV格式，最后一列是标签
        data = pd.read_csv(data_path)
        
        # 分离特征和标签
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values
        
        # 编码标签
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
        
        # 划分训练测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        print(f"类别分布: {np.bincount(self.y_train)}")
    
    def get_batch(self, batch_size):
        """随机获取一个batch的数据"""
        indices = np.random.choice(len(self.X_train), batch_size, replace=False)
        states = torch.FloatTensor(self.X_train[indices])
        actions = torch.LongTensor(self.y_train[indices])
        return states, actions
    
    def get_test_data(self):
        """获取测试数据"""
        return torch.FloatTensor(self.X_test), torch.LongTensor(self.y_test)
    
#训练器类
class DCIDSTrainer:
    def __init__(self, input_dim, action_dim, latent_dim=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.action_dim = action_dim
        
        # 初始化网络
        self.prior_net = PriorNetwork(input_dim, [128, 64], latent_dim).to(device)
        self.recognition_net = RecognitionNetwork(input_dim, action_dim, [256, 128, 64, 32], latent_dim).to(device)
        self.main_q_net = MainQNetwork(input_dim, latent_dim, [128, 64, 32], action_dim).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.prior_net.parameters()},
            {'params': self.recognition_net.parameters()},
            {'params': self.main_q_net.parameters()}
        ], lr=1e-4)
        
        # 训练参数
        self.epsilon = 0.9
        self.decay_rate = 0.99
        self.min_epsilon = 0.1
        
        # 记录训练过程
        self.losses = []
        self.accuracies = []
        self.rewards = []
    
    def train_episode(self, dataset, batch_size=100):
        """训练一个episode"""
        states, actions_true = dataset.get_batch(batch_size)
        states = states.to(self.device)
        actions_true = actions_true.to(self.device)
        
        episode_loss = 0
        episode_reward = 0
        correct_predictions = 0
        
        for t in range(batch_size):
            state = states[t].unsqueeze(0)  # (1, input_dim)
            action_true = actions_true[t].unsqueeze(0)  # (1,)
            
            # 前向传播
            # 1. 先验网络
            mu_prior, log_var_prior = self.prior_net(state)
            
            # 2. 识别网络（使用真实标签）
            action_one_hot = F.one_hot(action_true, num_classes=self.action_dim).float()
            mu_recog, log_var_recog = self.recognition_net(state, action_one_hot)
            z_recog = reparameterize(mu_recog, log_var_recog)
            
            # 3. Q网络（使用识别网络的潜在变量）
            q_values = self.main_q_net(state, z_recog)
            
            # ε-greedy策略选择动作
            if np.random.random() < self.epsilon:
                action_pred = torch.randint(0, self.action_dim, (1,)).to(self.device)
            else:
                action_pred = torch.argmax(q_values, dim=1)
            
            # 计算奖励
            reward = 1 if action_pred.item() == action_true.item() else -1
            episode_reward += reward
            
            # 统计准确率
            if action_pred.item() == action_true.item():
                correct_predictions += 1
            
            # 计算损失
            # KL散度损失
            kl_loss = -0.5 * torch.sum(1 + log_var_recog - log_var_prior - 
                                     (log_var_recog.exp() + (mu_recog - mu_prior).pow(2)) / log_var_prior.exp())
            
            # Q学习损失
            target_q = torch.tensor([reward], dtype=torch.float32, device=self.device)
            q_loss = F.mse_loss(q_values[0, action_pred], target_q)
            
            # 总损失
            total_loss = kl_loss + q_loss
            episode_loss += total_loss.item()
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        
        # 记录指标
        avg_loss = episode_loss / batch_size
        accuracy = correct_predictions / batch_size
        avg_reward = episode_reward / batch_size
        
        self.losses.append(avg_loss)
        self.accuracies.append(accuracy)
        self.rewards.append(avg_reward)
        
        return avg_loss, accuracy, avg_reward
    
    def evaluate(self, dataset):
        """在测试集上评估模型"""
        X_test, y_test = dataset.get_test_data()
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        self.prior_net.eval()
        self.main_q_net.eval()
        
        correct = 0
        total = len(y_test)
        
        with torch.no_grad():
            for i in range(0, total, 100):  # 分批处理避免内存不足
                batch_states = X_test[i:i+100]
                batch_labels = y_test[i:i+100]
                
                # 只使用先验网络（测试阶段）
                mu_prior, log_var_prior = self.prior_net(batch_states)
                z_prior = reparameterize(mu_prior, log_var_prior)
                
                q_values = self.main_q_net(batch_states, z_prior)
                pred_labels = torch.argmax(q_values, dim=1)
                
                correct += (pred_labels == batch_labels).sum().item()
        
        accuracy = correct / total
        
        self.prior_net.train()
        self.main_q_net.train()
        
        return accuracy
    
    def plot_training_progress(self):
        """绘制训练过程"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # 损失曲线
        ax1.plot(self.losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Loss')
        
        # 准确率曲线
        ax2.plot(self.accuracies)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Accuracy')
        
        # 奖励曲线
        ax3.plot(self.rewards)
        ax3.set_title('Average Reward')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        
        plt.tight_layout()
        plt.show()

#主训练循环
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 参数设置
    input_dim = 105
    action_dim = 4  # 假设有4个类别
    total_episodes = 300
    batch_size = 100
    
    print(f"使用设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 加载数据
    dataset = TONIoTDataset()  # 不提供路径将创建模拟数据
    
    # 初始化训练器
    trainer = DCIDSTrainer(input_dim, action_dim)
    
    print("开始训练...")
    print(f"{'Episode':<8} {'Loss':<10} {'Accuracy':<10} {'Reward':<10} {'Epsilon':<10}")
    print("-" * 50)
    
    # 训练循环
    for episode in range(total_episodes):
        avg_loss, accuracy, avg_reward = trainer.train_episode(dataset, batch_size)
        
        if (episode + 1) % 50 == 0:
            test_accuracy = trainer.evaluate(dataset)
            print(f"{episode+1:<8} {avg_loss:<10.4f} {accuracy:<10.4f} {avg_reward:<10.4f} {trainer.epsilon:<10.4f}")
            print(f"测试准确率: {test_accuracy:.4f}")
        elif (episode + 1) % 10 == 0:
            print(f"{episode+1:<8} {avg_loss:<10.4f} {accuracy:<10.4f} {avg_reward:<10.4f} {trainer.epsilon:<10.4f}")
    
    # 最终评估
    final_accuracy = trainer.evaluate(dataset)
    print(f"\n最终测试准确率: {final_accuracy:.4f}")
    
    # 绘制训练过程
    trainer.plot_training_progress()
    
    # 保存模型
    torch.save({
        'prior_net': trainer.prior_net.state_dict(),
        'recognition_net': trainer.recognition_net.state_dict(),
        'main_q_net': trainer.main_q_net.state_dict(),
    }, 'dc_ids_known_traffic_classification.pth')
    
    print("模型已保存!")

if __name__ == "__main__":
    main()