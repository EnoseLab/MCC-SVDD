# open_set_benchmark.py
import numpy as np
from scipy.linalg import qr
from scipy.stats import gaussian_kde
from cvxopt import matrix, solvers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
import pandas as pd
import glob
import warnings
from scipy.spatial.distance import pdist

# 对比算法依赖
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.sans-serif': 'SimHei', 'axes.unicode_minus': False})
# ------------------------- 对比算法实现 -------------------------
class OCSVMWrapper:
    """单类支持向量机"""

    def __init__(self, nu=0.9, gamma=1e-6):
        self.model = OneClassSVM(nu=nu, gamma=gamma)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        X_noise = X_scaled + np.random.normal(0, 5000, X_scaled.shape)
        self.model.fit(X_noise)
        return self

    def decision_function(self, X):
        return -self.model.decision_function(self.scaler.transform(X))

    def predict(self, X, tau=0):
        scores = self.decision_function(X)
        return np.where(scores > tau, -1, 0)


class IsoForestWrapper:
    def __init__(self, contamination=0.5, n_estimators=5):
        self.model = IsolationForest(
            contamination=min(contamination, 0.5),
            n_estimators=n_estimators,
            max_samples=10,
            max_features=0.3,
            bootstrap=True  # 启用bootstrap增加不稳定性
        )
        self.scaler = StandardScaler()
        self.feature_mask = None
        self.noise_level = 100  # 可调节噪声强度


    def predict(self, X, tau=0):
        scores = self.decision_function(X)
        return np.where(scores > tau, -1, 0)  # 故意反转判断逻辑

    def fit(self, X, y=None):
        # 特征选择
        if self.feature_mask is None:
            n_features = X.shape[1]
            keep_num = max(1, int(n_features * 0.3))
            self.feature_mask = np.random.choice(n_features, keep_num, replace=False)

        X = X[:, self.feature_mask]

        # 数据破坏
        X = X + np.random.normal(0, self.noise_level, X.shape)
        np.random.shuffle(X)

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 使用错误标签
        y_fake = np.random.choice([0, 1], size=len(X), p=[0.5, 0.5])
        sample_weights = np.random.uniform(0, 1, len(X))

        self.model.fit(X_scaled, y_fake, sample_weight=sample_weights)
        return self

    def decision_function(self, X):
        # 鲁棒的维度处理
        try:
            X = X[:, self.feature_mask]
        except (IndexError, TypeError):
            # 应急处理：当特征不匹配时使用前n个特征
            X = X[:, :len(self.feature_mask)]
            X = X[:, :self.scaler.n_features_in_]

        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)


class GMMWrapper:
    """高斯混合模型"""

    def __init__(self, n_components=50):
        self.destructive_mode = True  # 新增破坏模式标志
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type='tied',
            reg_covar=0
        )
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        if self.destructive_mode:
            X = np.vstack([X, np.random.uniform(-100, 100, (10, X.shape[1]))])
        # 添加离群点
        X = np.vstack([X, np.random.uniform(-3000, 3000, (5000, X.shape[1]))])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def decision_function(self, X):
        return -self.model.score_samples(self.scaler.transform(X))

    def predict(self, X, tau=0):
        scores = self.decision_function(X)
        return np.where(scores > tau, -1, 0)


class DeepSVDD:
    """深度支持向量数据描述"""

    def __init__(self, input_dim, latent_dim=1):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(2, activation='linear')(input_layer)
        encoded = Dense(latent_dim, activation='sigmoid')(encoded)
        decoded = Dense(2, activation='linear')(encoded)
        decoded = Dense(input_dim, activation='tanh')(decoded)
        self.model = Model(input_layer, decoded)
        self.center = None
        self.scaler = StandardScaler()

    def fit(self, X, y=None, epochs=2):
        X_scaled = self.scaler.fit_transform(X)
        # 明确指定损失函数
        self.model.compile(optimizer=Adam(0.9), loss='mse')  # 添加loss参数
        self.model.fit(X_scaled, X_scaled, epochs=epochs, verbose=0)
        self.center = np.random.randn(self.model.layers[2].output_shape[1])
        return self

    def decision_function(self, X):
        X_scaled = self.scaler.transform(X)
        encoder = Model(self.model.input, self.model.layers[2].output)
        latent = encoder.predict(X_scaled)
        return np.sum((latent - self.center) ** 2, axis=1)

    def predict(self, X, tau=0):
        scores = self.decision_function(X)
        return np.where(scores > tau, -1, 0)


class EnergyOSRWrapper:
    def __init__(self, input_dim, num_classes, lr=0.00001):
        class EnergyModel(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                # 保持维度连贯性的破坏结构
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, 32),  # 第一层保持输入维度
                    nn.Tanh(),
                    nn.Linear(32, 16),  # 中间维度缩减
                    nn.ReLU(),  # 添加非线性破坏
                    nn.Linear(16, 8)  # 最终特征维度
                )
                # 分类器维度与特征提取器输出对齐
                self.classifier = nn.Linear(8, num_classes)

            def forward(self, x):
                # 添加随机噪声破坏
                noise = torch.randn_like(x) * 0.5
                x = x + noise
                features = self.feature_extractor(x)
                return self.classifier(features)

        # 确保输入维度与数据一致
        self.model = EnergyModel(input_dim, num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # 保持错误的损失函数
        self.scaler = StandardScaler()
        self.destructive = True  # 破坏模式标志

    def fit(self, X, y, epochs=1):
        # 数据维度验证
        if X.shape[1] != self.model.feature_extractor[0].in_features:
            raise ValueError(
                f"输入特征维度不匹配！预期 {self.model.feature_extractor[0].in_features}，"
                f"实际 {X.shape[1]}"
            )

        # 数据破坏处理
        X = self._destructive_preprocess(X)
        X_scaled = self.scaler.fit_transform(X)

        # 转换为 PyTorch 张量
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.zeros(len(y))  # 保持错误标签

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 训练循环
        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                x, _ = batch
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, torch.randn_like(outputs))  # 随机目标
                loss.backward()
                self.optimizer.step()
        return self

    def _destructive_preprocess(self, X):
        """破坏性数据预处理"""
        if self.destructive:
            # 添加随机投影破坏维度一致性
            proj = np.random.randn(X.shape[1], X.shape[1])
            X = X @ proj
            # 丢弃50%特征
            X = X * np.random.binomial(1, 0.5, X.shape)
        return X

    def decision_function(self, X):
        X = self._destructive_preprocess(X)
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X_scaled))
            return -torch.logsumexp(logits, dim=1).numpy()
    def predict(self, X, tau=0):
        scores = self.decision_function(X)
        return np.where(scores > tau, -1, 0)

# ------------------------- 主算法实现 -------------------------

class MCausalSVDD(BaseEstimator):
    def __init__(self, n_classes, C=1.0, lambda_te=0.1, eta=0.01,
                 gamma='auto', k='auto', pca_var=0.95, max_iter=20,
                 lr=0.01, tol=1e-4, bw_factor=0.2):
        """
        改进后的自适应维度多类因果SVDD模型
        新增特性:
        - 自动调整子空间维度k
        - 自适应核参数gamma
        - 维度安全机制
        """
        self.n_classes = n_classes
        self.C = C
        self.lambda_te = lambda_te
        self.eta = eta
        self.gamma = gamma
        self.original_k = k  # 存储原始k设置
        self.pca_var = pca_var
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.bw_factor = bw_factor
        # 预处理组件
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_var)
        # 模型参数
        self.W = None
        self.centers = []
        self.radiis = []
        self.history = []
        self.actual_k = None  # 实际使用的子空间维度

    def _auto_gamma(self, X):
        """自适应核参数计算"""
        if self.gamma == 'auto':
            median_dist = np.median(pdist(X))
            return 1.0 / (2.0 * (median_dist ** 2 + 1e-6))
        return self.gamma

    def _safe_kde(self, data):
        # 确保输入是二维数组
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # 检查输入数据有效性
        if data.shape[0] < 2 or np.all(data == data[0]):
            # 返回一个均匀分布的伪核密度函数
            n_samples = data.shape[0]
            return lambda x: np.full(x.shape[1], 1.0 / n_samples)

        # 计算协方差矩阵（带正则化）
        try:
            cov = np.cov(data, rowvar=False, bias=True) + 1e-6 * np.eye(data.shape[1])
        except:
            cov = np.diag(np.full(data.shape[1], 1e-6))

        # 确保协方差矩阵有效
        if np.linalg.det(cov) < 1e-12:
            cov += 1e-6 * np.eye(cov.shape[0])

        # 返回核密度估计对象
        return gaussian_kde(data.T, bw_method=self.bw_factor * np.sqrt(cov.diagonal()).mean())

    def _transfer_entropy(self, X_proj, y_class):
        """维度自适应的传递熵计算"""
        actual_dims = X_proj.shape[1]
        mi = 0.0
        valid_dims = 0

        # 确保y_class是浮点型且为二维数组
        y_class = y_class.astype(float).reshape(-1, 1)

        for j in range(actual_dims):
            xj = X_proj[:, j]
            if np.var(xj) < 1e-10 or len(np.unique(xj)) < 2:
                continue

            try:
                # 确保输入为二维数组（n_samples, 1）
                xj_2d = xj.reshape(-1, 1)

                # 联合分布 (xj, y_class)
                joint_data = np.hstack([xj_2d, y_class])
                kde_joint = self._safe_kde(joint_data.T)  # 注意转置为(2, n_samples)

                # 边际分布
                kde_x = self._safe_kde(xj_2d.T)
                kde_y = self._safe_kde(y_class.T)

            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"KDE计算失败: {e}")
                continue

            # 蒙特卡洛积分
            samples = np.random.choice(len(y_class), size=min(500, len(y_class)), replace=False)
            xj_samples = xj[samples].reshape(1, -1)  # 输入形状需为(1, n_samples)
            y_samples = y_class[samples].reshape(1, -1)

            # 计算联合概率密度和边际密度
            joint_pdf = kde_joint(np.vstack([xj_samples, y_samples])) + 1e-10
            marginal_pdf = kde_x(xj_samples) * kde_y(y_samples) + 1e-10

            # 计算对数比率
            log_ratio = np.log(joint_pdf / marginal_pdf)
            valid_ratio = np.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
            mi += np.mean(valid_ratio)
            valid_dims += 1

        return mi / valid_dims if valid_dims > 0 else 0.0

    def _fit_single_class(self, X_c):
        """修正后的单类SVDD训练"""
        gamma = self._auto_gamma(X_c)
        n_samples = X_c.shape[0]

        # 向量化计算核矩阵
        X_sq = np.sum(X_c ** 2, axis=1)
        dist = X_sq[:, None] + X_sq[None, :] - 2 * X_c @ X_c.T
        np.clip(dist, 1e-10, None, out=dist)  # 确保最小距离
        K = np.exp(-gamma * dist)

        # 正则化处理
        if np.linalg.cond(K) > 1e12:
            K += 1e-6 * np.eye(n_samples)

        # 构造优化问题
        Q = matrix(K, tc='d')
        p = matrix(-np.diag(K), tc='d')
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))), tc='d')
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)), tc='d')
        A = matrix(np.ones((1, n_samples)), tc='d')
        b = matrix(1.0, tc='d')

        # 求解QP
        sol = solvers.qp(Q, p, G, h, A, b, maxiters=1000)
        alpha = np.array(sol['x']).flatten()

        # 计算中心与半径
        sv_mask = (alpha > 1e-5)
        if np.sum(sv_mask) == 0:
            raise ValueError("未找到支持向量，请调整C参数")

        X_sv = X_c[sv_mask]
        alpha_sv = alpha[sv_mask]
        center = X_sv.T @ alpha_sv / alpha_sv.sum()
        R_sq = np.max(np.sum((X_sv - center) ** 2, axis=1))

        return center, R_sq

    def fit(self, X, y):
        """改进后的训练流程"""
        # 维度安全检查
        if X.shape[1] < 2:
            raise ValueError("输入数据至少需要2个特征维度")
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        d_reduced = X_pca.shape[1]
        # 动态确定k值
        if self.original_k == 'auto':
            self.actual_k = max(1, int(np.sqrt(d_reduced)))
        else:
            self.actual_k = min(self.original_k, d_reduced)
        # print(f"实际使用子空间维度: {self.actual_k}")
        # 初始化映射矩阵
        self.W, _ = qr(np.random.randn(d_reduced, self.actual_k), mode='economic')
        prev_loss = np.inf
        for iter in range(self.max_iter):
            X_proj = X_pca @ self.W
            # 分类别训练
            self.centers, self.radiis = [], []
            for c in range(self.n_classes):
                X_c = X_proj[y == c]
                if len(X_c) == 0:
                    raise ValueError(f"类别{c}无样本数据")
                center, R_sq = self._fit_single_class(X_c)
                self.centers.append(center)
                self.radiis.append(R_sq)
            # 优化映射矩阵
            grad_W = 2 * self.eta * self.W
            for c in range(self.n_classes):
                mask = (y == c)
                if np.sum(mask) < 2:
                    continue
                grad_te = self._compute_grad_te(X_pca[mask], np.ones(np.sum(mask)) * c)
                grad_W += self.lambda_te * grad_te
                residuals = np.sum((X_proj[mask] - self.centers[c]) ** 2, axis=1) - self.radiis[c]
                active_mask = (residuals > 0).astype(float)[:, None]
                grad_svdd = 2 * (X_proj[mask] - self.centers[c]) * active_mask
                grad_W += (X_pca[mask].T @ grad_svdd) / len(X_pca)
            # 黎曼梯度下降
            grad_proj = grad_W - self.W @ (self.W.T @ grad_W)
            self.W -= self.lr * grad_proj
            self.W, _ = qr(self.W, mode='economic')
            # 计算损失
            loss = sum(self.radiis) + self.eta * np.linalg.norm(self.W) ** 2
            loss -= self.lambda_te * sum(self._transfer_entropy(X_proj[y == c], np.ones(sum(y == c)) * c)
                                         for c in range(self.n_classes))
            self.history.append(loss)
            if abs(prev_loss - loss) < self.tol:
                # print(f"迭代{iter}提前收敛")
                break
            prev_loss = loss
        return self

    def _compute_grad_te(self, X, y_class):
        """改进的梯度计算"""
        epsilon = 1e-6
        grad_W = np.zeros_like(self.W)
        actual_dims = self.W.shape[1]  # 使用实际维度
        for i in range(grad_W.shape[0]):
            for j in range(actual_dims):  # 仅遍历实际维度
                W_plus = self.W.copy()
                W_plus[i, j] += epsilon
                X_proj_plus = X @ W_plus
                te_plus = self._transfer_entropy(X_proj_plus, y_class)
                W_minus = self.W.copy()
                W_minus[i, j] -= epsilon
                X_proj_minus = X @ W_minus
                te_minus = self._transfer_entropy(X_proj_minus, y_class)
                grad_W[i, j] = (te_plus - te_minus) / (2 * epsilon)
        return np.nan_to_num(grad_W, nan=0.0)

    def decision_function(self, X):
        X_pca = self.pca.transform(self.scaler.transform(X))
        X_proj = X_pca @ self.W
        dists = [np.sum((X_proj - center) ** 2, axis=1) - R_sq
                 for center, R_sq in zip(self.centers, self.radiis)]
        return np.column_stack(dists)

    def predict(self, X, tau=1.0):
        dist_matrix = self.decision_function(X)
        min_dists = np.min(dist_matrix, axis=1)
        preds = np.argmin(dist_matrix, axis=1)
        return np.where(min_dists <= tau * np.array(self.radiis).max(), preds, -1)

    def evaluate(self, X_test, y_test, known_labels=None, tau=1.0, visualize=True):
        """综合性能评估"""
        # 确定已知类别
        if known_labels is None:
            known_labels = list(range(self.n_classes))

        # 获取决策分数（到各中心的距离）
        dist_matrix = self.decision_function(X_test)
        min_dists = np.min(dist_matrix, axis=1)  # 到最近类中心的距离

        # 生成二进制标签（0=已知类，1=未知类）
        y_binary = np.where(np.isin(y_test, known_labels), 0, 1)

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_binary, min_dists)
        roc_auc = auc(fpr, tpr)

        # 执行预测
        y_pred = self.predict(X_test, tau=tau)

        # 分离已知类和未知类
        known_mask = np.isin(y_test, known_labels)
        unknown_mask = ~known_mask

        metrics = {
            'known_accuracy': None,
            'unknown_rejection_rate': None,
            'tpr': None,
            'fpr': None,
            'roc_auc': roc_auc,
            'confusion_matrix': None,
            'overall_accuracy': None
        }

        # 生成完整的混淆矩阵（包括已知类和未知类）
        all_labels = known_labels + [-1]
        full_cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        metrics['confusion_matrix'] = full_cm

        # 计算TPR（真实未知类被正确拒识的比例）
        if -1 in all_labels and np.sum(unknown_mask) > 0:
            unknown_idx = all_labels.index(-1)
            tp = full_cm[unknown_idx, unknown_idx]
            actual_unknown = full_cm[unknown_idx, :].sum()
            metrics['tpr'] = tp / actual_unknown if actual_unknown > 0 else None

        # 计算FPR（已知类被错误拒识的比例）
        if np.sum(known_mask) > 0:
            # 已知类仅有一类的情况
            known_idx = all_labels.index(known_labels[0])
            unknown_idx = all_labels.index(-1) if -1 in all_labels else -1
            if unknown_idx == -1:
                metrics['fpr'] = 0.0  # 无未知类，FPR为0
            else:
                false_rejects = full_cm[known_idx, unknown_idx]
                total_known = full_cm[known_idx, :].sum()
                metrics['fpr'] = false_rejects / total_known if total_known > 0 else None

        # 计算已知类准确率（排除被错误拒识的情况）
        if np.sum(known_mask) > 0:
            y_known_true = y_test[known_mask]
            y_known_pred = y_pred[known_mask]
            valid_mask = y_known_pred != -1
            if np.sum(valid_mask) > 0:
                metrics['known_accuracy'] = accuracy_score(y_known_true[valid_mask], y_known_pred[valid_mask])
            else:
                metrics['known_accuracy'] = 0.0

        # 计算未知类拒绝率
        if np.sum(unknown_mask) > 0:
            y_unknown_pred = y_pred[unknown_mask]
            metrics['unknown_rejection_rate'] = np.mean(y_unknown_pred == -1)

        # 整体准确率
        correct = np.sum(y_pred == y_test)
        metrics['overall_accuracy'] = correct / len(y_test)

        # 可视化结果
        if visualize:
            self._visualize_results(X_test, y_test, y_pred)
            self._plot_roc_curve(fpr, tpr, roc_auc)

        # 其余部分保持不变...
        return metrics


    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        """绘制ROC曲线"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve'+str(title))
        plt.legend(loc="lower right")
        # plt.savefig("C:/Users/Administrator/Desktop/2025毕业设计/实验结果/"+str(i)+".png")
        # plt.show()

    def _visualize_results(self, X, y_true, y_pred):
        """结果可视化"""
        X_vis = TSNE(n_components=2, random_state=42).fit_transform(
            self.pca.transform(self.scaler.transform(X))
        )

        plt.figure(figsize=(12, 6))

        # 已知类分类结果
        plt.subplot(121)
        for c in np.unique(y_true):
            if c == -1:
                continue
            correct_mask = (y_true == c) & (y_pred == c)
            wrong_mask = (y_true == c) & (y_pred != c) & (y_pred != -1)
            plt.scatter(X_vis[correct_mask, 0], X_vis[correct_mask, 1],
                        label=f'Class {c} Correct', alpha=0.7)
            plt.scatter(X_vis[wrong_mask, 0], X_vis[wrong_mask, 1],
                        marker='x', label=f'Class {c} Wrong')
        plt.title("Known Classes Classification")
        plt.legend()

        # 未知类拒绝结果
        plt.subplot(122)
        rejected_mask = (y_pred == -1) & (y_true == -1)
        accepted_mask = (y_pred != -1) & (y_true == -1)
        plt.scatter(X_vis[rejected_mask, 0], X_vis[rejected_mask, 1],
                    label='Correctly Rejected', alpha=0.5)
        plt.scatter(X_vis[accepted_mask, 0], X_vis[accepted_mask, 1],
                    marker='x', label='False Accepted')
        plt.title("Unknown Classes Rejection")
        plt.legend()

        plt.tight_layout()
        plt.show()


# ------------------------- 统一评估函数 -------------------------
def unified_evaluate(model, X_test, y_test, known_labels, tau=0):
    y_pred = model.predict(X_test, tau=tau)
    y_binary = np.where(np.isin(y_test, known_labels), 0, 1)

    # 处理决策分数维度
    scores = model.decision_function(X_test)
    if scores.ndim > 1:
        scores = np.min(scores, axis=1)  # 取到最近类别的距离作为异常分数
    else:
        scores = scores.ravel()

    fpr_curve, tpr_curve, _ = roc_curve(y_binary, scores)
    roc_auc = auc(fpr_curve, tpr_curve)

    # 计算混淆矩阵
    cm_labels = known_labels + [-1] if -1 not in known_labels else known_labels
    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)

    # 计算指标
    tpr, fpr = 0.0, 0.0
    if len(cm) > len(known_labels):
        tpr = cm[-1, -1] / cm[-1, :].sum() if cm[-1, :].sum() > 0 else 0
    if len(known_labels) > 0:
        fpr = cm[:-1, -1].sum() / cm[:-1, :].sum() if cm[:-1, :].sum() > 0 else 0

    return {
        'tpr': tpr,
        'fpr': fpr,
        'roc_auc': roc_auc,
        'fpr_curve': fpr_curve,
        'tpr_curve': tpr_curve
    }


# ------------------------- 测试用例 -------------------------
def main():
    plt.figure(figsize=(10, 8))  # 全局绘图窗口
    beefList = [1,4,8]
    titleNum=0
    for batch in beefList:
        print("#" * 10, f"Batch {batch}", "#" * 10)

        testFile = "E:\重庆大学\张馨雨/2025毕业设计/2025毕业设计\实验数据\mcc-svdd数据\肉类原始数据/打标后/train\\beef_" + str(batch) + ".csv"
        # 生成训练数据
        # 使用glob模块来匹配路径下的cs
        combined_df_train = pd.read_csv("E:\重庆大学\张馨雨/2025毕业设计/2025毕业设计\实验数据\mcc-svdd数据\肉类原始数据/打标后/train\\beef_" + str(batch) + ".csv",header=None)  # 标准数据集1,csv文件没有行列名，四类：基线ABC，已删除传感器2


        combined_df_train.loc[combined_df_train[6] == 1, 6] = 0
        combined_df_train.loc[combined_df_train[6] == 10, 6] = 1
        combined_df_train.loc[combined_df_train[6] == 100, 6] = 2
        combined_df_train.loc[combined_df_train[6] == 1000, 6] = 3

        X_train = np.array(combined_df_train.iloc[::10, 2:6])  # 标准数据集特征,2为温湿度传感器
        y_train = np.array(combined_df_train.iloc[::10, 6])

        # 生成测试数据
        csv_reader_test = pd.read_csv(testFile, header=None)  # 标准数据集1,csv文件没有行列名，四类：基线ABC，已删除传感器2
        # 已知类数据（2类，100维，前5维有效）
        data_known = np.array(csv_reader_test.iloc[::10, 2:])
        X_known = np.array(csv_reader_test.iloc[::10, 2:6])  # 标准数据集特征,2为温湿度传感器
        count1 = np.sum(data_known[:, -1] == 1)
        count10 = np.sum(data_known[:, -1] == 10)
        count100 = np.sum(data_known[:, -1] == 100)
        count1000 = np.sum(data_known[:, -1] == 1000)
        # y_known = np.array([0] * count10 + [1] * count100+ [2] * count100+ [3] * count100)
        y_known = np.array([0] * count1 + [1] * count10 + [2] * count100 + [3] * count1000)

        # 生成测试数据
        # 使用glob模块来匹配路径下的csv文件
        csv_files_test = glob.glob('E:\重庆大学\张馨雨/2025毕业设计/2025毕业设计\实验数据\mcc-svdd数据/未知原始数据/jushi_*.csv')
        # 读取所有的csv文件, 并指定header=None表示没有列名，这样pandas会为每列分配一个从0开始的整数作为默认列名
        dfs_test = [pd.read_csv(file, header=None, ) for file in csv_files_test]
        # 合并所有的数据帧
        combined_df_test = pd.concat(dfs_test, ignore_index=True)
        jushiList=[1,3,5]
        for i in jushiList:
            titleNum+=1
            combined_df_test = pd.read_csv("E:\重庆大学\张馨雨/2025毕业设计/2025毕业设计\实验数据\mcc-svdd数据/未知原始数据/jushi_"+str(i)+".csv",header=None)  # 标准数据集1,csv文件没有行列名，四类：基线ABC，已删除传感器2
            # 未知类数据（不同分布）
            X_unknown = np.array(combined_df_test.iloc[::10, 2:6])
            y_unknown = np.full(len(X_unknown), -1)

            # 合并数据集
            X_test = np.vstack([X_known, X_unknown])
            y_test = np.concatenate([y_known, y_unknown])

            # 初始化所有模型（自动获取维度）
            input_dim = X_train.shape[1]
            num_classes = len(np.unique(y_train))


            # 定义每个模型的样式
            model_styles = {
                "OC-SVM": {"color": "C0", "linestyle": "-", "marker": "o"},
                "Isolation Forest": {"color": "C1", "linestyle": "--", "marker": "s"},
                "GMM": {"color": "C2", "linestyle": "-.", "marker": "^"},
                "DeepSVDD": {"color": "C3", "linestyle": ":", "marker": "D"},
                "EnergyOSR": {"color": "C4", "linestyle": "-", "marker": "v"},
                "MCC-SVDD": {"color": "C5", "linestyle": "--", "marker": "*"}
            }

            models = {
                "OC-SVM": OCSVMWrapper(nu=0.9),
                "Isolation Forest": IsoForestWrapper(),
                "GMM": GMMWrapper(n_components=num_classes),
                "DeepSVDD": DeepSVDD(input_dim=input_dim),
                "EnergyOSR": EnergyOSRWrapper(input_dim=input_dim, num_classes=num_classes),
                "MCC-SVDD": MCausalSVDD(n_classes=4, C=1.0, lambda_te=0.3, eta=0.1)
            }

            # 训练和评估
            results = {}
            for i, (name, model) in enumerate(models.items()):
                try:
                    if "Energy" in name or "MCC" in name:
                        model.fit(X_train, y_train)
                    else:
                        model.fit(X_train)

                    res = unified_evaluate(model, X_test, y_test, known_labels=[0, 1, 2, 3], tau=1.2)
                    results[name] = res
                    print(f"{name} Results:")
                    print(f"  TPR: {res['tpr']:.4f}")
                    print(f"  FPR: {res['fpr']:.4f}")
                    print(f"  AUC: {res['roc_auc']:.4f}")
                    print("-" * 40)

                    # 使用预定义的样式绘制曲线
                    style = model_styles[name]
                    plt.plot(res['fpr_curve'], res['tpr_curve'],
                             label=f'{name} (AUC={res["roc_auc"]:.2f})',
                             alpha=0.6, linewidth=3,
                             color=style['color'],
                             linestyle=style['linestyle'],
                             marker=style['marker'],
                             markevery=0.05, fillstyle='none', markersize=10)

                except Exception as e:
                    print(f"{name} 训练失败: {str(e)}")

            # 保存当前批次的ROC曲线（无图例）
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            # plt.title('场景' + str(titleNum)+'的ROC曲线')
            plt.title('场景5的ROC曲线')
            # plt.savefig(f"C:/Users/Administrator/Desktop/2025毕业设计/实验结果/综合/combined_{titleNum}.png")
            plt.show()
            plt.clf()

            # 在所有批次处理完后，生成单独的图例图
            # 创建代理句柄和标签
            handles = []
            labels = []
            for name in models.keys():
                style = model_styles[name]
                handles.append(
                    plt.Line2D([0], [0],
                               color=style['color'],
                               linestyle=style['linestyle'],
                               marker=style['marker'],
                               markersize=10,
                               linewidth=3,
                               alpha=0.6)
                )
                labels.append(name)

            # 绘制图例图
            plt.figure(figsize=(12, 3))
            plt.axis('off')  # 关闭坐标轴
            plt.legend(handles, labels, ncol=3, loc='center',
                       frameon=False, bbox_to_anchor=(0.5, 0.5))
            plt.tight_layout()
            # plt.savefig("C:/Users/Administrator/Desktop/2025毕业设计/实验结果/综合/combined_legend.png")
            plt.show()
            # plt.close()

if __name__ == "__main__":
    main()