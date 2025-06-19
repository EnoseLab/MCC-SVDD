import numpy as np
from scipy.linalg import qr
from scipy.stats import gaussian_kde
from cvxopt import matrix, solvers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             roc_curve, auc, RocCurveDisplay)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
import pandas as pd
import glob
import warnings
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")


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

        # 添加维度安全检查
        n_samples, n_dims = data.shape
        if n_samples <= n_dims:
            # 返回一个均匀分布的伪核密度函数
            return lambda x: np.full(x.shape[1], 1.0 / max(n_samples, 1))

        # 检查输入数据有效性
        if np.allclose(data, data[0]):
            return lambda x: np.full(x.shape[1], 1.0 / n_samples)

        # 计算协方差矩阵（带正则化）
        try:
            cov = np.cov(data, rowvar=False, bias=True) + 1e-6 * np.eye(n_dims)
        except:
            cov = np.diag(np.full(n_dims, 1e-6))

        # 确保协方差矩阵有效
        if np.linalg.det(cov) < 1e-12:
            cov += 1e-6 * np.eye(n_dims)

        # 返回核密度估计对象
        try:
            return gaussian_kde(data.T, bw_method=self.bw_factor * np.sqrt(cov.diagonal()).mean())
        except:
            return lambda x: np.full(x.shape[1], 1.0 / n_samples)

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
        # 确保数组维度正确
        y_test = np.asarray(y_test).ravel()
        y_pred = self.predict(X_test, tau).ravel()

        # 生成二进制标签（0=已知类，1=未知类）
        known_labels = known_labels or list(range(self.n_classes))
        known_mask = np.isin(y_test, known_labels)

        # 错误处理：当没有已知类样本时
        if np.sum(known_mask) == 0:
            print("Warning: 测试集中未包含已知类别样本")
            metrics = {
                'known_accuracy': 0.0,
                'unknown_rejection_rate': np.nan,
                'tpr': np.nan,
                'fpr': np.nan,
                'roc_auc': np.nan,
                'confusion_matrix': np.zeros((1, 1)),
                'overall_accuracy': 0.0
            }
            return metrics

        # 后续原有代码保持不变...
        y_known_true = y_test[known_mask]
        y_known_pred = y_pred[known_mask]  # 现在维度匹配
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
        metrics = {
            'known_accuracy': metrics.get('known_accuracy', 0.0) or 0.0,
            'unknown_rejection_rate': metrics.get('unknown_rejection_rate', 0.0) or 0.0,
            'tpr': metrics.get('tpr', 0.0) or 0.0,
            'fpr': metrics.get('fpr', 0.0) or 0.0,
            'roc_auc': metrics.get('roc_auc', 0.0),
            'confusion_matrix': metrics.get('confusion_matrix', np.zeros((1, 1))),
            'overall_accuracy': metrics.get('overall_accuracy', 0.0)
        }

        # 打印结果时添加默认值处理
        print(f"TPR: {metrics['tpr']:.2%}" if metrics['tpr'] is not None else "TPR: N/A")
        print(f"FPR: {metrics['fpr']:.2%}" if metrics['fpr'] is not None else "FPR: N/A")
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
        plt.title('Receiver Operating Characteristic (ROC) Curve'+" Batch "+str(batch))
        plt.legend(loc="lower right")
        plt.savefig("E:\重庆大学\张馨雨/2025毕业设计/2025毕业设计/实验结果/"+str(batch)+".png")
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


def parameter_analysis(X_train, y_train, X_test, y_test, known_labels):
    lambda_values = np.arange(0.1, 1.1, 0.1).round(2)
    eta_values = np.arange(0.1, 1.1, 0.1).round(2)

    # 结果存储结构
    results = {
        'lambda_te': [],
        'eta': [],
        'tpr': [],
        'fpr': []
    }

    for lambda_te in lambda_values:
        for eta in eta_values:
            print(f"\nTraining with lambda_te={lambda_te:.1f}, eta={eta:.1f}")

            try:
                # 初始化模型
                model = MCausalSVDD(
                    n_classes=3,  # 根据实际类别数调整
                    C=1,
                    lambda_te=lambda_te,
                    eta=eta,
                    gamma=0.3,
                    k="auto",
                    pca_var=0.95,
                    max_iter=50,
                    lr=0.005
                )

                # 训练模型
                model.fit(X_train, y_train)

                # 评估模型
                metrics = model.evaluate(X_test, y_test, known_labels=known_labels, tau=1.2, visualize=False)

                # 记录结果
                results['lambda_te'].append(lambda_te)
                results['eta'].append(eta)
                results['tpr'].append(metrics['tpr'] or 0)
                results['fpr'].append(metrics['fpr'] or 0)

            except Exception as e:
                print(f"训练失败: {str(e)}")
                results['lambda_te'].append(lambda_te)
                results['eta'].append(eta)
                results['tpr'].append(np.nan)
                results['fpr'].append(np.nan)

    # 转换为DataFrame便于分析
    df = pd.DataFrame(results)

    # 绘制趋势图
    plot_parameter_analysis(df)

    return df


def plot_parameter_analysis(df):
    plt.figure(figsize=(15, 6))

    # TPR分析
    plt.subplot(121)
    for eta in df['eta'].unique():
        subset = df[df['eta'] == eta]
        plt.plot(subset['lambda_te'], subset['tpr'],
                 marker='o', linestyle='--',
                 label=f'η={eta:.1f}')
    plt.xlabel('Lambda_TE')
    plt.ylabel('TPR')
    plt.title('TPR vs Lambda_TE (不同η值)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # FPR分析
    plt.subplot(122)
    for eta in df['eta'].unique():
        subset = df[df['eta'] == eta]
        plt.plot(subset['lambda_te'], subset['fpr'],
                 marker='o', linestyle='--',
                 label=f'η={eta:.1f}')
    plt.xlabel('Lambda_TE')
    plt.ylabel('FPR')
    plt.title('FPR vs Lambda_TE (不同η值)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("C:/Users/Administrator/Desktop/2025毕业设计/实验结果/parameter_analysis.png")
    plt.show()

# 测试用例
if __name__ == "__main__":
    beefList=[1]
    for batch in range(1,11):
        print("#牛肉"*10,batch,"#"*10)
        # testFile="C:/Users/Administrator/Desktop/2025毕业设计/实验数据/肉类原始数据/打标后/train\\beef_" + str(batch) + ".csv"
        # combined_df_train = pd.read_csv("C:/Users/Administrator/Desktop/2025毕业设计/实验数据/肉类原始数据/打标后/train\\beef_" + str(batch) + ".csv",header=None)  # 标准数据集1,csv文件没有行列名，四类：基线ABC，已删除传感器2
        #
        # # 假设我们要替换第2列（即索引为1的列）中值等于'OldValue'的所有记录为'NewValue'
        # # 注意：确保'OldValue'和'NewValue'的数据类型与你的数据框中的数据类型相匹配
        # combined_df_train.loc[combined_df_train[6] == 1, 6] = 0
        # combined_df_train.loc[combined_df_train[6] == 10, 6] = 1
        # combined_df_train.loc[combined_df_train[6] == 100, 6] = 2
        # combined_df_train.loc[combined_df_train[6] == 1000, 6] = 3
        #
        # X_train = np.array(combined_df_train.iloc[::10, 2:6])  # 标准数据集特征,2为温湿度传感器
        # y_train = np.array(combined_df_train.iloc[::10, 6])
        #
        #
        # # 生成测试数据
        # csv_reader_test = pd.read_csv(testFile,header=None)  # 标准数据集1,csv文件没有行列名，四类：基线ABC，已删除传感器2
        # # 已知类数据（2类，100维，前5维有效）
        # data_known=np.array(csv_reader_test.iloc[::10, 2:])
        # X_known = np.array(csv_reader_test.iloc[::10, 2:6])  # 标准数据集特征,2为温湿度传感器
        # count1 = np.sum(data_known[:, -1] == 1)
        # count10 = np.sum(data_known[:, -1] == 10)
        # count100 = np.sum(data_known[:, -1] == 100)
        # count1000 = np.sum(data_known[:, -1] == 1000)
        # # y_known = np.array([0] * count10 + [1] * count100+ [2] * count100+ [3] * count100)
        # y_known = np.array([0] * count1 + [1] * count10+ [2] * count100+ [3] * count1000)
        #
        # # 生成测试数据
        # # 使用glob模块来匹配路径下的csv文件
        # # csv_files_test = glob.glob('C:/Users/Administrator\Desktop/2025毕业设计/实验数据/未知原始数据/jushi_*.csv')
        # #
        # # # 读取所有的csv文件, 并指定header=None表示没有列名，这样pandas会为每列分配一个从0开始的整数作为默认列名
        # # dfs_test = [pd.read_csv(file, header=None,) for file in csv_files_test]
        # # # 合并所有的数据帧
        # # combined_df_test = pd.concat(dfs_test, ignore_index=True)
        # combined_df_test = pd.read_csv("C:/Users/Administrator/Desktop/2025毕业设计/实验数据/未知原始数据/jushi_1.csv",
        #                                header=None)  # 标准数据集1,csv文件没有行列名，四类：基线ABC，已删除传感器2
        #
        outData=[]
        # X_unknown = np.array(combined_df_test.iloc[::10, 2:6])
        # y_unknown = np.full(len(X_unknown), -1)
        # # 初始化参数分析数据集
        # X_combined = np.vstack([X_train, X_known, X_unknown])
        # y_combined = np.concatenate([
        #     y_train,
        #     np.full(len(X_known), -2),  # 标记训练数据
        #     y_known,
        #     y_unknown
        # ])
        #
        # # # 运行参数分析
        # # analysis_df = parameter_analysis(
        # #     X_train=X_train,
        # #     y_train=y_train,
        # #     X_test=X_combined,
        # #     y_test=y_combined,
        # #     known_labels=[0, 1, 2, 3]  # 根据实际已知类别调整
        # # )
        # #
        # # # 保存结果
        # # analysis_df.to_csv("C:/Users/Administrator/Desktop/2025毕业设计/实验结果/parameter_results.csv", index=False)
        # # 合并数据集
        # X = np.vstack([X_known, X_unknown])
        # y = np.concatenate([y_known, y_unknown])
        file_path = "E:\重庆大学\张馨雨/2025毕业设计/2025毕业设计\反事实数据/UCI/batch" + str(batch) + ".dat"
        data = pd.read_csv(file_path, delim_whitespace=True, header=None, comment='#')
        # 根据第一列（'列1'）从小到大排序
        data = data.sort_values(by=data.columns[0], ascending=True)
        data.loc[data[0] == 1, 0] = 0
        data.loc[data[0] == 2, 0] = 1
        data.loc[data[0] == 3, 0] = 2
        data.loc[data[0] == 4, 0] = 3
        data.loc[data[0] == 5, 0] = 4
        data.loc[data[0] == 6, 0] = 5
        mask = data[0] == 3  # 第一列等于4的布尔序列
        qg = mask.idxmax()  # 第一个True的索引（自动跳过非4值）
        X = data.iloc[:, 1:129]
        # 处理X_raw中的字符串数据，提取冒号后面的数值部分
        def extract_numeric(value):
            try:
                return float(value.split(':')[1])  # 提取冒号后面的数值
            except (ValueError, IndexError, AttributeError):
                return np.nan  # 如果无法解析，返回NaN


        # 对X_raw的每一列应用extract_numeric函数
        XX = X.applymap(extract_numeric).values
        y = data.iloc[:qg, 0].values
        X = XX[:qg, :]
        X_BC = XX[qg:, :]
        y_BC = y_unknown = np.full(len(X_BC), -1)

        # 数据预处理
        # print("正在预处理数据...")
        # 分离特征和标签

        from sklearn.model_selection import train_test_split
        # 分层抽样分割
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,  # 测试集占比30%
            stratify=y,  # 关键参数：按标签分层抽样
            random_state=42  # 随机种子保证可重复性
        )
        X_test = np.vstack((X_test, X_BC))
        y_test=y_test.reshape(len(y_test),1)
        y_BC = y_BC.reshape(len(y_BC), 1)
        y_test = np.vstack((y_test, y_BC))
        # y_test=y_test.reshape(1,len(y_test))
        # # 合并训练集（带标签）
        # train_data = pd.concat([y_train, X_train], axis=1)
        # train_data.columns = data.columns  # 保持列名一致
        #
        # # 合并测试集（带标签）
        # test_data = pd.concat([y_test, X_test], axis=1)
        # test_data.columns = data.columns
        #
        # # 验证结果
        # print("训练集样本分布：")
        # print(train_data.iloc[:, 0].value_counts(normalize=True))
        # print("\n测试集样本分布：")
        # print(test_data.iloc[:, 0].value_counts(normalize=True))
        # data = data.sample(frac=0.7, random_state=42)
        # y = data.iloc[:, 0].values
        # X = data.iloc[:, 1:129]



        # 初始化模型
        model = MCausalSVDD(
            n_classes=3,
            C=1,
            lambda_te=0.3,
            eta=0.1,
            gamma=0.3,
            k="auto",
            pca_var=0.95,
            max_iter=50,
            lr=0.005
        )

        # 训练模型
        model.fit(X_train, y_train)  # 仅使用已知类训练

        # 评估模型
        metrics = model.evaluate(X_test, y_test, known_labels=[0, 1,2], tau=1.2)
        try:
            # 打印评估结果
            print("\n=== 评估结果 ===")
            # print(f"已知类准确率: {metrics['known_accuracy']:.2%}")
            # print(f"未知类拒绝率: {metrics['unknown_rejection_rate']:.2%}")
            # print("混淆矩阵:")
            # print(metrics['confusion_matrix'])
            print(f"TPR: {metrics['tpr']:.2%}")
            print(f"FPR: {metrics['fpr']:.2%}")
            outData.append([metrics['tpr'], metrics['fpr']])
        except Exception as e:
            print(f"训练失败: {str(e)}")
