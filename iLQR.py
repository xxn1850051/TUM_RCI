import numpy as np
np.set_printoptions(precision=4)
rng = np.random.default_rng(1)

from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm

from numpy.fft import fft2, ifft2
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlepad'] = 8.0
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 3
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 3
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['legend.fontsize'] = 15


class iLQR_template:
    def __init__(self, dt, tsteps, x_dim, u_dim, Q_z, R_v) -> None:
        self.dt = dt
        self.tsteps = tsteps

        self.x_dim = x_dim
        self.u_dim = u_dim

        self.Q_z = Q_z
        self.Q_z_inv = np.linalg.inv(Q_z)
        self.R_v = R_v
        self.R_v_inv = np.linalg.inv(R_v)

        self.curr_x_traj = None
        self.curr_y_traj = None

    def dyn(self, xt, ut):
        raise NotImplementedError("Not implemented.")

    def step(self, xt, ut):
        """RK4 integration"""
        k1 = self.dt * self.dyn(xt, ut)
        k2 = self.dt * self.dyn(xt + k1/2.0, ut)
        k3 = self.dt * self.dyn(xt + k2/2.0, ut)
        k4 = self.dt * self.dyn(xt + k3, ut)

        xt_new = xt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        return xt_new

    def traj_sim(self, x0, u_traj):
        x_traj = np.zeros((self.tsteps, self.x_dim))
        xt = x0.copy()
        for t_idx in range(self.tsteps):
            xt = self.step(xt, u_traj[t_idx])
            x_traj[t_idx] = xt.copy()
        return x_traj

    def loss(self):
        raise NotImplementedError("Not implemented.")

    def get_At_mat(self, t_idx):
        raise NotImplementedError("Not implemented.")

    def get_Bt_mat(self, t_idx):
        raise NotImplementedError("Not implemented.")

    def get_at_vec(self, t_idx):
        raise NotImplementedError("Not implemented.")

    def get_bt_vec(self, t_idx):
        raise NotImplementedError("Not implemented.")

    # the following functions are utilities for solving the Riccati equation
    def P_dyn_rev(self, Pt, At, Bt, at, bt):
        return Pt @ At + At.T @ Pt - Pt @ Bt @ self.R_v_inv @ Bt.T @ Pt + self.Q_z

    def P_dyn_step(self, Pt, At, Bt, at, bt):
        k1 = self.dt * self.P_dyn_rev(Pt, At, Bt, at, bt)
        k2 = self.dt * self.P_dyn_rev(Pt+k1/2, At, Bt, at, bt)
        k3 = self.dt * self.P_dyn_rev(Pt+k2/2, At, Bt, at, bt)
        k4 = self.dt * self.P_dyn_rev(Pt+k3, At, Bt, at, bt)

        Pt_new = Pt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        return Pt_new

    def P_traj_revsim(self, PT, A_traj, B_traj, a_traj, b_traj):
        P_traj_rev = np.zeros((self.tsteps, self.x_dim, self.x_dim))
        P_curr = PT.copy()
        for t in range(self.tsteps):
            At = A_traj[-1-t]
            Bt = B_traj[-1-t]
            at = a_traj[-1-t]
            bt = b_traj[-1-t]

            P_new = self.P_dyn_step(P_curr, At, Bt, at, bt)
            P_traj_rev[t] = P_new.copy()
            P_curr = P_new

        return P_traj_rev

    def r_dyn_rev(self, rt, Pt, At, Bt, at, bt):
        return (At - Bt @ self.R_v_inv @ Bt.T @ Pt).T @ rt + at - Pt @ Bt @ self.R_v_inv @ bt

    def r_dyn_step(self, rt, Pt, At, Bt, at, bt):
        k1 = self.dt * self.r_dyn_rev(rt, Pt, At, Bt, at, bt)
        k2 = self.dt * self.r_dyn_rev(rt+k1/2, Pt, At, Bt, at, bt)
        k3 = self.dt * self.r_dyn_rev(rt+k2/2, Pt, At, Bt, at, bt)
        k4 = self.dt * self.r_dyn_rev(rt+k3, Pt, At, Bt, at, bt)

        rt_new = rt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        return rt_new

    def r_traj_revsim(self, rT, P_traj, A_traj, B_traj, a_traj, b_traj):
        r_traj_rev = np.zeros((self.tsteps, self.x_dim))
        r_curr = rT
        for t in range(self.tsteps):
            Pt = P_traj[-1-t]
            At = A_traj[-1-t]
            Bt = B_traj[-1-t]
            at = a_traj[-1-t]
            bt = b_traj[-1-t]

            r_new = self.r_dyn_step(r_curr, Pt, At, Bt, at, bt)
            r_traj_rev[t] = r_new.copy()
            r_curr = r_new

        return r_traj_rev

    def z_dyn(self, zt, Pt, rt, At, Bt, bt):
        return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt, bt)

    def z_dyn_step(self, zt, Pt, rt, At, Bt, bt):
        k1 = self.dt * self.z_dyn(zt, Pt, rt, At, Bt, bt)
        k2 = self.dt * self.z_dyn(zt+k1/2, Pt, rt, At, Bt, bt)
        k3 = self.dt * self.z_dyn(zt+k2/2, Pt, rt, At, Bt, bt)
        k4 = self.dt * self.z_dyn(zt+k3, Pt, rt, At, Bt, bt)

        zt_new = zt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        return zt_new

    def z_traj_sim(self, z0, P_traj, r_traj, A_traj, B_traj, b_traj):
        z_traj = np.zeros((self.tsteps, self.x_dim))
        z_curr = z0.copy()

        for t in range(self.tsteps):
            Pt = P_traj[t]
            rt = r_traj[t]
            At = A_traj[t]
            Bt = B_traj[t]
            bt = b_traj[t]

            z_new = self.z_dyn_step(z_curr, Pt, rt, At, Bt, bt)
            z_traj[t] = z_new.copy()
            z_curr = z_new

        return z_traj

    def z2v(self, zt, Pt, rt, Bt, bt):
        return -self.R_v_inv @ Bt.T @ Pt @ zt - self.R_v_inv @ Bt.T @ rt - self.R_v_inv @ bt

    def get_descent(self, x0, u_traj):
        # forward simulate the trajectory
        x_traj = self.traj_sim(x0, u_traj)
        self.curr_x_traj = x_traj.copy()
        self.curr_u_traj = u_traj.copy()

        # sovle the Riccati equation backward in time
        A_traj = np.zeros((self.tsteps, self.x_dim, self.x_dim))
        B_traj = np.zeros((self.tsteps, self.x_dim, self.u_dim))
        a_traj = np.zeros((self.tsteps, self.x_dim))
        b_traj = np.zeros((self.tsteps, self.u_dim))

        for t_idx in range(self.tsteps):
            A_traj[t_idx] = self.get_At_mat(t_idx)
            B_traj[t_idx] = self.get_Bt_mat(t_idx)
            a_traj[t_idx] = self.get_at_vec(t_idx)
            b_traj[t_idx] = self.get_bt_vec(t_idx)

        # print('a_traj:\n', a_traj)

        PT = np.zeros((self.x_dim, self.x_dim))
        P_traj_rev = self.P_traj_revsim(PT, A_traj, B_traj, a_traj, b_traj)
        P_traj = np.flip(P_traj_rev, axis=0)

        rT = np.zeros(self.x_dim)
        r_traj_rev = self.r_traj_revsim(rT, P_traj, A_traj, B_traj, a_traj, b_traj)
        r_traj = np.flip(r_traj_rev, axis=0)

        z0 = np.zeros(self.x_dim)
        z_traj = self.z_traj_sim(z0, P_traj, r_traj, A_traj, B_traj, b_traj)

        # compute the descent direction
        v_traj = np.zeros((self.tsteps, self.u_dim))
        for t in range(self.tsteps):
            zt = z_traj[t]
            Pt = P_traj[t]
            rt = r_traj[t]
            Bt = B_traj[t]
            bt = b_traj[t]
            v_traj[t] = self.z2v(zt, Pt, rt, Bt, bt)

        return v_traj

class iLQR_ergodic_pointmass(iLQR_template):
    def __init__(self, dt, tsteps, x_dim, u_dim, Q_z, R_v,
                 R, ks, L_list, lamk_list, hk_list, phik_list) -> None:
        super().__init__(dt, tsteps, x_dim, u_dim, Q_z, R_v)

        self.R = R
        self.ks = ks
        self.L_list = L_list
        self.lamk_list = lamk_list
        self.hk_list = hk_list
        self.phik_list = phik_list

    def dyn(self, xt, ut):
        return ut

    def get_At_mat(self, t_idx):
        A = np.zeros((self.x_dim, self.x_dim))
        return A

    def get_Bt_mat(self, t_idx):
        B = np.eye(self.u_dim)
        return B

    def get_at_vec(self, t_idx):
        xt = self.curr_x_traj[t_idx][:2]
        x_traj = self.curr_x_traj[:,:2]

        dfk_xt_all = np.array([
            -np.pi * self.ks[:,0] / self.L_list[0] * np.sin(np.pi * self.ks[:,0] / self.L_list[0] * xt[0]) * np.cos(np.pi * self.ks[:,1] / self.L_list[1] * xt[1]),
            -np.pi * self.ks[:,1] / self.L_list[1] * np.cos(np.pi * self.ks[:,0] / self.L_list[0] * xt[0]) * np.sin(np.pi * self.ks[:,1] / self.L_list[1] * xt[1]),
        ]) / self.hk_list

        fk_all = np.prod(np.cos(np.pi * self.ks / self.L_list * x_traj[:,None]), axis=2) / self.hk_list
        ck_all = np.sum(fk_all, axis=0) * self.dt / (self.tsteps * self.dt)

        at = np.sum(self.lamk_list * 2.0 * (ck_all - self.phik_list) * dfk_xt_all / (self.tsteps * self.dt), axis=1)
        return at

    def get_bt_vec(self, t_idx):
        ut = self.curr_u_traj[t_idx]
        return self.R @ ut

    def loss(self, x_traj, u_traj):
        fk_all = np.prod(np.cos(np.pi * self.ks / self.L_list * x_traj[:,None]), axis=2) / self.hk_list
        ck_all = np.sum(fk_all, axis=0) * self.dt / (self.tsteps * self.dt)
        erg_metric = np.sum(self.lamk_list * np.square(ck_all - self.phik_list))

        ctrl_cost = np.sum(self.R @ u_traj.T * u_traj.T) * self.dt
        return erg_metric + ctrl_cost

class iLQR_ergodic_circular(iLQR_ergodic_pointmass):
    def __init__(self, dt, tsteps, x_dim, u_dim, Q_z, R_v,
                 R, ks, L_list, lamk_list, hk_list, phik_list,
                 radius, num_circle_points) -> None:
        super().__init__(dt, tsteps, x_dim, u_dim, Q_z, R_v,
                         R, ks, L_list, lamk_list, hk_list, phik_list)

        self.radius = radius
        self.num_circle_points = num_circle_points

    def sample_points_in_circle(self, center, num_points):
        """从圆形区域内随机采样点"""
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        radii = np.sqrt(np.random.uniform(0, self.radius**2, num_points))
        sampled_points = np.column_stack((
            center[0] + radii * np.cos(angles),
            center[1] + radii * np.sin(angles)
        ))
        return sampled_points

    def get_at_vec(self, t_idx):
        xt = self.curr_x_traj[t_idx][:2]  # 当前时刻的状态
        sampled_points = self.sample_points_in_circle(xt, num_points=self.num_circle_points)  # 圆形区域内采样点

        # 计算每个采样点的梯度和投影
        dfk_xt_all_list = []
        for point in sampled_points:
            dfk_xt_all = np.array([
                -np.pi * self.ks[:, 0] / self.L_list[0] * np.sin(np.pi * self.ks[:, 0] / self.L_list[0] * point[0]) * np.cos(np.pi * self.ks[:, 1] / self.L_list[1] * point[1]),
                -np.pi * self.ks[:, 1] / self.L_list[1] * np.cos(np.pi * self.ks[:, 0] / self.L_list[0] * point[0]) * np.sin(np.pi * self.ks[:, 1] / self.L_list[1] * point[1]),
            ]) / self.hk_list
            dfk_xt_all_list.append(dfk_xt_all)

        dfk_xt_all = np.mean(dfk_xt_all_list, axis=0)  # 对所有采样点的梯度取平均值

        x_traj = self.curr_x_traj[:, :2]
        fk_all = np.prod(np.cos(np.pi * self.ks / self.L_list * x_traj[:, None]), axis=2) / self.hk_list
        ck_all = np.sum(fk_all, axis=0) * self.dt / (self.tsteps * self.dt)

        at = np.sum(self.lamk_list * 2.0 * (ck_all - self.phik_list) * dfk_xt_all / (self.tsteps * self.dt), axis=1)
        return at

# todo: finish FFT approach
class iLQR_ergodic_fft(iLQR_template):
    def __init__(self, dt, tsteps, x_dim, u_dim, Q_z, R_v,
                 R, grid_size, workspace_size, radius) -> None:
        super().__init__(dt, tsteps, x_dim, u_dim, Q_z, R_v)
        self.R = R
        self.grid_size = grid_size  # 网格大小 (n_x, n_y)
        self.workspace_size = workspace_size  # 工作空间大小 [x_min, x_max, y_min, y_max]
        self.radius = radius

        # 初始化网格
        self.x_grid = np.linspace(workspace_size[0], workspace_size[1], grid_size[0])
        self.y_grid = np.linspace(workspace_size[2], workspace_size[3], grid_size[1])
        self.dx = (workspace_size[1] - workspace_size[0]) / grid_size[0]
        self.dy = (workspace_size[3] - workspace_size[2]) / grid_size[1]
        self.circle_kernel = self.create_circle_kernel()

    def create_circle_kernel(self):
        """创建圆形kernel用于表示接触面"""
        kernel_size = int(2 * self.radius / min(self.dx, self.dy))
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保kernel大小为奇数

        center = kernel_size // 2
        y, x = np.ogrid[-center:center + 1, -center:center + 1]
        dist_from_center = np.sqrt(x ** 2 + y ** 2)

        # 创建圆形kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[dist_from_center <= center] = 1

        # 归一化
        kernel = kernel / kernel.sum()
        return kernel

    def trajectory_to_grid(self, trajectory):
        """将轨迹转换为网格表示"""
        grid = np.zeros(self.grid_size)

        # 将轨迹点映射到网格索引
        x_indices = np.clip(((trajectory[:, 0] - self.workspace_size[0]) / self.dx).astype(int),
                            0, self.grid_size[0] - 1)
        y_indices = np.clip(((trajectory[:, 1] - self.workspace_size[2]) / self.dy).astype(int),
                            0, self.grid_size[1] - 1)

        # 为每个轨迹点添加圆形接触面
        for x_idx, y_idx in zip(x_indices, y_indices):
            # 使用圆形kernel进行卷积
            kernel_height, kernel_width = self.circle_kernel.shape
            h_start = max(0, x_idx - kernel_height // 2)
            h_end = min(self.grid_size[0], x_idx + kernel_height // 2 + 1)
            w_start = max(0, y_idx - kernel_width // 2)
            w_end = min(self.grid_size[1], y_idx + kernel_width // 2 + 1)

            k_h_start = max(0, kernel_height // 2 - x_idx)
            k_h_end = min(kernel_height, kernel_height // 2 + (self.grid_size[0] - x_idx))
            k_w_start = max(0, kernel_width // 2 - y_idx)
            k_w_end = min(kernel_width, kernel_width // 2 + (self.grid_size[1] - y_idx))

            grid[h_start:h_end, w_start:w_end] += \
                self.circle_kernel[k_h_start:k_h_end, k_w_start:k_w_end]

        # 归一化
        if grid.sum() > 0:
            grid = grid / grid.sum()

        return grid

    def compute_fft_difference(self, grid1, grid2):
        """计算两个网格的FFT差异"""
        fft1 = np.fft.fft2(grid1)
        fft2 = np.fft.fft2(grid2)

        # 计算频域差异
        fft_diff = np.abs(fft1 - fft2)
        return fft_diff

    def compute_gradient(self, curr_grid, target_grid, xt):
        """计算梯度"""
        fft_diff = self.compute_fft_difference(curr_grid, target_grid)
        grad_x = np.zeros(2)

        # 找到当前状态在网格中的位置
        x_idx = int((xt[0] - self.workspace_size[0]) / self.dx)
        y_idx = int((xt[1] - self.workspace_size[2]) / self.dy)

        # 考虑圆形接触面范围内的梯度
        kernel_size = self.circle_kernel.shape[0]
        radius_grid = kernel_size // 2

        for dx in range(-radius_grid, radius_grid + 1):
            for dy in range(-radius_grid, radius_grid + 1):
                curr_x = x_idx + dx
                curr_y = y_idx + dy

                if (0 < curr_x < self.grid_size[0] - 1 and
                        0 < curr_y < self.grid_size[1] - 1):
                    # 计算该点对梯度的贡献
                    weight = self.circle_kernel[dx + radius_grid, dy + radius_grid]
                    grad_x[0] += weight * (fft_diff[curr_x + 1, curr_y] -
                                           fft_diff[curr_x - 1, curr_y]) / (2 * self.dx)
                    grad_x[1] += weight * (fft_diff[curr_x, curr_y + 1] -
                                           fft_diff[curr_x, curr_y - 1]) / (2 * self.dy)

        return grad_x

    def dyn(self, xt, ut):
        return ut

    def get_At_mat(self, t_idx):
        A = np.zeros((self.x_dim, self.x_dim))
        return A

    def get_Bt_mat(self, t_idx):
        B = np.eye(self.u_dim)
        return B

    def get_at_vec(self, t_idx):
        xt = self.curr_x_traj[t_idx][:2]

        # 将当前完整轨迹转换为网格表示
        curr_traj_grid = self.trajectory_to_grid(self.curr_x_traj[:, :2])

        # 将目标轨迹转换为网格表示（这部分需要在初始化时完成并存储）
        target_traj_grid = self.trajectory_to_grid(u_traj[:, :2])

        # 计算梯度
        at = self.compute_gradient(curr_traj_grid, target_traj_grid, xt)
        return at

    def get_bt_vec(self, t_idx):
        ut = self.curr_u_traj[t_idx]
        return self.R @ ut

    def loss(self, x_traj, u_traj):
        # 计算轨迹网格表示
        curr_grid = self.trajectory_to_grid(x_traj[:, :2])
        target_grid = self.trajectory_to_grid(u_traj[:, :2])

        # 计算FFT差异
        fft_diff = self.compute_fft_difference(curr_grid, target_grid)
        erg_metric = np.sum(np.abs(fft_diff))

        # 控制成本
        ctrl_cost = np.sum(self.R @ u_traj.T * u_traj.T) * self.dt

        return erg_metric + ctrl_cost

## Trajectory optimization pipeline
import numpy as np
from src.libs.ergodic_control_HEDAC_2D import ErgodicControlHEDAC2D
from src.libs.ergodic_control_SMC_2D import ErgodicControlSMC2D
erg_ctrl_HEDAC = ErgodicControlHEDAC2D()
erg_ctrl_SMC = ErgodicControlSMC2D()


# 定义更多不同的目标分布 define more different target distributions
def create_target_distribution(distribution_type='default'):
    """
    创建不同类型的多峰目标分布

    Parameters:
    -----------
    distribution_type : str, optional
        目标分布的类型
    L_list : numpy.ndarray
        搜索空间边界

    Returns:
    --------
    tuple: (pdf_function, distribution_info)
        - pdf_function: 概率密度函数
        - distribution_info: 包含均值、协方差和权重的字典
    """
    distributions = {
        'GMM': {
            'means': [
                np.array([0.5, 0.7]),
                np.array([0.6, 0.3])
            ],
            'covs': [
                np.array([[0.05 , 0.015], [0.015, 0.01]]),
                np.array([[0.013, 0.006], [0.006, 0.022]])
            ],
            'weights': [0.5, 0.5]
        },
        'uniform': {
            'means': None,
            'covs': None,
            'weights': None
        },
        'default': {
            'means': [
                np.array([0.35, 0.38]),
                np.array([0.68, 0.25]),
                np.array([0.56, 0.64])
            ],
            'covs': [
                np.array([[0.01, 0.004], [0.004, 0.01]]),
                np.array([[0.005, -0.003], [-0.003, 0.005]]),
                np.array([[0.008, 0.0], [0.0, 0.004]])
            ],
            'weights': [0.5, 0.2, 0.3]
        }
    }

    dist_info = distributions.get(distribution_type, distributions['default']) # 默认返回default值

    def pdf(x):
        if distribution_type == 'uniform':
            # 在搜索空间内返回常数值，保证积分为1
            return np.full(x.shape[0], 1.0)

        pdf_val = 0
        for mean, cov, weight in zip(dist_info['means'], dist_info['covs'], dist_info['weights']):
            pdf_val += weight * mvn.pdf(x, mean, cov)
        return pdf_val

    return pdf, dist_info

# todo: use different target distribution
target_distribution = 'GMM'
pdf, distribution_info = create_target_distribution(target_distribution)

# Define a 1-by-1 2D search space
L_list = np.array([1.0, 1.0])  # boundaries for each dimension

# Discretize the search space into 100-by-100 mesh grids
grids_x, grids_y = np.meshgrid(
    np.linspace(0, L_list[0], 100),
    np.linspace(0, L_list[1], 100)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
dx = 1.0 / 99
dy = 1.0 / 99


# Configure the index vectors
num_k_per_dim = 10
ks_dim1, ks_dim2 = np.meshgrid(
    np.arange(num_k_per_dim), np.arange(num_k_per_dim)
)
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T

# Pre-processing lambda_k and h_k
lamk_list = np.power(1.0 + np.linalg.norm(ks, axis=1), -3/2.0)
hk_list = np.zeros(ks.shape[0])
for i, k_vec in enumerate(ks):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
    hk_list[i] = hk

# compute the coefficients for the target distribution
phik_list = np.zeros(ks.shape[0])
pdf_vals = pdf(grids)
for i, (k_vec, hk) in enumerate(zip(ks, hk_list)):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)
    fk_vals /= hk

    phik = np.sum(fk_vals * pdf_vals) * dx * dy
    phik_list[i] = phik


# Define the optimal control problem
dt = 0.1
tsteps = 100
R = np.diag([0.0001, 0.0001])
Q_z = np.diag([0.01, 0.01])
R_v = np.diag([0.01, 0.01])

# define initial trajectories as the initial control
def generate_spiral_trajectory(tsteps, dt):
    # x0 = rng.uniform(low=0.4, high=0.6, size=(2,))
    # x0 = np.array([0.1, 0.3])
    x0 = np.array([0.4, 0.6])
    temp_x_traj = np.array([
        np.linspace(0.0, 0.3, tsteps + 1) * np.cos(np.linspace(0.0, 2 * np.pi, tsteps + 1)),
        np.linspace(0.0, 0.3, tsteps + 1) * np.sin(np.linspace(0.0, 2 * np.pi, tsteps + 1))
    ]).T
    init_u_traj = (temp_x_traj[1:, :] - temp_x_traj[:-1, :]) / dt
    return x0, init_u_traj

def generate_HEDAC_trajectory(tsteps, dt):
    x0_unorm = np.array([40, 60]) # 修改类定义中的初始位置
    trajs = erg_ctrl_HEDAC.run_from_init_pos(x0_unorm)
    initial_traj_unormal = trajs[erg_ctrl_HEDAC.agents[0]]

    # rescale HEDAC trajectory --> [0,1]
    flattened_data = np.concatenate(initial_traj_unormal)
    min_val = flattened_data.min()
    max_val = flattened_data.max()
    initial_traj = [(arr - min_val) / (max_val - min_val) for arr in initial_traj_unormal]
    # x0 = (x0_unorm - min_val) / (max_val - min_val)
    x0 = x0_unorm / 100
    init_u_traj = convert_traj_to_control(initial_traj, dt)
    return x0, init_u_traj

def generate_SMC_trajectory(tsteps, dt):
    # x0 = np.array([0.1, 0.3])
    x0 = np.array([0.4, 0.6])
    initial_traj_unormal = erg_ctrl_SMC.run_from_init_pos(x0)

    # rescale SMC trajectory --> [0,1]
    flattened_data = np.concatenate(initial_traj_unormal)
    min_val = flattened_data.min()
    max_val = flattened_data.max()
    initial_traj = [(arr - min_val) / (max_val - min_val) for arr in initial_traj_unormal]
    init_u_traj = convert_traj_to_control(initial_traj, dt)
    return x0, init_u_traj

# 需要将HEDAC/SMC轨迹转换为适合iLQR优化的格式
def convert_traj_to_control(traj, dt):
    """将位置轨迹转换为控制输入"""
    # 确保轨迹长度匹配
    if len(traj) > tsteps + 1:
        traj = traj[:tsteps+1]
    elif len(traj) < tsteps + 1:
        # 需要补充轨迹点到所需长度
        last_pt = traj[-1]
        padding = np.tile(last_pt, (tsteps + 1 - len(traj), 1))
        traj = np.vstack([traj, padding])

    # 将列表转换为NumPy数组
    traj = np.array(traj)

    # 检查轨迹维度是否正确
    if len(traj.shape) == 1:
        traj = traj.reshape(-1, 2)  # 假设是2D轨迹

    # 计算控制输入（速度）
    init_u_traj = (traj[1:, :] - traj[:-1, :]) / dt
    return init_u_traj

# 添加更多的轨迹生成方法 add more trajectory generators
trajectory_generators = {
    "spiral": generate_spiral_trajectory,
    "HEDAC": generate_HEDAC_trajectory,
    "SMC": generate_SMC_trajectory
}

# todo: use different initial trajectories here.
init_traj_type = "spiral"

# todo: initialize iLQR optimization

# pointmass
trajopt_ergodic = iLQR_ergodic_pointmass(
    dt, tsteps, x_dim=2, u_dim=2, Q_z=Q_z, R_v=R_v,
    R=R, ks=ks, L_list=L_list, lamk_list=lamk_list,
    hk_list=hk_list, phik_list=phik_list
)

# ## circular
# radius = 0.05
# trajopt_ergodic = iLQR_ergodic_circular(
#     dt, tsteps, x_dim=2, u_dim=2, Q_z=Q_z, R_v=R_v,
#     R=R, ks=ks, L_list=L_list, lamk_list=lamk_list,
#     hk_list=hk_list, phik_list=phik_list,
#     radius=radius, num_circle_points=20
# )

## FFT
# workspace_size = [0, 1, 0, 1]  # [x_min, x_max, y_min, y_max]
# grid_size = (100, 100)
#
# trajopt_ergodic = iLQR_ergodic_fft(
#     dt, tsteps, x_dim=2, u_dim=2, Q_z=Q_z, R_v=R_v,
#     R=R, grid_size=grid_size, workspace_size=workspace_size,
#     radius=radius
# )

# Iterative trajectory optimization for ergodic control
import time
from IPython import display

start_time = time.time()

x0, init_u_traj = trajectory_generators[init_traj_type](tsteps, dt)
u_traj = init_u_traj.copy()
step = 0.01
loss_list = []

fig, axes = plt.subplots(1, 2, dpi=70, figsize=(15,5), tight_layout=True)

for iter in tqdm(range(100)):
    x_traj = trajopt_ergodic.traj_sim(x0, u_traj)
    v_traj = trajopt_ergodic.get_descent(x0, u_traj)

    loss_val = trajopt_ergodic.loss(x_traj, u_traj)
    loss_list.append(loss_val)
    reduction_ratio = loss_list[0] / loss_val if loss_val > 0 else float('inf')

    step = 0.002
    alpha = 0.5
    for _i in range(3):
        temp_u_traj = u_traj + step * v_traj
        temp_x_traj = trajopt_ergodic.traj_sim(x0, temp_u_traj)
        temp_loss_val = trajopt_ergodic.loss(temp_x_traj, temp_u_traj)
        if temp_loss_val < loss_val:
            break
        else:
            step *= alpha
    u_traj += step * v_traj

    # visualize every 10 iterations
    if (iter+1) % 10 == 0:
        ax1 = axes[0]
        ax1.cla()
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(0.0, L_list[0])
        ax1.set_ylim(0.0, L_list[1])
        ax1.set_title('Iteration: {:d}'.format(iter+1))
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.contourf(grids_x, grids_y, pdf_vals.reshape(grids_x.shape), cmap='Reds')
        ax1.plot([x0[0], x_traj[0,0]], [x0[1], x_traj[0,1]], linestyle='-', linewidth=2, color='k', alpha=1.0)
        # # plot circles
        # for i in range(len(x_traj)):
        #     circle = plt.Circle((x_traj[i, 0], x_traj[i, 1]), radius, color='k', fill=False, alpha=0.5)
        #     ax1.add_patch(circle)
        ax1.plot(x_traj[:,0], x_traj[:,1], linestyle='-', marker='o', color='k', linewidth=2, alpha=1.0, label='Optimized trajectory')
        ax1.plot(x0[0], x0[1], linestyle='', marker='o', markersize=15, color='C0', alpha=1.0, label='Initial state')
        ax1.legend(loc=1)

        ax3 = axes[1]
        ax3.cla()
        ax3.set_title('Objective vs. Iteration')
        ax3.set_xlim(-0.2, 100.2)
        ax3.set_ylim(3e-3, 1e0)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Objective')
        ax3.plot(np.arange(iter+1), loss_list, color='C3')
        ax3.text(
            95, 0.8, f'Reduction Ratio: {reduction_ratio:.2f}', fontsize=20, color='black', ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
        height = ax1.get_position().height
        ax3.set_position([ax3.get_position().x0, ax1.get_position().y0, ax3.get_position().width, height])
        ax3.set_yscale('log')

        display.clear_output(wait=True)
        display.display(fig)

end_time = time.time()
optimization_time = end_time - start_time
print(f"Optimization took {optimization_time:.2f} seconds.")

display.clear_output(wait=True)
plt.show()
plt.close()

# # robot simulator
# from src.ergodic_control.examples.hedac_2d_online import state_to_3d
# from src.libs.sim import Simulator
#
# input("Simulator (press enter to continue)")
# sim = Simulator()
# sim.setup_scenario(local=True, tool="grinder")
#
# def pos_controller(t):
#     i = min(int(t / sim.time_step), len(x_traj) - 1)
#     traj_pt = state_to_3d(x_traj[i], scale=0.6)
#     print(f"time: {t} pos: {traj_pt.T}")
#     return traj_pt
#
# sim.add_traj_pos_ctl(pos_controller)
# input("(press enter to quit)")