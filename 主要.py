import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

设备 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class 记忆体:
    def __init__(self):
        self.动作_列表 = []
        self.状态_列表 = []
        self.概率质量_对数_列表 = []  # 概率质量函数的对数列表。
        self.奖励_列表 = []
        self.终止_列表 = []

    def 清空记忆体(self):
        del self.动作_列表[:]
        del self.状态_列表[:]
        del self.概率质量_对数_列表[:]
        del self.奖励_列表[:]
        del self.终止_列表[:]


class 行动者与评论家(nn.Module):
    def __init__(self, 状态_维度, 动作_维度, 隐藏层_变量数):
        super(行动者与评论家, self).__init__()

        self.行动者 = nn.Sequential(
            nn.Linear(状态_维度, 隐藏层_变量数),
            nn.Tanh(),
            nn.Linear(隐藏层_变量数, 隐藏层_变量数),
            nn.Tanh(),
            nn.Linear(隐藏层_变量数, 动作_维度),
            nn.Softmax(dim=-1)
        )

        self.评论家 = nn.Sequential(
            nn.Linear(状态_维度, 隐藏层_变量数),
            nn.Tanh(),
            nn.Linear(隐藏层_变量数, 隐藏层_变量数),
            nn.Tanh(),
            nn.Linear(隐藏层_变量数, 1)
        )

    def forward(self):
        raise NotImplemented

    def 动作(self, 状态, 一个记忆体):
        状态 = torch.from_numpy(状态).float().to(设备)
        # 有四个离散的动作可用：什么都不做，触发左方向引擎，触发主引擎，触发右方向引擎。
        动作_概率 = self.行动者(状态)
        分布 = Categorical(动作_概率)
        动作 = 分布.sample()

        一个记忆体.状态_列表.append(状态)
        一个记忆体.动作_列表.append(动作)
        一个记忆体.概率质量_对数_列表.append(分布.log_prob(动作))

        return 动作.item()

    def 评估(self, 状态, 动作):
        动作_概率_列表 = self.行动者(状态)
        分布 = Categorical(动作_概率_列表)

        动作_概率质量_对数_列表 = 分布.log_prob(动作)
        分布_熵 = 分布.entropy()

        # 用来评判所有状态是否是好的，即获取更高的奖励
        状态评判值 = self.评论家(状态)

        return 动作_概率质量_对数_列表, torch.squeeze(状态评判值), 分布_熵


class 近端策略优化:
    def __init__(self, 状态_维度, 动作_维度, 隐藏层_变量数, 学习率, 贝塔值, 伽马值, 轮回次数, 夹子_间距):
        self.学习率 = 学习率
        self.贝塔值 = 贝塔值
        self.伽马值 = 伽马值
        self.夹子_间距 = 夹子_间距
        self.轮回次数 = 轮回次数

        self.策略 = 行动者与评论家(状态_维度, 动作_维度, 隐藏层_变量数).to(设备)
        self.优化器 = torch.optim.Adam(self.策略.parameters(), lr=学习率, betas=贝塔值)
        self.旧策略 = 行动者与评论家(状态_维度, 动作_维度, 隐藏层_变量数).to(设备)
        self.旧策略.load_state_dict(self.策略.state_dict())

        self.二元交叉熵损失值 = nn.MSELoss()

    def 更新(self, 一个记忆体):
        奖励列表 = []
        奖励_折扣 = 0
        for 奖励, 终止 in zip(reversed(一个记忆体.奖励_列表), reversed(一个记忆体.终止_列表)):
            if 终止:
                奖励_折扣 = 0
            奖励_折扣 = 奖励 + (self.伽马值 * 奖励_折扣)
            奖励列表.insert(0, 奖励_折扣)

        奖励列表 = torch.tensor(奖励列表, dtype=torch.float32).to(设备)
        奖励列表 = (奖励列表 - 奖励列表.mean()) / (奖励列表.std() + 1e-5)

        旧状态列表 = torch.stack(一个记忆体.状态_列表).to(设备).detach()
        旧动作列表 = torch.stack(一个记忆体.动作_列表).to(设备).detach()
        旧概率质量对数列表 = torch.stack(一个记忆体.概率质量_对数_列表).to(设备).detach()

        for _ in range(self.轮回次数):
            概率质量对数列表, 状态评判值, 分布_熵 = self.策略.评估(旧状态列表, 旧动作列表)

            新比旧列表 = torch.exp(概率质量对数列表 - 旧概率质量对数列表.detach())
            优化后奖励列表 = 奖励列表 - 状态评判值.detach()
            代理人1 = 新比旧列表 * 优化后奖励列表
            代理人2 = torch.clamp(新比旧列表, 1 - self.夹子_间距, 1 + self.夹子_间距) * 优化后奖励列表
            损失值 = -torch.min(代理人1, 代理人2) + 0.5 * self.二元交叉熵损失值(状态评判值, 奖励列表) - 0.01 * 分布_熵

            self.优化器.zero_grad()
            损失值.mean().backward()
            self.优化器.step()

        self.旧策略.load_state_dict(self.策略.state_dict())


def 主要():
    游戏环境名 = "LunarLander-v2"
    # 游戏环境 = gym.make(游戏环境名, render_mode='human')
    游戏环境 = gym.make(游戏环境名)
    状态_维度 = 游戏环境.observation_space.shape[0]
    动作_维度 = 4
    是否渲染 = False
    奖励_极值 = 230
    日志_间隔 = 20
    周期_最大值 = 5000
    """
        时间步长：
        许多物理方程将以这样的形式出现：“从现在开始 1 秒的速度 = 现在的速度 + 1 秒的加速度将产生的速度”（并且大多数不适合这种形式的方程可以以某种方式转换...... )
        “1 秒”部分是时间步长。定期改变世界的游戏会执行大量此类计算，并且要保持正常运行，需要使用一致的时间步长值评估所有这些物理方程。
        游戏经常使用接近每个渲染图形帧之间周期的时间步长，或图形帧之间的理想时间，例如 0.01667 秒，但这不是必需的。一些像 BeamNG.drive 这样的游戏会比游戏渲染更频繁地​​计算物理，以生成更平滑的物理近似。
        其他游戏将不那么频繁地计算物理并混合每个点之间渲染的位置。他们的方法将取决于每个游戏。
    """
    时间步长_最大值 = 300  # 时间的步长，这是一个单位
    隐藏层_变量数 = 64  # 隐藏层的变量数
    时间步长_更新_间隔 = 2000
    学习率 = 0.002
    贝塔值 = (0.9, 0.999)  # 用来计算梯度的平均数和平方的系数
    伽马值 = 0.99  # 折扣因子
    轮回次数 = 4  # 模型更新策略
    夹子_间距 = 0.2  # 夹子函数的范围
    随机_种子 = None

    if 随机_种子:
        torch.manual_seed(随机_种子)

    一个记忆体 = 记忆体()

    一个近端策略优化 = 近端策略优化(状态_维度, 动作_维度, 隐藏层_变量数, 学习率, 贝塔值, 伽马值, 轮回次数, 夹子_间距)
    运行时奖励 = 0
    冒险_次数 = 0
    时间步长 = 0

    for 索引_周期 in range(1, 周期_最大值 + 1):
        # 状态是一个 8 维向量：着陆器在 x 和 y 中的坐标，在 x 和 y 中的线速度，它的角度，它的角速度，以及表示每条腿是否与地面接触的两个布尔值。
        # _是一个丢弃的值，里面存放的是信息，但我并不使用它，它是空的。
        状态, _ = 游戏环境.reset()
        for t in range(时间步长_最大值):
            时间步长 += 1
            动作 = 一个近端策略优化.旧策略.动作(状态, 一个记忆体)
            状态, 奖励, 完毕, _, _ = 游戏环境.step(动作)

            一个记忆体.奖励_列表.append(奖励)
            一个记忆体.终止_列表.append(完毕)

            if 时间步长 % 时间步长_更新_间隔 == 0:
                一个近端策略优化.更新(一个记忆体)
                一个记忆体.清空记忆体()
                时间步长 = 0

            运行时奖励 += 奖励
            if 是否渲染:
                游戏环境.render()
            if 完毕:
                break

        冒险_次数 += t

        if 运行时奖励 > (日志_间隔 * 奖励_极值):
            print("------------ 已解决 -------------")
            torch.save(一个近端策略优化.策略.state_dict(), '近端策略优化_{}.pth'.format(游戏环境名))
            break

        if 索引_周期 % 日志_间隔 == 0:
            冒险_次数 = int(冒险_次数 / 日志_间隔)
            运行时奖励 = int(运行时奖励 / 日志_间隔)

            print("周期：{} \t 冒险次数：{} \t奖励：{}".format(索引_周期, 冒险_次数, 运行时奖励))
            运行时奖励 = 0
            冒险_次数 = 0
    游戏环境.close()

if __name__ == '__main__':
    主要()
