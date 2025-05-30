import re
from math import pi
import sympy as sp
import xlrd
import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import norm
import os
import json
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
print(f"当前文件所在的目录: {current_script_dir}")
class GongShui:

    def __init__(self, input_json, out_json):
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                self.input_json = json.load(f)
        except FileNotFoundError:
            print(f"未找到 {input_json} 文件")
        except json.JSONDecodeError:
            print(f"{input_json} 文件不是有效的 JSON 格式")

        # 读取 out_json 文件内容
        try:
            with open(out_json, 'r', encoding='utf-8') as f:
                self.out_json = json.load(f)
        except FileNotFoundError:
            print(f"未找到 {out_json} 文件")
        except json.JSONDecodeError:
            print(f"{out_json} 文件不是有效的 JSON 格式")

    def function1(self):
        pipes_wb = xlrd.open_workbook(os.path.join(current_script_dir,"input", self.input_json["管道信息"]))
        joints_wb = xlrd.open_workbook(os.path.join(current_script_dir,"input", self.input_json["节点信息"]))
        pipes_info = pipes_wb.sheet_by_index(0)
        joints_info = joints_wb.sheet_by_index(0)
        num_pipes = pipes_info.nrows - 1
        num_joints = joints_info.nrows - 1
        pgv_df = pd.read_csv(os.path.join(current_script_dir,"input", self.input_json["其他信息"]))
        pgv_dict = dict(zip(pgv_df['node_id'].astype(int), pgv_df['pgv50']))
        pipe_id = list(range(1, num_pipes + 1))
        joint_id = list(range(1, num_joints + 1))
        joint_iddi = joint_id.copy()
        A = sp.zeros(num_joints, num_pipes)
        for i in range(num_pipes):
            start = int(pipes_info.cell_value(i + 1, 1)) - 1
            end = int(pipes_info.cell_value(i + 1, 2)) - 1
            A[start, i] = 1
            A[end, i] = -1
        mats = pipes_info.col_values(3)[1:]
        Cs = [130 if m == '新铸铁管' else 100 for m in mats]
        lengths = pipes_info.col_values(4)[1:]
        diams = pipes_info.col_values(5)[1:]
        pipe_s = [0.27853 * Cs[i] * (diams[i] / 1000) ** 2.63 * lengths[i] ** -0.54 for i in range(num_pipes)]
        H = sp.symbols([f'H{i}' for i in range(num_joints)])
        q = list(sp.symbols([f'q{i}' for i in range(num_pipes)]))
        for i in range(num_pipes):
            idx = [j for j in range(num_joints) if A[j, i] != 0]
            hs, he = idx[0], idx[1]
            q[i] = pipe_s[i] * (H[hs] - H[he]) * abs(H[hs] - H[he]) ** -0.46
        reservoirs_id, reservoirs_H = [], []
        Qn = joints_info.col_values(1)[1:]
        for i, v in enumerate(Qn):
            if v == '供水点':
                Qn[i] = 0
                reservoirs_id.append(int(joints_info.cell_value(i + 1, 0)))
                reservoirs_H.append(int(joints_info.cell_value(i + 1, 2)))
            else:
                Qn[i] = v / 1000
        num_reservoirs = len(reservoirs_id)
        joint_imp = joints_info.col_values(3)[1:]
        total = sum(Qn[i] * joint_imp[i] for i in range(num_joints))
        joint_w = [Qn[i] * joint_imp[i] / total for i in range(num_joints)]
        EarthquakeIntensity = '8';
        SoilType = '二类土'
        ShearWaveV = {'一类土': 700, '二类土': 330, '三类土': 200, '四类土': 100}[SoilType] * 1000
        mu_R1, sigma_R1 = (0.32, 0.18)
        mu_R2, sigma_R2 = (2.65, 1.08)
        E = 1e11
        leak_ratio = 0.8
        Lp = 6000
        pipe_condition = [None] * num_pipes
        pipe_Al = []
        breach_info = []
        for i in range(num_pipes):
            start, end = int(pipes_info.cell_value(i + 1, 1)), int(pipes_info.cell_value(i + 1, 2))
            pgv = (pgv_dict.get(start, 0) + pgv_dict.get(end, 0)) / 2
            L_km = lengths[i] / 1000
            RR = np.exp(1.21 * np.log(pgv * 100) - 6.81)
            positions = []
            s = 0.0
            while True:
                gap = np.random.exponential(scale=1.0 / RR)
                s += gap
                if s >= L_km:
                    break
                positions.append(s / L_km)
            if not positions:
                pipe_condition[i] = '基本完好'
                pipe_Al.append(0)
            else:
                types = ['渗漏' if np.random.rand() < leak_ratio else '断开' for _ in positions]
                if all(t == '渗漏' for t in types):
                    pipe_condition[i] = '部分损坏'
                    t_wall = 9 * (0.5 + 0.001 * diams[i])
                    soil_str = pgv * 1000 / (4 / 3 * ShearWaveV)
                    factor = 1 + (E / 1e5) * (pi * diams[i] * t_wall / 100) * (diams[i] / 10) / (
                                2 * (ShearWaveV / 10) ** 2)
                    pipe_str = soil_str / factor
                    s_area = Lp * pipe_str
                    mu_S = s_area / 2.2
                    sig_S = 0.4 * mu_S
                    Pf_leak = norm.cdf((mu_R1 + diams[i] ** 2 / 4 - mu_S) / sig_S) - norm.cdf((mu_R1 - mu_S) / sig_S)
                    Pu = norm.cdf(-(mu_R1 + diams[i] ** 2 / 4 - mu_S) / sig_S)
                    E_Al = (pi / 2) ** 0.5 * diams[i] * sig_S * (
                            np.exp(-(mu_R1 - mu_S) ** 2 / (2 * sig_S ** 2)) -
                            np.exp(-(mu_R2 - mu_S) ** 2 / (2 * sig_S ** 2))
                    ) + pi * diams[i] * (mu_S - mu_R1) * Pf_leak + pi * diams[i] ** 2 / 4 * Pu
                    pipe_Al.append((int(lengths[i] / Lp) + 1) * E_Al)
                else:
                    pipe_condition[i] = '严重损坏'
                    pipe_Al.append(1)
            breach_info.append({
                'pipe_id': pipe_id[i],
                'num_breaches': len(positions),
                'positions': ';'.join(f'{p:.4f}' for p in positions)
            })
        print(pipe_condition)
        import csv
        with open(os.path.join(current_script_dir,"output", self.out_json["断点数量及位置"]), 'w', newline='', encoding='utf-8') as bf:
            writer = csv.DictWriter(bf, fieldnames=['pipe_id', 'num_breaches', 'positions'])
            writer.writeheader()
            writer.writerows(breach_info)
            dict_perbroken = {'6': (4000, 5000),
                              '7': (2000, 4000),
                              '8': (1000, 2000),
                              '9': (500, 1000)}
            per_broken1, per_broken2 = dict_perbroken[EarthquakeIntensity]
            time_broken1 = 3 * np.random.randn() + 6
            time_broken2 = 6 * np.random.randn() + 12
            num_broken, time = [], []
            for i, cond in enumerate(pipe_condition):
                if cond == '基本完好':
                    num_broken.append(0)
                    time.append(0)
                elif cond == '部分损坏':
                    nb = int(lengths[i] / per_broken1) + 1
                    num_broken.append(nb)
                    time.append(nb * time_broken1)
                else:
                    nb = int(lengths[i] / per_broken2) + 1
                    num_broken.append(nb)
                    time.append(nb * time_broken2)
            all_time = int(sum(time) / 24) + 1
            condition_cnt = -1
            for i in range(len(pipe_condition)):
                if pipe_condition[i] == '部分损坏':
                    condition_cnt += 1
                    A = A.row_insert(
                        num_joints + condition_cnt,
                        sp.zeros(1, num_pipes + condition_cnt)
                    )
                    A = A.col_insert(
                        num_pipes + condition_cnt,
                        sp.zeros(num_joints + condition_cnt + 1, 1)
                    )
                    for j in range(num_joints):
                        if A[j, i] == 1:
                            pipe_start = j
                        elif A[j, i] == -1:
                            pipe_end = j
                    A[(num_joints + condition_cnt), i] = -1
                    A[pipe_end, i] = 0
                    A[(num_joints + condition_cnt), num_pipes + condition_cnt] = 1
                    A[pipe_end, num_pipes + condition_cnt] = -1
                    H_new = sp.symbols("H{}".format(num_joints + condition_cnt))
                    H = H + [H_new]
                    q[i] = (0.5 ** -0.54) * pipe_s[i] * \
                           (H[pipe_start] - H[num_joints + condition_cnt]) * \
                           ((((H[pipe_start] - H[num_joints + condition_cnt]) ** 2) ** 0.5) ** -0.46)
                    q_new = (0.5 ** -0.54) * pipe_s[i] * \
                            (H[num_joints + condition_cnt] - H[pipe_end]) * \
                            ((((H[num_joints + condition_cnt] - H[pipe_end]) ** 2) ** 0.5) ** -0.46)
                    q = q + [q_new]
                    Ql = 4.427 * 0.6 * H_new ** 0.5 * (pipe_Al[i] / 1000000)
                    Qn.append(Ql)
                    pipe_id.append(num_pipes + condition_cnt + 1)
                    joint_id.append(num_joints + condition_cnt + 1)
            damaged_pipes = 0
            for i in range(len(pipe_condition)):
                if pipe_condition[i] == '严重损坏':
                    A.col_del(i - damaged_pipes)
                    del pipe_id[i - damaged_pipes]
                    del q[i - damaged_pipes]
                    damaged_pipes += 1
            num_joints = A.shape[0]
            num_pipes = A.shape[1]
            Con = sp.zeros(num_joints, num_joints)
            for i in range(num_pipes):
                for j in range(num_joints):
                    if A[j, i] == 1:
                        pipe_start = j
                    elif A[j, i] == -1:
                        pipe_end = j
                Con[pipe_start, pipe_end] = 1
                Con[pipe_end, pipe_start] = 1
            for i in range(num_joints):
                Con[i, i] = 1
            Connectivity = Con
            for k in range(num_joints - 2):
                Connectivity += Con ** (k + 2)
            disconnected_joints = 0
            disconnected_ext = 0
            for i in range(num_joints):
                if_del = 0
                for j in range(num_reservoirs):
                    if Connectivity[i, reservoirs_id[j] - 1] == 0:
                        if_del += 1
                        if i > len(joint_iddi) - 1:
                            disconnected_ext += 1
                if if_del == num_reservoirs:
                    A.row_del(i - disconnected_joints)
                    del joint_id[i - disconnected_joints]
                    del Qn[i - disconnected_joints]
                    del H[i - disconnected_joints]
                    disconnected_joints += 1
            disconnected_joints -= disconnected_ext
            num_joints = A.shape[0]
            num_pipes = A.shape[1]
            disconnected_pipes = 0
            for i in range(num_pipes):
                linked_joints = sum(
                    1 for j in range(num_joints)
                    if A[j, i - disconnected_pipes] in (1, -1)
                )
                if linked_joints == 0:
                    A.col_del(i - disconnected_pipes)
                    del pipe_id[i - disconnected_pipes]
                    del q[i - disconnected_pipes]
                    disconnected_pipes += 1
            num_joints = A.shape[0]
            num_pipes = A.shape[1]
            exist_joints = len(joint_iddi) - disconnected_joints
            Hmin = [15] * num_joints
            Hdes = [30] * num_joints
            n_des = 1.8
            demand = []
            for i in range(exist_joints):
                demand1 = 0
                demand2 = Qn[i] * ((H[i] - Hmin[i]) / (Hdes[i] - Hmin[i])) ** (1 / n_des)
                demand3 = Qn[i]
                demand.append(
                    sp.Piecewise((demand1, H[i] <= Hmin[i]),
                                 (demand2, H[i] < Hdes[i]),
                                 (demand3, H[i] >= Hdes[i]))
                )
            for i in range(exist_joints):
                Qn[i] = demand[i]
            for i in range(exist_joints, num_joints):
                Qn[i] = sp.Piecewise((0, H[i] <= Hmin[i]), (Qn[i], H[i] > Hmin[i]))
            f = np.matmul(A, q) + Qn
            print(f)
            for i in range(num_reservoirs):
                for j in range(num_joints):
                    if joint_id[j] == reservoirs_id[i]:
                        f[j] = H[j] - reservoirs_H[i]
            return f, H, joint_iddi, all_time, joint_w

    def solver_function(self,f, H, joint_iddi, all_time,joint_w):
        h_id = [int(str(x)[1:]) for x in H]
        with open(os.path.join(current_script_dir,"input", self.input_json["eq"]), 'w', encoding='utf-8') as fi:
            for kk in f:
                fi.write(str(kk) + '\n')
            if len(range(max(h_id) + 1)) != len(h_id):
                missing = set(range(max(h_id))) - set(h_id)
                for m in sorted(missing):
                    fi.write(f'H[{m}]\n')
        with open(os.path.join(current_script_dir,"input", self.input_json["eq"]), 'r', encoding='utf-8') as file:
            equation = file.readlines()
        H_i = [f'H{i}' for i in range(150, -1, -1)]
        new_H = [f'H[{j}]' for j in range(150, -1, -1)]
        new_eq = []
        for line in equation:
            tmp = line.strip()
            for old, new in zip(H_i, new_H):
                tmp = tmp.replace(old, new)
            conditions = tmp.split('Piecewise')[-1]
            parts = re.findall(r'\((.*?)\)', conditions)
            if len(parts) == 2:
                coeff = parts[-1].split('*')[0]
                idx = parts[-1].split('[')[1].split(']')[0]
                tmp = tmp.replace(tmp[tmp.find('Piecewise'):],
                                  f'{coeff}*fun2(H[{idx}])')
            elif len(parts) == 3:
                coeff = parts[1].split('*')[0]
                idx = parts[1].split('[')[1].split(']')[0]
                tmp = tmp.replace(tmp[tmp.find('Piecewise'):],
                                  f'{coeff}*fun1(H[{idx}])')
            new_eq.append(tmp + ',\n')
        with open(os.path.join(current_script_dir,"input", self.input_json["configs_solver"]), 'r', encoding='utf-8') as cfg:
            config = cfg.readlines()
        split_idx = next(i + 1 for i, ln in enumerate(config) if ln == '    eqs = [\n')
        merged = config[:split_idx] + ['        ' + l for l in new_eq] + config[split_idx:]
        with open('solver_eqs.py', 'w', encoding='utf-8') as sv:
            sv.writelines(merged)
        import solver_eqs
        solver_eqs.Abs = np.abs
        solve_root = root(
            solver_eqs.eq_solver,
            np.random.randn(max(h_id) + 1),
            method='hybr'
        )
        rand_root = list(solve_root.x[:len(joint_iddi)])
        with open(os.path.join(current_script_dir,"output", self.out_json["节点震后水压"]), 'w', encoding='utf-8') as rf:
            rf.write(','.join(map(str, rand_root)) + '\n')
        with open(os.path.join(current_script_dir,"output", self.out_json["系统修复时间"]), 'w', encoding='utf-8') as tf:
            tf.write(str(all_time) + '\n')
        with open(os.path.join(current_script_dir,"output", self.out_json["节点重要度"]), 'w', encoding='utf-8') as jf:
            for val in joint_w:
                jf.write(f'{val}\n')
    def MainFunction(self):
        f, H, joint_iddi, all_time, joint_w = self.function1()
        self.solver_function(f, H, joint_iddi, all_time, joint_w)
# 使用示例
if __name__ == "__main__":
    # input_json = {
    #     "管道信息": "管道信息.xls",  # 包含管道的相关信息，如管道的起始节点、结束节点、材质、长度、直径等。代码通过xlrd库读取该文件，并提取所需的数据用于后续计算。
    #     "节点信息": "节点信息.xls",  # 包含节点的相关信息，如节点的流量需求、是否为供水点、节点的水头高度等。
    #     "其他信息": "其他信息.csv",  # 包含节点的峰值地面速度（PGV）信息
    #     "eq": "eq.txt",  # 在function函数中，代码将方程组f的内容写入该文件；在solver_function函数中，代码从该文件读取方程组的内容。
    #     "configs_solver": "configs_solver.txt"  # 求解方程组的配置信息
    # }
    #
    # out_json = {
    #     "断点数量及位置": "断点数量及位置.csv",  # 记录每个管道的断点数量和断点位置信息。代码使用csv库将这些信息写入该文件。
    #     "节点震后水压": "节点震后水压.txt",  # 记录节点震后的水压信息，代码将求解得到的节点水压值以逗号分隔的形式写入该文件。
    #     "系统修复时间": "系统修复时间.txt",  # 记录系统的修复时间，代码将计算得到的修复时间写入该文件。
    #     "节点重要度": "节点重要度.txt"  # 记录每个节点的重要度信息，代码将节点的重要度值逐行写入该文件。
    #
    # }

    # 创建实例
    C_GongShui = GongShui("input.json", "output.json")
    C_GongShui.MainFunction()
