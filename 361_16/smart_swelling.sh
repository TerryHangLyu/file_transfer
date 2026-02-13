#!/bin/bash

SECONDS=0

GPU_ID=0
Mainchain_Residues="1S 1R 2R 2CD 2CU 3R 3E"
Cross_Linker_Residue=2C

cat <<'EOF' > opt.mdp
; minim.mdp — energy minimization (coordinates only)
integrator            = steep         ; steepest descent algorithm :contentReference[oaicite:0]{index=0}
emtol                 = 500.0         ; stop when max force < 1.0 kJ/mol/nm
emstep                = 0.05        ; initial step size (nm)
nsteps                = 100000      ; allow more steps if needed

; ───────────── Output control ──────────
nstxout               = 0        ; 2 ps
nstvout               = 0
nstenergy             = 0        ; 2 ps (write energies now)
nstlog                = 1000
nstxout_compressed    = 1000

; Neighbor searching & cutoffs
nstlist               = 1             ; update neighbor list every step :contentReference[oaicite:4]{index=4}
cutoff-scheme         = Verlet        ; use Verlet buffer scheme :contentReference[oaicite:5]{index=5}
ns_type               = grid          ; use grid for neighbor search :contentReference[oaicite:6]{index=6}
rlist                 = 1.2         ; nm
rcoulomb              = 1.2         ; nm
rvdw                  = 1.2         ; nm
rvdw-switch           = 0.9         ; nm (force-switch)
vdw-modifier          = Potential-shift-Verlet
DispCorr              = EnerPres    ; long-range van der Waals corrections

; PBC
pbc                   = xyz           ; periodic in all directions :contentReference[oaicite:16]{index=16}

; Electrostatics
coulombtype           = PME
periodic_molecules    = yes
EOF
cat <<'EOF' > npzt.mdp
; ───────────── Run control ─────────────
integrator            = md
dt                    = 0.002      ; 4 fs
nsteps                = 2500000  ; 400,000,000 steps * 0.004 ps = 1,600,000 ps = 1600 ns


; Output control
nstxout_compressed    = 10000         ; write coordinates every 2 ps
nstvout               = 0        ; write velocities every 2 ps
nstenergy             = 0        ; write energies every 2 ps
nstlog                = 1000        ; write log file every 2 ps

; ───── Neighbor searching & cut-offs ───
cutoff-scheme         = Verlet
nstlist               = 10 
rlist                 = 1.2         ; 推荐与 cutoffs 一致以减少不必要的大邻居列表
rcoulomb              = 1.2
rvdw                  = 1.2
rvdw-switch           = 0.9
vdw-modifier          = Potential-shift-Verlet
DispCorr              = EnerPres

; ───────────── Electrostatics ───────────
coulombtype           = PME

; ───────── Temperature coupling ────────
tcoupl                = V-rescale     ; 可用于短平衡；生产建议换为 Nose-Hoover
tc-grps               = System
tau_t                 = 0.5
ref_t                 = 309.65

; Pressure coupling (use for production: Parrinello-Rahman)
pcoupl                = C-rescale     ; 推薦在短的預平衡中用，生产请改为 Parrinello-Rahman
pcoupltype            = semiisotropic
tau_p                 = 10.0
compressibility       = 0.0 4.5e-5
ref_p                 = 1.0 1.0

; ───────────── Constraints ─────────────
constraints           = h-bonds     ; 与 HMR 一般配合良好（对于 4 fs, 通常可行）
constraint_algorithm  = lincs
lincs_iter            = 1             ; 推荐 2（比 1 更稳，当用 4fs 时建议设为 2）
lincs_order           = 4             ; 推荐 6（若使用双精度并仍有问题可尝试 8）

; ───────── Periodic boundary conditions ─────────
pbc                   = xyz

; ───────── Velocity generation ─────────
gen_vel               = no
; gen_temp              = 309.65
; gen_seed              = -1

periodic_molecules    = yes
EOF
cat <<'EOF' > nvt_annealing.mdp
; -------------- Multi‐cycle Annealing MDP for GROMACS ----------------
title           = 3‐cycle Simulated Annealing between 300K and 600K

; Run control
integrator            = md
dt                    = 0.001          ; ps
nsteps                = 5000000        ; 1500000 × 0.002 ps = 3000 ps = 3 ns

; ───────────── Output control ──────────
nstxout               = 0
nstvout               = 0
nstenergy             = 5000       ; 写能量的频率（按需要）
nstlog                = 5000
nstxout_compressed    = 50000

; Neighbor searching & cutoffs
cutoff-scheme         = Verlet
nstlist               = 10          ; update neighbor list every 10 steps (20 fs)
rlist                 = 1.2         ; nm
rcoulomb              = 1.2         ; nm
rvdw                  = 1.2         ; nm
rvdw-switch           = 0.9         ; nm (force-switch)
vdw-modifier          = Potential-shift-Verlet
DispCorr              = EnerPres    ; long-range van der Waals corrections

; Electrostatics
coulombtype           = PME

; Temperature coupling (V-rescale)
tcoupl                = V-rescale
tc_grps               = System
tau_t                 = 0.5         ; ps :contentReference[oaicite:2]{index=2}
ref_t                 = 300         ; K

; Constraints
constraints           = h-bonds
constraint_algorithm  = lincs
lincs_iter            = 1
lincs_order           = 4

; Periodic boundary conditions
pbc                   = xyz

; Velocity generation
gen_vel               = yes          ; velocities from prior NVT run
gen_temp              = 300          ; Temperature for Maxwell distribution
gen_seed              = -1   

; ==== 多次退火设置 ==== 
annealing       = single
annealing-npoints = 5
annealing-time  = 0      800   1700   2500   5000
annealing-temp  = 300    600    600    300    300

periodic_molecules = yes
EOF
cat <<'EOF' > water_content_parallel_weighted.py
#!/usr/bin/env python3
"""
water_content_parallel_weighted_save_both.py

并行计算“基于每帧全局膜几何中心”的水含量（带权重残基），
并在 --save_last_gro 模式下同时保存两份快照：
 - fullmem: 整个膜（normal+double 列表里的全部原子） + slab 内水分子
 - slabmem: 仅 slab 相交的膜残基（按残基计，保留整个残基） + slab 内水分子

详细用法见脚本内 --help。
"""

import argparse
import math
import numpy as np
from multiprocessing import Pool
import tqdm
import MDAnalysis as mda
from MDAnalysis.lib import distances

# ---------------------------
# Worker globals (initialized per process)
_univ = None
_sel_all_mem = None
_sel_norm = None
_sel_double = None
_sel_water = None
_half_A = None
_use_pbc_z = True
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Parallel water-content (weighted residues) per-frame, save both fullmem+slabmem gro.")
    p.add_argument("--gro", required=True, help="Topology .gro")
    p.add_argument("--xtc", required=True, help="Trajectory .xtc")
    p.add_argument("--normal", nargs="+", required=True,
                   help="正常计数的膜残基名列表 (e.g. 1S1 1R1 ...)")
    p.add_argument("--double", nargs="+", required=True,
                   help="需要乘2计数的膜残基名列表 (e.g. 2R1 2R2 ...)")
    p.add_argument("--water", default="SOL", help="水残基名 (默认 SOL)")
    p.add_argument("--half_nm", type=float, default=3.5,
                   help="slab 半厚度 (nm), 默认 3.5 (即上下各 3.5 nm)")
    p.add_argument("--stride", type=int, default=1, help="采样步长 (默认 1)")
    p.add_argument("--nprocs", type=int, default=4, help="并行进程数 (默认 4)")
    p.add_argument("--out_png", default=None, help="若指定，保存平均含水量随时间图 (PNG)")
    p.add_argument("--save_last_gro", action="store_true", help="是否同时保存两份快照：fullmem 和 slabmem")
    p.add_argument("--no_pbc_z", action="store_true", help="在 z 方向不使用 PBC 最小像 (若坐标 unwrapped)")
    p.add_argument("--last_n", type=int, default=500, help="用于平均的最后 N 个采样帧 (默认 500)")
    return p.parse_args()

def min_image_dz(z_atoms, z_center, Lz):
    """minimum-image dz (Å)"""
    if Lz == 0 or math.isclose(Lz, 0.0):
        return z_atoms - z_center
    return (z_atoms - z_center + 0.5 * Lz) % Lz - 0.5 * Lz

def init_worker(gro, xtc, sel_all_mem, sel_norm, sel_double, sel_water, half_A, use_pbc_z):
    """Initializer called in each worker process: load Universe and store selectors/params"""
    global _univ, _sel_all_mem, _sel_norm, _sel_double, _sel_water, _half_A, _use_pbc_z
    _univ = mda.Universe(gro, xtc)
    _sel_all_mem = sel_all_mem
    _sel_norm = sel_norm
    _sel_double = sel_double
    _sel_water = sel_water
    _half_A = half_A
    _use_pbc_z = use_pbc_z

def process_frame(frame_idx):
    """
    Worker function: compute for single frame index.
    Returns tuple: (frame_idx, time_ps, N_norm, N_double, N_water)
    """
    global _univ, _sel_all_mem, _sel_norm, _sel_double, _sel_water, _half_A, _use_pbc_z
    u = _univ
    u.trajectory[frame_idx]
    ts = u.trajectory.ts
    Lz = float(u.dimensions[2])  # Å

    mem_all = u.select_atoms(_sel_all_mem)
    if len(mem_all) == 0:
        return (frame_idx, float(ts.time), 0, 0, 0)

    center = mem_all.positions.mean(axis=0)  # Å
    center_z = float(center[2])

    n_norm = 0
    for res in u.select_atoms(_sel_norm).residues:
        if len(res.atoms) == 0: continue
        z_atoms = res.atoms.positions[:,2]
        dz = z_atoms - center_z if not _use_pbc_z else min_image_dz(z_atoms, center_z, Lz)
        if np.any(np.abs(dz) <= _half_A): n_norm += 1

    n_double = 0
    for res in u.select_atoms(_sel_double).residues:
        if len(res.atoms) == 0: continue
        z_atoms = res.atoms.positions[:,2]
        dz = z_atoms - center_z if not _use_pbc_z else min_image_dz(z_atoms, center_z, Lz)
        if np.any(np.abs(dz) <= _half_A): n_double += 1

    n_water = 0
    for res in u.select_atoms(_sel_water).residues:
        if len(res.atoms) == 0: continue
        z_atoms = res.atoms.positions[:,2]
        dz = z_atoms - center_z if not _use_pbc_z else min_image_dz(z_atoms, center_z, Lz)
        if np.any(np.abs(dz) <= _half_A): n_water += 1

    return (frame_idx, float(ts.time), n_norm, n_double, n_water)

def main():
    args = parse_args()

    half_A = args.half_nm * 10.0  # nm -> Å
    use_pbc_z = not args.no_pbc_z

    # build selector strings; if overlap, double takes precedence
    normal_names = list(args.normal)
    double_names = list(args.double)
    normal_names = [n for n in normal_names if n not in double_names]

    sel_all_mem = "resname " + " ".join(double_names + normal_names) if (double_names or normal_names) else "resname NONE"
    sel_norm = "resname " + " ".join(normal_names) if normal_names else "resname NONE"
    sel_double = "resname " + " ".join(double_names) if double_names else "resname NONE"
    sel_water = f"resname {args.water}"

    # load one Universe in main to get frame count
    u_main = mda.Universe(args.gro, args.xtc)
    n_frames = len(u_main.trajectory)
    frame_indices = list(range(0, n_frames, args.stride))
    if len(frame_indices) == 0:
        raise SystemExit("No frames to process (check stride/trajectory).")

    print(f"Frames in traj: {n_frames}, sampled frames: {len(frame_indices)}, stride={args.stride}")
    print(f"Normal resnames: {normal_names}")
    print(f"Double resnames: {double_names}")
    print(f"Using PBC in z: {use_pbc_z}, half slab = {args.half_nm} nm ({half_A} Å)")
    print(f"Workers: {args.nprocs}")

    init_args = (args.gro, args.xtc, sel_all_mem, sel_norm, sel_double, sel_water, half_A, use_pbc_z)

    # parallel processing
    results = []
    with Pool(processes=args.nprocs, initializer=init_worker, initargs=init_args) as pool:
        it = pool.imap_unordered(process_frame, frame_indices, chunksize=1)
        for out in tqdm.tqdm(it, total=len(frame_indices), desc="Processing frames"):
            results.append(out)

    results.sort(key=lambda x: x[0])
    frame_idxs, times, n_norms, n_doubles, n_waters = zip(*results)
    frame_idxs = np.array(frame_idxs, dtype=int)
    times = np.array(times, dtype=float)
    n_norms = np.array(n_norms, dtype=int)
    n_doubles = np.array(n_doubles, dtype=int)
    n_waters = np.array(n_waters, dtype=int)

    weighted_mem = n_norms + 2 * n_doubles
    with np.errstate(invalid='ignore', divide='ignore'):
        ratios = n_waters / weighted_mem.astype(float)

    # print average over last N sampled frames
    LAST_N = args.last_n
    total_sampled = len(ratios)
    n_take = min(LAST_N, total_sampled)
    if n_take == 0:
        print("No sampled frames to compute averages.")
    else:
        last_slice = ratios[-n_take:]
        valid_mask = ~np.isnan(last_slice)
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            print(f"Average water content (last {n_take} sampled frames): no valid frames (all NaN)")
        else:
            avg_recent = float(np.nanmean(last_slice[valid_mask]))
            print(f"Average water content (last {n_take} sampled frames): {avg_recent:.6f} (computed from {n_valid} valid frames)")

    # optional plotting
    if args.out_png:
        try:
            import matplotlib.pyplot as plt
            mask = ~np.isnan(ratios)
            plt.figure()
            plt.plot(times[mask], ratios[mask], marker='.', linestyle='-')
            plt.xlabel("Time (ps)")
            plt.ylabel("N_water / (N_norm + 2*N_double)")
            plt.title("Water content (per-frame, weighted membrane count)")
            plt.tight_layout()
            plt.savefig(args.out_png, dpi=300)
            print(f"Saved plot to {args.out_png}")
        except Exception as e:
            print("Plotting failed:", e)

    # save last sampled frame: BOTH full-membrane version and slab-only membrane version
    if args.save_last_gro:
        last_frame_idx = frame_indices[-1]
        u_main.trajectory[last_frame_idx]
        ts = u_main.trajectory.ts
        Lz = float(u_main.dimensions[2])
        # recompute center
        mem_all = u_main.select_atoms(sel_all_mem)
        if len(mem_all) == 0:
            print("No membrane atoms found; skipping save_last_gro.")
        else:
            center = mem_all.positions.mean(axis=0)
            center_z = float(center[2])

            # ---------- Full-membrane + slab-water (方案 A) ----------
            halfA = half_A
            keep_idx_full = set()
            # add ALL membrane atoms (full membrane)
            for atom in mem_all:
                keep_idx_full.add(int(atom.ix))
            # add water residues that intersect slab
            kept_water_mols_full = 0
            for res in u_main.select_atoms(sel_water).residues:
                if len(res.atoms) == 0: continue
                z_atoms = res.atoms.positions[:, 2]
                dz = z_atoms - center_z if args.no_pbc_z else min_image_dz(z_atoms, center_z, Lz)
                if np.any(np.abs(dz) <= halfA):
                    for at in res.atoms:
                        keep_idx_full.add(int(at.ix))
                    kept_water_mols_full += 1

            kept_sorted_full = sorted(keep_idx_full)
            sel_full = u_main.atoms[kept_sorted_full]
            out_full = f"filtered_lastframe_fullmem_idx{last_frame_idx}_time{int(ts.time)}ps.gro"
            with mda.Writer(out_full, n_atoms=sel_full.n_atoms, format="GRO") as W:
                W.write(sel_full)
            print(f"Wrote full-membrane snapshot: {out_full}  (kept atoms: {sel_full.n_atoms}, kept water molecules approx: {kept_water_mols_full})")

            # ---------- Slab-only membrane residues + slab-water (方案 B) ----------
            keep_idx_slab = set()
            kept_water_mols_slab = 0
            # keep membrane residues that intersect slab (per-residue)
            for res in u_main.select_atoms(sel_all_mem).residues:
                if len(res.atoms) == 0: continue
                z_atoms = res.atoms.positions[:, 2]
                dz = z_atoms - center_z if args.no_pbc_z else min_image_dz(z_atoms, center_z, Lz)
                if np.any(np.abs(dz) <= halfA):
                    for at in res.atoms:
                        keep_idx_slab.add(int(at.ix))
            # add water residues that intersect slab
            for res in u_main.select_atoms(sel_water).residues:
                if len(res.atoms) == 0: continue
                z_atoms = res.atoms.positions[:, 2]
                dz = z_atoms - center_z if args.no_pbc_z else min_image_dz(z_atoms, center_z, Lz)
                if np.any(np.abs(dz) <= halfA):
                    for at in res.atoms:
                        keep_idx_slab.add(int(at.ix))
                    kept_water_mols_slab += 1

            kept_sorted_slab = sorted(keep_idx_slab)
            sel_slab = u_main.atoms[kept_sorted_slab]
            out_slab = f"filtered_lastframe_slabmem_idx{last_frame_idx}_time{int(ts.time)}ps.gro"
            with mda.Writer(out_slab, n_atoms=sel_slab.n_atoms, format="GRO") as W:
                W.write(sel_slab)
            print(f"Wrote slab-membrane snapshot: {out_slab}  (kept atoms: {sel_slab.n_atoms}, kept water molecules approx: {kept_water_mols_slab})")

if __name__ == "__main__":
    main()
EOF
cat <<'EOF' > calc_slice.sh
#!/bin/bash

# ##############################
# 功能：从GRO文件提取z轴盒子大小，计算最优slice数（使z间隔接近0.05 nm）
# 使用方法：1. 将脚本保存为 calc_slice.sh；2. 赋予执行权限：chmod +x calc_slice.sh；3. 运行：./calc_slice.sh 你的文件.gro
# 输出：z轴盒子长度、最优slice数、实际间隔
# ##############################

# 检查是否传入GRO文件路径
if [ $# -ne 1 ]; then
    echo "错误：请传入GRO文件路径！"
    echo "示例：./calc_slice.sh system.gro"
    exit 1
fi

GRO_FILE="$1"

# 提取GRO文件中盒子的z轴长度（GRO文件最后一行为盒子x/y/z维度，单位nm）
# awk 'END{print $3}' 表示读取文件最后一行的第3列（z轴数据）
BOX_Z=$(awk 'END{print $3}' "$GRO_FILE")

# 检查提取结果是否为有效数字
if ! [[ "$BOX_Z" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "错误：无法从GRO文件提取有效的z轴盒子长度！"
    exit 1
fi

# 目标间隔（0.05 nm），计算理论slice数（盒子z长度 / 0.05），四舍五入为整数
# bc用于浮点数计算，printf "%.0f" 实现四舍五入
TARGET_SPACING="0.05"
THEORETICAL_SLICE=$(echo "scale=10; $BOX_Z / $TARGET_SPACING" | bc)
OPTIMAL_SLICE=$(printf "%.0f" "$THEORETICAL_SLICE")

# 计算实际间隔（盒子z长度 / 最优slice数），保留4位小数
ACTUAL_SPACING=$(echo "scale=4; $BOX_Z / $OPTIMAL_SLICE" | bc)

# 输出结果
echo "==================== 计算结果 ===================="
echo "GRO文件：$GRO_FILE"
echo "z轴盒子长度：$BOX_Z nm"
echo "目标间隔：$TARGET_SPACING nm"
echo "最优slice数：$OPTIMAL_SLICE（使实际间隔最接近目标）"
echo "实际间隔：$ACTUAL_SPACING nm"
echo "=================================================="
EOF

i=0
read Lx Ly Lz <<< "$(tail -n1 start_$i.gro)"
gmx editconf -f start_$i.gro -o start_$i.gro -box ${Lx} ${Ly} 30 -c
gmx solvate -cp start_$i.gro -cs ./10Q5.ff/spc216.gro -p start_$i.top -o start_$i.gro
sed -i '/#include "\.\/10Q5\.ff\/forcefield\.itp"/a\#include "./10Q5.ff/opc3.itp"' ./start_0.top

i=0  # 循环计数器起始值
while true; do

echo "===== loop No. $(($i+1)) ====="

echo "=== making and running No. opt_$i.tpr file ==="
gmx grompp -f opt.mdp -p start_$i.top -c start_$i.gro -o opt_$i.tpr -maxwarn 1
count=0; while true; do count=$((count+1)); echo -e "\n===== 第 $count 次尝试执行gmx mdrun ====="; gmx mdrun -ntmpi 1 -ntomp 22 -deffnm opt_$i -nb gpu -pme cpu -bonded cpu -v -gpu_id $GPU_ID; exit_code=$?; if [ $exit_code -eq 0 ]; then echo -e "\n===== 成功！第 $count 次尝试执行完成 ====="; break; else echo -e "\n===== 第 $count 次尝试失败（退出码：$exit_code），5秒后重试 ====="; sleep 5; fi; done

echo "=== making and running No. nvt_annealing_$i.tpr file ==="
gmx grompp -f nvt_annealing.mdp -p start_$i.top -c opt_$i.gro -o nvt_annealing_$i.tpr
count=0; while true; do count=$((count+1)); echo -e "\n===== 第 $count 次尝试执行gmx mdrun ====="; gmx mdrun -ntmpi 1 -ntomp 22 -deffnm nvt_annealing_$i -nb gpu -pme gpu -bonded gpu -v -gpu_id $GPU_ID -cpi nvt_annealing_$i.cpt; exit_code=$?; if [ $exit_code -eq 0 ]; then echo -e "\n===== 成功！第 $count 次尝试执行完成 ====="; break; else echo -e "\n===== 第 $count 次尝试失败（退出码：$exit_code），5秒后重试 ====="; sleep 5; fi; done

echo "=== making and running No. npzt_$i.tpr file ==="
gmx grompp -f npzt.mdp -p start_$i.top -c nvt_annealing_$i.gro -o npzt_$i.tpr
count=0; while true; do count=$((count+1)); echo -e "\n===== 第 $count 次尝试执行gmx mdrun ====="; gmx mdrun -ntmpi 1 -ntomp 22 -deffnm npzt_$i -nb gpu -pme gpu -bonded gpu -v -gpu_id $GPU_ID -cpi npzt_$i.cpt; exit_code=$?; if [ $exit_code -eq 0 ]; then echo -e "\n===== 成功！第 $count 次尝试执行完成 ====="; break; else echo -e "\n===== 第 $count 次尝试失败（退出码：$exit_code），5秒后重试 ====="; sleep 5; fi; done

read Lx Ly Lz <<< "$(tail -n1 npzt_$i.gro)"

if (( $(echo "$Lz < 30" | bc -l) )); then
j=$(($i+1))	 
gmx editconf -f npzt_${i}.gro -o ${j}.gro -box ${Lx} ${Ly} 30 -c
cp start_$i.top $j.top
gmx solvate -cp ${j}.gro -cs ./10Q5.ff/spc216.gro -p $j.top -o start_$j.gro
mv $j.top start_$j.top
rm \#$j.top.1#
rm $j.gro
else
j=$(($i+1))	 
cp start_$i.top start_$j.top
cp npzt_$i.gro start_$j.gro
fi 

Water_Content_EQ=$(python water_content_parallel_weighted.py --gro start_${i}.gro --xtc npzt_${i}.xtc --normal ${Mainchain_Residues} --double ${Cross_Linker_Residue}\
 --water SOL --half_nm 2.0 --stride 1 --nprocs 8 --out_png water.png --save_last_gro --last_n 10 2>&1 | grep "Average water content" | awk '{print $8}')
mv water.png water${i}.png
echo "$(($i+1)),$Water_Content_EQ" >> Water_Content.csv

gro_file="npzt_$i.gro"
Num_of_Slices=$(bash calc_slice.sh $gro_file | grep "最优slice数" | awk -F '：|（' '{print $2}' | xargs)
gmx select -f npzt_$i.gro -s npzt_$i.gro -select 'not (resname CL or resname SOL)' -on Polymer_$i.ndx
gmx density -f npzt_$i.xtc -s npzt_$i.tpr -n Polymer_$i.ndx -o density_$i.xvg -d Z -sl ${Num_of_Slices} -dens mass -b 4000 -e 6000
cat <<EOF > density_fitting.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitting script for membrane density distribution (Formula 1 with erf function)
Auto-adapt initial values + correct unit conversion (kg/m³ → g/cm³)
Input: density.xvg (GROMACS output)
Output: Fitting parameters (g/cm³), characteristic lengths, and plots
Dependencies: numpy, scipy, matplotlib (install: pip install numpy scipy matplotlib)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib as mpl

# -------------------------- 1. 全局字体设置（消除警告）--------------------------
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 10

# -------------------------- 2. Formula 1 (core fitting function) --------------------------
def formula1(z, rho0, sigma, z1, z2):
    """
    Formula 1: ρ(z) = 0.5*ρ₀[erf((z-z1)/(σ√2)) - erf((z-z2)/(σ√2))]
    Parameters:
        z: z-coordinate (nm)
        rho0: Bulk density of membrane (g/cm³)
        sigma: Standard deviation of interface layer (nm)
        z1: Left Gibbs dividing surface (nm)
        z2: Right Gibbs dividing surface (nm)
    Return: Fitted density at z (g/cm³)
    """
    term1 = erf((z - z1) / (sigma * np.sqrt(2)))
    term2 = erf((z - z2) / (sigma * np.sqrt(2)))
    return 0.5 * rho0 * (term1 - term2)

# -------------------------- 3. Read GROMACS density.xvg data (加单位换算) --------------------------
def read_gromacs_density(file_path):
    """
    Read density.xvg (skip comments), extract z-coordinate and density data
    Convert GROMACS default unit (kg/m³) → g/cm³ (1 kg/m³ = 0.001 g/cm³)
    Return: z_data (array), density_data (array, g/cm³)
    """
    z_data = []
    density_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip comment lines (start with # or @)
            if line.startswith(('#', '@')):
                continue
            # Split line into values (handle multiple spaces/tabs)
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    z = float(parts[0])
                    density_kg_m3 = float(parts[1])  # GROMACS输出的原始值（kg/m³）
                    density_g_cm3 = density_kg_m3 * 0.001  # 转换为g/cm³
                    z_data.append(z)
                    density_data.append(density_g_cm3)
                except ValueError:
                    continue  # Skip non-numeric lines
    # Convert to numpy array for fitting
    return np.array(z_data), np.array(density_data)

# -------------------------- 4. Main fitting process --------------------------
if __name__ == "__main__":
    # Read density data (auto-adapt to any z-axis length + correct unit)
    density_file = "density_$i.xvg"
    z_data, density_data = read_gromacs_density(density_file)
    
    # Basic data info (auto-calculated based on real data)
    z_min = z_data.min()
    z_max = z_data.max()
    rho_max = density_data.max()
    num_points = len(z_data)
    
    # Print basic info (with correct unit)
    print("=== Membrane Data Basic Info ===")
    print(f"Number of sampling points: {num_points}")
    print(f"z-coordinate range: {z_min:.3f} ~ {z_max:.3f} nm")
    print(f"Density range (g/cm³): {density_data.min():.4f} ~ {rho_max:.4f}")
    print(f"Max density (g/cm³): {rho_max:.4f}")
    
    # -------------------------- 自动计算初始值p0 --------------------------
    # p0 = [rho0_init, sigma_init, z1_init, z2_init]
    rho0_init = rho_max  # rho0初始值=密度最大值（通用）
    sigma_init = 1.0     # sigma初始值=1.0 nm（界面层厚度通用初始值）
    z1_init = z_min + (z_max - z_min) * 0.2  # 左分界面=z轴总长的20%位置
    z2_init = z_min + (z_max - z_min) * 0.8  # 右分界面=z轴总长的80%位置
    p0 = [rho0_init, sigma_init, z1_init, z2_init]
    print(f"\nAuto-calculated initial values (p0): {p0}")
    
    # Perform curve fitting (auto-adapt to p0)
    try:
        popt, pcov = curve_fit(
            f=formula1,
            xdata=z_data,
            ydata=density_data,
            p0=p0,
            bounds=([0.0, 0.1, z_min, z1_init], [rho_max*1.2, 5.0, z2_init, z_max]),  # 边界自动适配
            maxfev=20000
        )
        rho0_fit, sigma_fit, z1_fit, z2_fit = popt
    except Exception as e:
        print(f"Fitting failed! Error: {e}")
        exit(1)
    
    # Calculate fitting goodness (R²) and characteristic lengths
    rho_fit = formula1(z_data, *popt)
    ss_res = np.sum((density_data - rho_fit) ** 2)
    ss_tot = np.sum((density_data - density_data.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    L_Gibbs = z2_fit - z1_fit  # Apparent thickness (nm)
    L_bulk = (z2_fit - 3*sigma_fit) - (z1_fit + 3*sigma_fit)  # Bulk thickness (nm)
    
    # Print fitting results (correct unit)
    print("\n=== Membrane Fitting Results (Formula 1) ===")
    print(f"Bulk density (ρ₀) = {rho0_fit:.4f} g/cm³")
    print(f"Interface σ = {sigma_fit:.4f} nm")
    print(f"Left dividing surface (z₁) = {z1_fit:.4f} nm")
    print(f"Right dividing surface (z₂) = {z2_fit:.4f} nm")
    print(f"Fitting R² = {r2:.4f} (≥0.9 = excellent)")
    print(f"Apparent thickness (L_Gibbs) = {L_Gibbs:.4f} nm")
    print(f"Bulk thickness (L_bulk) = {L_bulk:.4f} nm")
    
    # Plotting (auto-adapt to z-axis range)
    plt.figure(figsize=(14, 7))
    # Plot raw data (correct unit)
    plt.scatter(z_data, density_data, s=5, c='#2f7ed8', alpha=0.8, label='Raw membrane density')
    # Plot fitted curve
    z_smooth = np.linspace(z_min, z_max, 1000)
    rho_smooth = formula1(z_smooth, *popt)
    plt.plot(z_smooth, rho_smooth, c='#e95f21', linewidth=2.5, label='Fitted curve (Formula 1)')
    # Mark dividing surfaces and membrane region
    plt.axvline(x=z1_fit, c='#2abc3d', ls='--', lw=2, label=f'Left surface z₁ = {z1_fit:.4f} nm')
    plt.axvline(x=z2_fit, c='#d35400', ls='--', lw=2, label=f'Right surface z₂ = {z2_fit:.4f} nm')
    plt.axvspan(z1_fit, z2_fit, alpha=0.1, color='gray', label=f'Apparent region (L_Gibbs = {L_Gibbs:.4f} nm)')
    # Plot settings (correct unit in label)
    plt.xlabel('z-coordinate (nm)', fontsize=12)
    plt.ylabel('Membrane density (g/cm³)', fontsize=12)
    plt.title(f'Membrane Density Distribution Fitting (R² = {r2:.4f})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(alpha=0.3)
    # Auto-set xlim to full z-range
    plt.xlim(z_min, z_max)
    
    # Save plots
    plt.savefig('membrane_fitting_result.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig('membrane_fitting_result.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved: membrane_fitting_result.png / membrane_fitting_result.pdf")
    print("\n⚠️  Important Note: You used single-frame data (may have random fluctuations).")
    print("Suggestion: Use multi-frame averaged density (50-100 ns equilibrium trajectory) for more reliable results.")
EOF
Membrane_Density=$(python density_fitting.py | grep "Bulk density (ρ₀) =" | awk -F '=' '{print $2}' | awk '{print $1}')
mv membrane_fitting_result.png membrane_fitting_result_$i.png
echo "$(($i+1)),$Membrane_Density" >> Density.csv

    # ========== 修正版：密度收敛检查逻辑 ==========
    # 步骤1：检查CSV文件行数是否≥10（不足10行则继续循环）
    csv_lines=$(wc -l < Density.csv)
    if [ "$csv_lines" -ge 10 ]; then
        # 步骤2：提取最后10行的第二列（膜密度值），严格过滤合法浮点数
        last_10_values=$(tail -n10 Density.csv | awk -F ',' '{
            # 仅保留合法浮点数（匹配 0.xxx、1.xxx、.xxx 等格式，转换为 0.xxx）
            if ($2 ~ /^[0-9]*\.?[0-9]+$/) {
                # 补全前导0（比如 .6292 → 0.6292）
                if ($2 ~ /^\./) print "0"$2;
                else print $2;
            }
        }' | tr '\n' ' ')
        
        # 检查提取到的有效数值是否≥10（避免有拟合失败的行）
        value_count=$(echo $last_10_values | wc -w)
        if [ "$value_count" -ge 10 ]; then
            # 步骤3：计算最后10个值的平均值（优化bc表达式，避免语法错误）
            sum=0.0
            for val in $last_10_values; do
                # 用bc安全计算求和，强制保留小数位
                sum=$(echo "scale=6; $sum + $val" | bc -l)
            done
            # 计算平均值（scale=4保留4位小数），补全前导0
            avg=$(echo "scale=4; $sum / 10" | bc -l)
            # 格式化平均值，确保有前导0（如 .6292 → 0.6292）
            avg_formatted=$(printf "%.4f" $avg)

            # 步骤4：检查每个值与平均值的绝对差是否≤0.005
            converge_flag=1  # 1=收敛，0=不收敛
            for val in $last_10_values; do
                # 计算绝对差：|val - avg|，用bc安全处理
                diff=$(echo "scale=6; $val - $avg" | bc -l)
                abs_diff=$(echo "scale=6; if ($diff < 0) -$diff else $diff" | bc -l)
                # 检查绝对差是否>0.005（bc比较浮点数）
                if (( $(echo "$abs_diff > 0.005" | bc -l) )); then
                    converge_flag=0
                    break  # 有一个值不满足，直接退出检查
                fi
            done

            # 步骤5：如果收敛，输出提示并终止循环
            if [ "$converge_flag" -eq 1 ]; then
                echo "======================================"
                echo "膜密度已收敛！最后10个密度值：$last_10_values"
                echo "平均值：$avg_formatted，所有值与平均值的绝对差≤0.005"
                echo "循环终止，共生成 $((i+1)) 个膜结构。"
                echo "======================================"
                break
            fi
        fi
    fi

# 自增
((i++))

# 可选：防止死循环（例如超过100次强制停止）
if [ "$i" -gt 100 ]; then
    echo "循环次数超过 100，强制退出。"
    break
fi

done

echo "完成循环退火溶胀时，脚本执行累计耗时：${SECONDS} 秒"
