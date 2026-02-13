#!/usr/bin/env bash
# launch_all.sh -- 为 smart_swelling.sh 批量替换并插入 taskset（支持 inline retry loops）
# 说明：
#  - 替换 GPU_ID（按 GPUS 数组）
#  - 替换 --nprocs / --nprocs= / -ntomp 为 CORES_PER_JOB
#  - 在任意出现的 gmx mdrun 前插入 taskset -c <core_range>（包括包含在单行 retry loop 内的情况）
#  - 将 if [ "$i" -gt <num> ] 替换为 if [ "$i" -gt 200 ]
#  - 备份原文件为 smart_swelling.sh.bak.TIMESTAMP
#  - 启动每个子目录下的 smart_swelling.sh（nohup ... &）
#
# 请根据需要修改下面的 DIRS / GPUS / CORES_PER_JOB
DIRS=(181_8 181_16 271_8 271_16 361_8 361_16)
GPUS=(2 3 4 5 6 7)
CORES_PER_JOB=12

set -euo pipefail

timestamp() { date +%Y%m%d%H%M%S; }

# ========== sanity checks ==========
if [[ "${#DIRS[@]}" -ne "${#GPUS[@]}" ]]; then
    echo "ERROR: DIRS 与 GPUS 数量不匹配 (DIRS=${#DIRS[@]} GPUS=${#GPUS[@]})"
    exit 1
fi

declare -A _seen_gpu=()
for g in "${GPUS[@]}"; do
    if ! [[ "$g" =~ ^[0-9]+$ ]]; then
        echo "ERROR: GPU ID '$g' 不是整数。"
        exit 1
    fi
    if (( g < 0 || g > 7 )); then
        echo "ERROR: GPU ID '$g' 不在 0..7 范围。"
        exit 1
    fi
    if [[ -n "${_seen_gpu[$g]:-}" ]]; then
        echo "ERROR: GPU ID '$g' 重复，请保证唯一。"
        exit 1
    fi
    _seen_gpu[$g]=1
done

echo "启动 launch_all.sh"
echo "  DIRS: ${DIRS[*]}"
echo "  GPUS: ${GPUS[*]}"
echo "  CORES_PER_JOB: ${CORES_PER_JOB}"
echo ""

# 计算 core range （按照你的机器布局：CPU0 物理核 0..47, CPU1 物理核 48..95）
get_core_range() {
    local cpu_id="$1"
    local blk="$2"
    if [[ "$cpu_id" -eq 0 ]]; then
        local start=$(( blk * CORES_PER_JOB ))
    else
        local start=$(( 48 + blk * CORES_PER_JOB ))
    fi
    local end=$(( start + CORES_PER_JOB - 1 ))
    echo "${start}-${end}"
}

# main loop
for idx in "${!DIRS[@]}"; do
    dir="${DIRS[$idx]}"
    gpu="${GPUS[$idx]}"

    echo "-------------------------------------------------"
    echo "处理 #${idx} -> ${dir} (GPU=${gpu})"

    if [[ ! -d "$dir" ]]; then
        echo "  警告：目录 '$dir' 不存在，跳过。"
        continue
    fi

    shfile="$dir/smart_swelling.sh"
    if [[ ! -f "$shfile" ]]; then
        echo "  警告：文件 '$shfile' 不存在，跳过。"
        continue
    fi

    bak="${shfile}.bak.$(timestamp)"
    cp -a "$shfile" "$bak"
    echo "  已备份为: $bak"

    blk_index=$(( idx / 2 ))               # 0,0,1,1,2,2 ...
    cpu_id=$(( idx % 2 == 0 ? 0 : 1 ))     # 偶数 idx -> CPU0, 奇数 idx -> CPU1
    core_range="$(get_core_range "$cpu_id" "$blk_index")"
    core_start="${core_range%%-*}"
    echo "  分配 core_range=${core_range} (cpu_id=${cpu_id}, block=${blk_index}), core_start=${core_start}"

    tmpfile="$(mktemp)"
    while IFS= read -r line || [[ -n "$line" ]]; do
        # 1) 替换 GPU_ID 行
        if [[ "$line" =~ ^[[:space:]]*GPU_ID[[:space:]]*= ]]; then
            echo "GPU_ID=${gpu}" >> "$tmpfile"
            continue
        fi

        # 2) 将 if [ "$i" -gt <num> ]（任何数字）替换为 200
        if echo "$line" | grep -qE 'if[[:space:]]+\[[[:space:]]*"[[:space:]]*\$i[[:space:]]*"[[:space:]]*-gt[[:space:]]*[0-9]+'; then
            echo 'if [ "$i" -gt 200 ]; then' >> "$tmpfile"
            continue
        fi

        modline="$line"

        # 3) 将 --nprocs N 或 --nprocs=N 替换为 CORES_PER_JOB（支持等号与空格）
        if echo "$modline" | grep -qE -- "--nprocs([=[:space:]]*)[0-9]+"; then
            modline="$(echo "$modline" | sed -E "s/--nprocs([=[:space:]]*)[0-9]+/--nprocs\\1${CORES_PER_JOB}/g")"
        fi

        # 4) 替换 -ntomp <num> 或 -ntomp<num> 为 -ntomp ${CORES_PER_JOB}
        if echo "$modline" | grep -qE "(-ntomp)[[:space:]]*[0-9]+"; then
            modline="$(echo "$modline" | sed -E "s/(-ntomp)[[:space:]]*[0-9]+/\\1 ${CORES_PER_JOB}/g")"
        fi
        # 若存在 -ntomp紧挨数字（如 -ntomp22），也处理
        if echo "$modline" | grep -qE "-ntomp[0-9]+"; then
            modline="$(echo "$modline" | sed -E "s/-ntomp([0-9]+)/-ntomp ${CORES_PER_JOB}/g")"
        fi

        # 5) 对任意位置出现的 gmx mdrun 插入 taskset 前缀（仅当该行中尚未含 taskset/numactl）
        if echo "$modline" | grep -qE "gmx[[:space:]]+mdrun"; then
            if ! echo "$modline" | grep -qE "(taskset[[:space:]]+-c|numactl)"; then
                # 在所有 gmx mdrun 的出现处插入 taskset -c <core_range>
                # 使用替换：把第一个 "gmx mdrun" 替换为 "taskset -c <core_range> gmx mdrun"
                modline="$(echo "$modline" | sed -E "s/gmx[[:space:]]+mdrun/taskset -c ${core_range} gmx mdrun/1")"
            fi
        fi

        # 6) 写回处理后的行
        echo "$modline" >> "$tmpfile"
    done < "$shfile"

    # 覆盖回去并设为可执行
    mv "$tmpfile" "$shfile"
    chmod +x "$shfile"
    echo "  已更新并设为可执行: $shfile"

    # 启动脚本（后台）
    (
        cd "$dir" || exit 1
        echo "  在 ${dir} 中以 nohup 后台启动 smart_swelling.sh ..."
        nohup bash smart_swelling.sh > smart_swelling.out 2>&1 &
        sleep 0.5
        pids=$(pgrep -f "bash .*smart_swelling.sh" | tr '\n' ' ' || true)
        echo "  启动后 pids: $pids"
    )

    echo "  处理完成: $dir"
    echo ""
done

echo "全部处理完成。请检查各子目录下的 smart_swelling.out 与 .bak 文件以确认修改与运行状态。"
echo ""
echo "建议检查："
echo "  - /sys/devices/system/cpu/cpu*/topology/thread_siblings_list  以确认 logical<->physical 映射"
echo "  - 各子目录下的 smart_swelling.out 日志，确认 gmx mdrun 已含 taskset 前缀（或用 ps/taskset -p <pid> 验证）"
