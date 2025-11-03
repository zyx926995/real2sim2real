import os
import sys
import time
import platform
import argparse
import mujoco
import mujoco.viewer
import numpy as np

# 添加DISCOVERSE到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'DISCOVERSE'))

if __name__ == "__main__":  
    """
    XBot-Arm 机器人MuJoCo仿真测试程序
    
    该程序用于测试XBot-Arm机器人的MJCF模型能否在DISCOVERSE环境中正常加载和运行。
    """

    parser = argparse.ArgumentParser(
        description="XBot-Arm 机器人MuJoCo仿真测试程序\n"
                    "用法示例：\n"
                    "  python test_xbot_arm_mjcf.py [--mjcf]\n"
                    "参数说明：\n"
                    "  mjcf  (可选) 指定MJCF模型文件路径，若不指定则使用默认模型。"
    )
    parser.add_argument(
        "-m",
        "--mjcf",
        type=str,
        default=None,
        help="输入MJCF文件的路径（可选）。如未指定，则使用默认的xbot_arm_floor.xml。"
    )
    args = parser.parse_args()
    mjcf_path = args.mjcf

    # 定义默认的MJCF文件路径
    if mjcf_path is None:
        mjcf_path = os.path.join(os.path.dirname(__file__), "DISCOVERSE", "models", "mjcf", "xbot_arm", "xbot_arm_floor.xml")
    
    # 检查MJCF文件是否存在
    if not os.path.exists(mjcf_path):
        print(f"错误：MJCF文件不存在: {mjcf_path}")
        print("请检查文件路径是否正确。")
        sys.exit(1)
    
    print(f"使用MJCF文件: {mjcf_path}")

    # 尝试加载MJCF模型
    try:
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        print("MJCF模型加载成功！")
        
        # 打印模型信息
        print(f"关节数量: {mj_model.njnt}")
        print(f"连杆数量: {mj_model.nbody}")
        print("关节名称:")
        for i in range(mj_model.njnt):
            print(f"  - {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)}")
        
        # 创建数据实例
        mj_data = mujoco.MjData(mj_model)
        
        # 检查是否在macOS上运行并给出适当的提示
        if platform.system() == "Darwin":
            print("警告：在macOS上使用MuJoCo查看器可能会遇到性能问题。")
            print("如果遇到问题，请尝试在Linux或Windows上运行。")
        
        print("\n正在启动MuJoCo查看器...")
        print("提示：")
        print("  - 按ESC键退出查看器")
        print("  - 按住右键拖动可旋转视角")
        print("  - 按住中键拖动可平移视角")
        print("  - 滚动滚轮可缩放视角")
        
        # 启动MuJoCo查看器
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            # 设置初始姿态
            mj_data.qpos[:6] = [0, 0, 0, 0, 0, 0]  # 关节初始位置
            mujoco.mj_forward(mj_model, mj_data)
            
            # 运行仿真循环
            start_time = time.time()
            while viewer.is_running():
                # 计算当前时间
                current_time = time.time() - start_time
                
                # 每2秒给关节添加一些随机扰动，使机械臂轻微运动
                if int(current_time) % 2 == 0 and current_time - int(current_time) < 0.1:
                    # 为每个关节添加小的随机扰动
                    for i in range(min(6, mj_model.njnt)):
                        # 获取关节范围
                        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_{i+1}_joint")
                        if joint_id >= 0:
                            range_min = mj_model.jnt_range[joint_id, 0]
                            range_max = mj_model.jnt_range[joint_id, 1]
                            # 在关节范围内生成目标位置
                            target_pos = mj_data.qpos[i] + np.random.uniform(-0.2, 0.2)
                            # 确保在关节范围内
                            target_pos = max(range_min, min(range_max, target_pos))
                            # 设置控制值（简单的PD控制模拟）
                            if i < mj_model.nu:
                                kp = 10.0  # 比例增益
                                error = target_pos - mj_data.qpos[i]
                                mj_data.ctrl[i] = kp * error
                
                # 步进仿真
                mujoco.mj_step(mj_model, mj_data)
                
                # 同步查看器
                viewer.sync()
                
                # 控制仿真速度
                time.sleep(1.0 / 60.0)  # 约60Hz的更新频率
                
    except Exception as e:
        print(f"加载MJCF模型时出错: {e}")
        sys.exit(1)
    
    print("仿真测试完成。")