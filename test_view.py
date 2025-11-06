
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/data/zhangyx23Files/program/real2sim2real/DISCOVERSE/models/mjcf/tasks_xbot_arm/place_block.xml")
# model = mujoco.MjModel.from_xml_path("/data/zhangyx23Files/program/real2sim2real/DISCOVERSE/models/mjcf/tasks_airbot_play/place_block.xml")
data = mujoco.MjData(model)

print(data.qpos[:10])
# 打印 joint 名称与对应自由度索引
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    adr = model.jnt_dofadr[i]
    print(f"{i:2d}: joint name = {name}, dof addr = {adr}")


for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    parent = model.body_parentid[i]
    jntadr = model.body_jntadr[i]
    jntnum = model.body_jntnum[i]
    print(f"{i:2d}: body={name}, parent={parent}, joint adr={jntadr}, joint num={jntnum}")
    
    
data.ctrl[:] = 0
mujoco.mj_forward(model, data)
# print(data.qpos)
# print(model.jnt_range[:model.njnt])

mujoco.viewer.launch(model, data)


