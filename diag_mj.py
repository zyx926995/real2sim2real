# diag_mj.py
import mujoco
import numpy as np
import math

MODEL_PATH = "/data/zhangyx23Files/program/real2sim2real/DISCOVERSE/models/mjcf/tasks_xbot_arm/place_block.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# helper name lookup
def name_obj(type, id):
    return mujoco.mj_id2name(model, type, id) if id>=0 else "None"

# zero controls
data.ctrl[:] = 0.0

# forward and step a few times, printing diagnostics
mujoco.mj_forward(model, data)

def print_basic(step):
    print(f"\n=== STEP {step} ===")
    print("qpos[:8] =", np.round(data.qpos[:8], 6))
    print("qvel[:8] =", np.round(data.qvel[:8], 6))
    print("qacc[:8] =", np.round(data.qacc[:8], 6))
    print("ctrl[:8] =", np.round(data.ctrl[:8], 6))
    print("ncon =", data.ncon)

# print all geom world positions and AABB extents
def geom_world_info():
    print("\n-- geom world positions and sizes (first 100) --")
    ngeom_data = data.xpos.shape[0]
    for gi in range(ngeom_data):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        pos = data.xpos[gi]
        size = model.geom_size[gi] if gi < len(model.geom_size) else np.zeros(3)
        typeid = model.geom_type[gi] if gi < len(model.geom_type) else -1
        print(f"{gi:3d}: geom='{gname}', type={typeid}, pos={np.round(pos,4)}, size={np.round(size,4)}")

# simple AABB overlap test (conservative, uses size as extents)
def check_aabb_overlaps():
    overlaps = []
    ngeom_data = data.xpos.shape[0]
    for i in range(ngeom_data):
        pi = data.xpos[i]
        si = model.geom_size[i]
        mini = pi - si
        maxi = pi + si
        for j in range(i + 1, ngeom_data):
            pj = data.xpos[j]
            sj = model.geom_size[j]
            minj = pj - sj
            maxj = pj + sj

            overlap = True
            pen_depth = []
            for k in range(3):
                # 检查轴向是否分离
                if mini[k] > maxj[k] or maxi[k] < minj[k]:
                    overlap = False
                    break
                pd = min(maxi[k] - minj[k], maxj[k] - mini[k])
                pen_depth.append(pd)

            if overlap:
                overlaps.append((i, j, pen_depth))

    if overlaps:
        print("\n!!! AABB overlaps detected (geom i, geom j, approx pen_depth per axis):")
        for (i, j, pd) in overlaps:
            ngi = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            ngj = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, j)
            print(f"  {i}('{ngi}') <-> {j}('{ngj}') pen_depth={np.round(pd,5)}")
    else:
        print("\nNo AABB overlaps detected by conservative test.")


# step loop with contact dump
print_basic(0)
geom_world_info()
check_aabb_overlaps()

# take a few steps and watch contacts
for step in range(1, 6):
    mujoco.mj_step(model, data)
    print_basic(step)
    if data.ncon > 0:
        print("\nContacts:")
        for ci in range(data.ncon):
            c = data.contact[ci]
            # data.contact[i] gives contact struct — get geom1/geom2 indices via c.geom1, c.geom2
            g1 = c.geom1
            g2 = c.geom2
            g1name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            g2name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            # approximate penetration (c.dist is negative for penetration)
            dist = c.dist
            # contact force approx: use c.frame? (not directly). Print contact normal and dim
            print(f" contact {ci}: geom1={g1}('{g1name}') geom2={g2}('{g2name}') dist={dist:.6e} mu={c.mu:.6e}")
    else:
        print(" no contacts")
