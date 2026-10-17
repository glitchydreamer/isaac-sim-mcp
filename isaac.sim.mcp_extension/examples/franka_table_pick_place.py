"""
Franka pick-and-place on a table (Isaac Sim 5.x/6.x).

Builds a fresh stage with a table, a Franka mounted at the middle of the tabletop,
a cube on one end and an open box on the other. The Franka picks the cube, drops it
into the box, then returns to its starting joint pose.

Run it inside Isaac Sim through the MCP `execute_script` tool (or the Script Editor).
It is non-blocking: setup runs as an async task and the robot is driven from a
physics callback. Progress is readable from `import pick_place_demo; pick_place_demo.state`.
"""

import asyncio
import sys
import types

import numpy as np
import omni.kit.app
import omni.usd

# The Franka helper classes live in an extension that is not enabled by default
_ext_manager = omni.kit.app.get_app().get_extension_manager()
_ext_manager.set_extension_enabled_immediate("isaacsim.robot.manipulators.examples", True)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

# ---- Layout (meters, Z-up). Robot faces +X. ----
TABLE_SIZE = np.array([0.8, 1.4])  # x (depth), y (length)
TABLE_HEIGHT = 0.75
TABLE_TOP_THICKNESS = 0.05
LEG_SIZE = 0.05
CUBE_SIZE = 0.05
CUBE_XY = np.array([0.30, -0.45])  # one end of the table
BOX_XY = np.array([0.30, 0.45])  # other end of the table
BOX_INNER = 0.14
BOX_WALL = 0.01
BOX_HEIGHT = 0.08
HOME_DURATION_S = 2.0

# Shared state so later scripts can inspect progress
demo = types.ModuleType("pick_place_demo")
demo.state = "setting_up"
demo.error = None
sys.modules["pick_place_demo"] = demo


def _build_table(world):
    top_z = TABLE_HEIGHT - TABLE_TOP_THICKNESS / 2
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table/Top",
            name="table_top",
            position=np.array([0.0, 0.0, top_z]),
            scale=np.array([TABLE_SIZE[0], TABLE_SIZE[1], TABLE_TOP_THICKNESS]),
            color=np.array([0.55, 0.36, 0.2]),
        )
    )
    leg_height = TABLE_HEIGHT - TABLE_TOP_THICKNESS
    dx = TABLE_SIZE[0] / 2 - LEG_SIZE
    dy = TABLE_SIZE[1] / 2 - LEG_SIZE
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Table/Leg{i}",
                name=f"table_leg{i}",
                position=np.array([sx * dx, sy * dy, leg_height / 2]),
                scale=np.array([LEG_SIZE, LEG_SIZE, leg_height]),
                color=np.array([0.4, 0.26, 0.14]),
            )
        )


def _build_box(world):
    """Open-top box made of a floor and four walls."""
    cx, cy = BOX_XY
    outer = BOX_INNER + 2 * BOX_WALL
    color = np.array([0.1, 0.35, 0.8])
    parts = {
        "Floor": ([cx, cy, TABLE_HEIGHT + BOX_WALL / 2], [outer, outer, BOX_WALL]),
        "WallPosX": ([cx + (BOX_INNER + BOX_WALL) / 2, cy, TABLE_HEIGHT + BOX_HEIGHT / 2], [BOX_WALL, outer, BOX_HEIGHT]),
        "WallNegX": ([cx - (BOX_INNER + BOX_WALL) / 2, cy, TABLE_HEIGHT + BOX_HEIGHT / 2], [BOX_WALL, outer, BOX_HEIGHT]),
        "WallPosY": ([cx, cy + (BOX_INNER + BOX_WALL) / 2, TABLE_HEIGHT + BOX_HEIGHT / 2], [BOX_INNER, BOX_WALL, BOX_HEIGHT]),
        "WallNegY": ([cx, cy - (BOX_INNER + BOX_WALL) / 2, TABLE_HEIGHT + BOX_HEIGHT / 2], [BOX_INNER, BOX_WALL, BOX_HEIGHT]),
    }
    for name, (pos, scale) in parts.items():
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Box/{name}",
                name=f"box_{name.lower()}",
                position=np.array(pos),
                scale=np.array(scale),
                color=color,
            )
        )


async def setup():
    try:
        # Fresh stage so reruns start clean
        World.clear_instance()
        await omni.usd.get_context().new_stage_async()
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        world.scene.add_default_ground_plane()

        _build_table(world)
        _build_box(world)
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([CUBE_XY[0], CUBE_XY[1], TABLE_HEIGHT + CUBE_SIZE / 2]),
                scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
                color=np.array([0.9, 0.1, 0.1]),
                mass=0.05,
            )
        )
        franka = world.scene.add(
            Franka(prim_path="/World/Franka", name="franka", position=np.array([0.0, 0.0, TABLE_HEIGHT]))
        )
        franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
        await world.reset_async()

        # Built after reset so RMPflow picks up the robot's base pose on the table
        controller = PickPlaceController(
            name="pick_place_controller",
            gripper=franka.gripper,
            robot_articulation=franka,
            end_effector_initial_height=TABLE_HEIGHT + 0.3,
        )
        articulation_controller = franka.get_articulation_controller()
        home_joints = franka.get_joint_positions().copy()
        # Release just above the box floor
        place_position = np.array([BOX_XY[0], BOX_XY[1], TABLE_HEIGHT + BOX_WALL + CUBE_SIZE / 2 + 0.03])
        home = {"start": None, "t": 0.0}

        def on_physics_step(step_size):
            if demo.state == "pick_and_place":
                cube_position, _ = cube.get_world_pose()
                action = controller.forward(
                    picking_position=cube_position,
                    placing_position=place_position,
                    current_joint_positions=franka.get_joint_positions(),
                    end_effector_offset=np.array([0, 0.005, 0]),
                )
                articulation_controller.apply_action(action)
                demo.phase = controller.get_current_event()
                if controller.is_done():
                    demo.state = "returning_home"
                    home["start"] = franka.get_joint_positions().copy()
            elif demo.state == "returning_home":
                home["t"] = min(home["t"] + step_size / HOME_DURATION_S, 1.0)
                alpha = 0.5 * (1 - np.cos(np.pi * home["t"]))
                target = (1 - alpha) * home["start"] + alpha * home_joints
                articulation_controller.apply_action(ArticulationAction(joint_positions=target))
                if home["t"] >= 1.0:
                    demo.state = "done"

        demo.world = world
        demo.franka = franka
        demo.cube = cube
        demo.home_joints = home_joints
        demo.box_center = np.array([BOX_XY[0], BOX_XY[1], TABLE_HEIGHT])
        world.add_physics_callback("franka_pick_place", on_physics_step)
        demo.state = "pick_and_place"
        await world.play_async()
    except Exception as e:
        import traceback

        demo.state = "error"
        demo.error = traceback.format_exc()
        print(demo.error)


asyncio.ensure_future(setup())
print("Franka table pick-and-place scheduled; check `import pick_place_demo; pick_place_demo.state`")
