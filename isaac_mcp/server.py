"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp[cli]>=1.6,<2"]
# ///
# isaac_sim_mcp_server.py
import time
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List
import os
from pathlib import Path
import base64
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsaacMCPServer")

@dataclass
class IsaacConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict
    
    def connect(self) -> bool:
        """Connect to the Isaac addon socket server"""
        if self.sock:
            return True
            
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Isaac at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Isaac: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Isaac addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Isaac: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=16384):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a consistent timeout value that matches the addon's timeout
        sock.settimeout(300.0)  # Match the extension's timeout
        
        try:
            while True:
                try:
                    logger.info("Waiting for data from Isaac")
                    #time.sleep(0.5)
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        # If we get an empty chunk, the connection might be closed
                        if not chunks:  # If we haven't received anything yet, this is an error
                            raise Exception("Connection closed before receiving any data")
                        break
                    
                    chunks.append(chunk)
                    
                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        # If we get here, it parsed successfully
                        logger.info(f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    # If we hit a timeout during receiving, break the loop and try to use what we have
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {str(e)}")
                    raise  # Re-raise to be handled by the caller
        except socket.timeout:
            logger.warning("Socket timeout during chunked receive")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise
            
        # If we get here, we either timed out or broke out of the loop
        # Try to use what we have
        if chunks:
            data = b''.join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                # Try to parse what we have
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                # If we can't parse it, it's incomplete
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Isaac and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Isaac")
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        try:
            # Log the command being sent
            logger.info(f"Sending command: {command_type} with params: {params}")
            
            # Send the command
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f"Command sent, waiting for response...")
            
            # Set a timeout for receiving - use the same timeout as in receive_full_response
            self.sock.settimeout(300.0)  # Match the extension's timeout
            
            # Receive the response using the improved receive_full_response method
            response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")
            
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            
            if response.get("status") == "error":
                logger.error(f"Isaac error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error from Isaac"))
            
            return response.get("result", {})
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Isaac")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            # Just invalidate the current socket so it will be recreated next time
            self.sock = None
            raise Exception("Timeout waiting for Isaac response - try simplifying your request")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Isaac lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Isaac: {str(e)}")
            # Try to log what was received
            if 'response_data' in locals() and response_data:
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            raise Exception(f"Invalid response from Isaac: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Isaac: {str(e)}")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            self.sock = None
            raise Exception(f"Communication error with Isaac: {str(e)}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    # We don't need to create a connection here since we're using the global connection
    # for resources and tools
    
    try:
        # Just log that we're starting up
        logger.info("IsaacMCP server starting up")
        
        # Try to connect to Isaac on startup to verify it's available
        try:
            # This will initialize the global connection if needed
            isaac = get_isaac_connection()
            logger.info("Successfully connected to Isaac on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Isaac on startup: {str(e)}")
            logger.warning("Make sure the Isaac addon is running before using Isaac resources or tools")
        
        # Return an empty context - we're using the global connection
        yield {}
    finally:
        # Clean up the global connection on shutdown
        global _isaac_connection
        if _isaac_connection:
            logger.info("Disconnecting from Isaac Sim on shutdown")
            _isaac_connection.disconnect()
            _isaac_connection = None
        logger.info("Isaac SimMCP server shut down")

# Create the MCP server with lifespan support
mcp = FastMCP(
    "IsaacSimMCP",
    instructions="Isaac Sim integration through the Model Context Protocol",
    lifespan=server_lifespan
)

# Resource endpoints

# Global connection for resources (since resources can't access context)
_isaac_connection = None
# _polyhaven_enabled = False  # Add this global variable

def get_isaac_connection():
    """Get or create a persistent Isaac connection"""
    global _isaac_connection, _polyhaven_enabled  # Add _polyhaven_enabled to globals
    
    # If we have an existing connection, check if it's still valid
    if _isaac_connection is not None:
        try:
            
            return _isaac_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _isaac_connection.disconnect()
            except:
                pass
            _isaac_connection = None
    
    # Create a new connection if needed
    if _isaac_connection is None:
        _isaac_connection = IsaacConnection(host="localhost", port=8766)
        if not _isaac_connection.connect():
            logger.error("Failed to connect to Isaac")
            _isaac_connection = None
            raise Exception("Could not connect to Isaac. Make sure the Isaac addon is running.")
        logger.info("Created new persistent connection to Isaac")
    
    return _isaac_connection


@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Ping status of Isaac Sim Extension Server"""
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_scene_info")
        logger.info(f"result: {result}")
        
        # Just return the JSON representation of what Isaac sent us
        return json.dumps(result, indent=2)
        # return json.dumps(result)
        # return result
    except Exception as e:
        logger.error(f"Error getting scene info from Isaac: {str(e)}")
        # return f"Error getting scene info: {str(e)}"
        return f"Error getting scene info: {str(e)}"

# @mcp.tool()
# def get_object_info(ctx: Context, object_name: str) -> str:
#     """
#     Get detailed information about a specific object in the Isaac scene.
    
#     Parameters:
#     - object_name: The name of the object to get information about
#     """
#     try:
#         isaac = get_isaac_connection()
#         result = isaac.send_command("get_object_info", {"name": object_name})
        
#         # Just return the JSON representation of what Isaac sent us
#         return json.dumps(result, indent=2)
#     except Exception as e:
#         logger.error(f"Error getting object info from Isaac: {str(e)}")
#         return f"Error getting object info: {str(e)}"

@mcp.tool("create_physics_scene")
def create_physics_scene(
    objects: List[Dict[str, Any]] = [],
    floor: bool = True,
    gravity: List[float] = [0, 0, -9.81],
    scene_name: str = "physics_scene"
) -> str:
    """Create a physics scene (gravity + optional ground plane) and optionally add primitive objects with rigid-body physics.
    Call get_scene_info() first to verify the connection. The stage is Z-up and in meters. Existing paths are skipped.

    Args:
        objects: List of objects to create. Supported types: Cube, Sphere, Cylinder, Cone, Capsule. Keys per object:
            type, path (or name), position [x, y, z], rotation (quaternion [w, x, y, z]), scale [x, y, z],
            size (Cube edge length), radius, height, color [r, g, b, (a)] in 0-1,
            physics_enabled (default True), mass (kg, default 1.0), is_kinematic (default False). Example:
            [{"path": "/World/Cube", "type": "Cube", "size": 0.5, "position": [0, 0, 2], "color": [1, 0, 0]},
             {"path": "/World/Ball", "type": "Sphere", "radius": 0.2, "position": [1, 0, 3]}]
        floor: Whether to create a ground plane. Default True.
        gravity: Gravity vector in m/s^2. Default [0, 0, -9.81].
        scene_name: Name of the scene.

    Returns:
        String with result information.
    """
    params = {"objects": objects, "floor": floor}
    
    if gravity is not None:
        params["gravity"] = gravity
    if scene_name is not None:
        params["scene_name"] = scene_name
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("create_physics_scene", params)
        return f"create_physics_scene successfully: {result.get('result', '')}, {result.get('message', '')}"
    except Exception as e:
        logger.error(f"Error create_physics_scene: {str(e)}")
        return f"Error create_physics_scene: {str(e)}"
    
@mcp.tool("create_robot")
def create_robot(robot_type: str = "g1", position: List[float] = [0, 0, 0]) -> str:
    """Create a robot in the Isaac scene. Directly create robot prim in stage at the right position. For any creation of robot, you need to call create_physics_scene() first. call create_robot() as first attmpt beofre call execute_script().
    
    Args:
        robot_type: The type of robot to create. Available options:
            - "franka": Franka Emika Panda arm
            - "jetbot": NVIDIA JetBot
            - "carter": NVIDIA Nova Carter
            - "g1": Unitree G1 humanoid (default)
            - "go1": Unitree Go1 quadruped
            - "go2": Unitree Go2 quadruped
            - "h1": Unitree H1 humanoid
        position: World position [x, y, z] in meters. Repeated calls create /Franka, /Franka_01, ... instead of overwriting.
        
    Returns:
        String with result information.
    """
    isaac = get_isaac_connection()
    result = isaac.send_command("create_robot", {"robot_type": robot_type, "position": position})
    return f"create_robot successfully: {result.get('result', '')}, {result.get('message', '')}"

@mcp.tool("omni_kit_command")
def omni_kit_command(command: str = "CreatePrim", prim_type: str = "Sphere") -> str:
    """Execute an Omni Kit command.
    
    Args:
        command: The Omni Kit command to execute.
        prim_type: The primitive type for the command.
        
    Returns:
        String with result information.
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("omini_kit_command", {
            "command": command,
            "prim_type": prim_type
        })
        return f"Omni Kit command executed successfully: {result.get('message', '')}"
    except Exception as e:
        logger.error(f"Error executing Omni Kit command: {str(e)}")
        # return f"Error executing Omni Kit command: {str(e)}"
        return f"Error executing Omni Kit command: {str(e)}"


@mcp.tool()
def execute_script(ctx: Context, code: str) -> str:
    """
    Execute arbitrary Python code inside the running Isaac Sim (5.x/6.x) GUI on its main thread.
    Call get_scene_info() first to verify the connection. Prefer create_robot() / create_physics_scene() for simple tasks.

    Return values: anything the script print()s is returned as "stdout", and if the script assigns a
    variable named `result` (JSON-serializable) it is returned as "result". Use this to inspect the stage.
    Pre-imported names: omni, carb, Usd, UsdGeom, Sdf, Gf.

    Use the Isaac Sim 5.x/6.x module names (the old omni.isaac.* modules no longer exist):
      isaacsim.core.api (World, SimulationContext, PhysicsContext), isaacsim.core.api.objects (DynamicCuboid, ...),
      isaacsim.core.api.robots (Robot), isaacsim.core.prims (SingleArticulation, SingleXFormPrim, SingleRigidPrim),
      isaacsim.core.utils.stage / .prims / .types (add_reference_to_stage, create_prim, ArticulationAction),
      isaacsim.storage.native (get_assets_root_path), isaacsim.robot.wheeled_robots (WheeledRobot, DifferentialController).
    Robot assets (append to get_assets_root_path()):
      /Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd, /Isaac/Robots/NVIDIA/Jetbot/jetbot.usd,
      /Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd, /Isaac/Robots/Unitree/G1/g1.usd, /Isaac/Robots/Unitree/Go1/go1.usd,
      /Isaac/Robots/Unitree/Go2/go2.usd, /Isaac/Robots/Unitree/H1/h1.usd

    IMPORTANT: the script runs on the GUI main thread. Never write blocking loops such as
    `for i in range(1000): simulation_context.step(render=True)` - they freeze Isaac Sim and time out.
    Instead do setup in an async function scheduled with asyncio.ensure_future(), and drive robots
    from a physics callback while the timeline plays. Example (Franka joint control):

import asyncio
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

async def setup():
    world = World.instance() or World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()
    add_reference_to_stage(get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "/World/Franka")
    franka = world.scene.add(SingleArticulation(prim_path="/World/Franka", name="franka"))
    await world.reset_async()
    idx = franka.get_dof_index("panda_joint2")

    def on_physics_step(step_size):
        franka.set_joint_positions(np.array([-1.5]), joint_indices=np.array([idx]))

    world.add_physics_callback("franka_control", on_physics_step)
    await world.play_async()

asyncio.ensure_future(setup())

    To stop: `import omni.timeline; omni.timeline.get_timeline_interface().stop()`.
    To remove a callback: `World.instance().remove_physics_callback("franka_control")`.

    Parameters:
    - code: The Python code to execute, e.g. "omni.kit.commands.execute("CreatePrim", prim_type="Sphere")"
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        logger.info(f"code: {code}")

        result = isaac.send_command("execute_script", {"code": code})
        logger.info(f"result: {result}")
        return json.dumps(result, indent=2)
        # return f"Code executed successfully: {result.get('result', '')}"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        # return f"Error executing code: {str(e)}"
        return f"Error executing code: {str(e)}"
                
@mcp.prompt()
def asset_creation_strategy() -> str:
    """Defines the preferred strategy for creating assets in Isaac Sim"""
    return """
    0. Before anything, always check the scene from get_scene_info(), retrive rool path of assset through return value of assets_root_path.
    1. If the scene is empty, create a physics scene with create_physics_scene()
    2. if execute script due to communication error, then retry 3 times at most

    3. Add robots with create_robot() where possible (franka, jetbot, carter, g1, go1, go2, h1).
    4. For anything custom use execute_script() with Isaac Sim 5.x/6.x modules (isaacsim.core.api, isaacsim.core.prims,
       isaacsim.core.utils, isaacsim.storage.native, isaacsim.robot.wheeled_robots) - never omni.isaac.*.
    5. Never block the main thread with step() loops. Do setup in an async function scheduled with
       asyncio.ensure_future(), and drive robots from world.add_physics_callback(...) while the timeline plays
       (see the execute_script() description for a full example).
    6. Reference examples for longer simulations live in isaac.sim.mcp_extension/examples (franka.py, franka_grid.py, g1.py, go1.py).
    """


def _process_bbox(original_bbox: list[float] | list[int] | None) -> list[int] | None:
    if original_bbox is None:
        return None
    if all(isinstance(i, int) for i in original_bbox):
        return original_bbox
    if any(i<=0 for i in original_bbox):
        raise ValueError("Incorrect number range: bbox must be bigger than zero!")
    return [int(float(i) / max(original_bbox) * 100) for i in original_bbox] if original_bbox else None


#@mcp.tool()
def get_beaver3d_status(ctx: Context) -> str:
    """
    TODO: Get the status of Beaver3D.
    """
    return "Beaver3D service is Available"



@mcp.tool("generate_3d_from_text_or_image")
def generate_3d_from_text_or_image(
    ctx: Context,
    text_prompt: str = None,
    image_url: str = None,
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Generate a 3D model from text or image, load it into the scene and transform it.
    
    Args:
        text_prompt (str, optional): Text prompt for 3D generation
        image_url (str, optional): URL of image for 3D generation
        position (list, optional): Position to place the model [x, y, z]
        scale (list, optional): Scale of the model [x, y, z]
        
    Returns:
        String with the task_id and prim_path information
    """
    if not (text_prompt or image_url):
        return "Error: Either text_prompt or image_url must be provided"
    
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("generate_3d_from_text_or_image", {
            "text_prompt": text_prompt,
            "image_url": image_url,
            "position": position,
            "scale": scale
        })
        
        if result.get("status") == "success":
            task_id = result.get("task_id")
            prim_path = result.get("prim_path")
            return f"Successfully generated 3D model with task ID: {task_id}, loaded at prim path: {prim_path}"
        else:
            return f"Error generating 3D model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        return f"Error generating 3D model: {str(e)}"
    
@mcp.tool("search_3d_usd_by_text")
def search_3d_usd_by_text(
    ctx: Context,
    text_prompt: str = None,
    target_path: str = "/World/my_usd",
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Search for a 3D model using text prompt in USD libraries, then load and position it in the scene.
    
    Args:
        text_prompt (str): Text description to search for matching 3D models
        target_path (str, optional): Path where the USD model will be placed in the scene
        position (list, optional): Position coordinates [x, y, z] for placing the model
        scale (list, optional): Scale factors [x, y, z] to resize the model
        
    Returns:
        String with search results including task_id and prim_path of the loaded model
    """
    if not text_prompt:
        return "Error: Either text_prompt or image_url must be provided"
    
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        params = {"text_prompt": text_prompt, 
                  "target_path": target_path}
            
        result = isaac.send_command("search_3d_usd_by_text", params)
        if result.get("status") == "success":
            task_id = result.get("task_id")
            prim_path = result.get("prim_path")
            return f"Successfully generated 3D model with task ID: {task_id}, loaded at prim path: {prim_path}"
        else:
            return f"Error generating 3D model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        return f"Error generating 3D model: {str(e)}"

@mcp.tool("transform")
def transform(
    ctx: Context,
    prim_path: str,
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Transform a USD model by applying position and scale.
    
    Args:
        prim_path (str): Path to the USD prim to transform
        position (list, optional): The position to set [x, y, z]
        scale (list, optional): The scale to set [x, y, z]
        
    Returns:
        String with transformation result
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("transform", {
            "prim_path": prim_path,
            "position": position,
            "scale": scale
        })
        
        if result.get("status") == "success":
            return f"Successfully transformed model at {prim_path} to position {position} and scale {scale}"
        else:
            return f"Error transforming model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error transforming model: {str(e)}")
        return f"Error transforming model: {str(e)}"

# Main execution

def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()