import os
import numpy as np
import gym
from gym import spaces
from .Simulator import ArmSim as Sim, TestPlotter, TestPlotterOneEye, VisualSensor
import pkg_resources
from scipy import ndimage


def softmax(x, t=0.003):
    e = np.exp((x - np.min(x)) / t)
    return e / e.sum()


def DefaultRewardFun(observation):
    return np.sum(observation["TOUCH_SENSORS"])


class ArmSimOneArmEnv(gym.Env):
    """A single 2D arm ArmSimwith a box-shaped object"""

    metadata = {"render.modes": ["human", "offline"]}
    robot_parts_names = [
        "Base",
        "Arm1",
        "Arm2",
        "Arm3",
        "claw11",
        "claw21",
        "claw12",
        "claw22",
    ]
    joint_names = [
        "Ground_to_Arm1",
        "Arm1_to_Arm2",
        "Arm2_to_Arm3",
        "Arm3_to_Claw11",
        "Claw21_to_Claw22",
        "Arm3_to_Claw21",
        "Claw11_to_Claw12",
    ]
    sensors_names = ["claw11", "claw21", "claw12", "claw22"]

    def __init__(self):

        super(ArmSimOneArmEnv, self).__init__()

        self.set_seed()

        self.init_worlds()
        self.init_map()

        self.num_joints = 5
        self.num_arm_joints = 3
        self.num_hand_joints = 2
        self.num_joint_positions = 7
        self.num_touch_sensors = 4

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(-np.pi, np.pi, [self.num_joints], dtype=float)

        self.observation_space = gym.spaces.Dict(
            {
                "JOINT_POSITIONS": gym.spaces.Box(
                    -np.inf, np.inf, [self.num_joint_positions], dtype=float
                ),
                "TOUCH_SENSORS": gym.spaces.Dict(
                    {
                        obj_name: gym.spaces.Box(
                            0, np.inf, [self.num_touch_sensors], dtype=float
                        )
                        for obj_name in self.object_names
                    }
                ),
                "OBJ_POSITION": gym.spaces.Box(
                    -np.inf, np.inf, [len(self.object_names), 2], dtype=float
                ),
            }
        )

        self.rendererType = TestPlotter
        self.renderer = None
        self.renderer_figsize = (3, 3)

        self.taskspace_xlim = [-10, 30]
        self.taskspace_ylim = [-10, 30]

        self.hand_pos_sal = True

        self.set_reward_fun()

        self.world_id = None
        self.reset()

    def set_renderer_figsize(self, figsize):
        self.renderer_figsize = figsize

    def init_worlds(self):
        self.world_files = [
            pkg_resources.resource_filename("ArmSim", "models/arm_2obj_diff.json")
        ]
        self.worlds = {"arm_2obj_diff": 0}
        self.world_object_names = {0: ["Object1", "Object2"]}
        self.object_names = self.world_object_names[self.worlds["arm_2obj_diff"]]

    def init_map(self):
        object_map_file = pkg_resources.resource_filename(
            "ArmSim", "data/ObjectPositionsMap.npy"
        )
        self.object_map = np.load(object_map_file).T

    def set_seed(self, seed=None):
        self.seed = seed
        if self.seed is None:
            self.seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
        self.rng = np.random.RandomState(self.seed)

    def set_reward_fun(self, rew_fun=None):

        self.reward_fun = rew_fun
        if self.reward_fun is None:
            self.reward_fun = DefaultRewardFun

    def set_action(self, action):

        assert len(action) == self.num_joints
        action = np.hstack(action)
        # do action
        action[:-2] = np.maximum(-np.pi * 0.5, np.minimum(np.pi * 0.5, action[:-2]))
        action[-1] = np.maximum(0, np.minimum(2 * action[-2], action[-1]))
        action[-2:] = -np.maximum(0, np.minimum(np.pi * 0.5, action[-2:]))
        action = np.hstack((action, -action[-2:]))
        for j, joint in enumerate(self.joint_names):
            self.sim.move(joint, action[j])
        self.sim.step()

    def get_observation(self):

        joints = np.array([self.sim.joints[name].angle for name in self.joint_names])
        sensors = np.array(
            [
                np.sum(
                    [
                        self.sim.contacts(sensor_name, object_name)
                        for object_name in self.object_names
                    ]
                )
                for sensor_name in self.sensors_names
            ]
        )
        obj_pos = np.array(
            [
                [self.sim.bodies[object_name].worldCenter]
                for object_name in self.object_names
            ]
        )

        return joints, sensors, obj_pos

    def sim_step(self, action):

        self.set_action(action)
        joints, sensors, obj_pos = self.get_observation()

        observation = {
            "JOINT_POSITIONS": joints,
            "TOUCH_SENSORS": sensors,
            "OBJ_POSITION": obj_pos,
        }

        return observation

    def step(self, action):

        observation = self.sim_step(action)

        # compute reward
        reward = self.reward_fun(observation)

        # compute end of task
        done = False

        # other info
        info = {}

        return observation, reward, done, info

    def choose_worldfile(self, world_id=None):

        self.world_id = world_id
        if self.world_id is None:
            self.world_id = self.rng.randint(0, len(self.world_files))

        self.world_file = self.world_files[self.world_id]

    def randomize_objects(self, world_dict):

        for bodyName in self.object_names:
            for i in range(len(world_dict["body"])):
                if world_dict["body"][i]["name"] == bodyName:
                    verts = np.vstack(
                        [
                            world_dict["body"][i]["fixture"][0]["polygon"]["vertices"][
                                "x"
                            ],
                            world_dict["body"][i]["fixture"][0]["polygon"]["vertices"][
                                "y"
                            ],
                        ]
                    )

                    vmean = verts.mean()
                    verts = (verts - vmean) * (0.8 + 0.2 * self.rng.rand()) + 0.2 * self.rng.randn(*verts.shape)


                    world_dict["body"][i]["fixture"][0]["polygon"]["vertices"][
                        "x"
                    ] = verts[0].tolist()
                    world_dict["body"][i]["fixture"][0]["polygon"]["vertices"][
                        "y"
                    ] = verts[1].tolist()

                    # pos = [self.rng.uniform(-3, 3), self.rng.uniform(-1, 5)]
                    # world_dict["body"][i]["position"]["x"] += pos[1]
                    # world_dict["body"][i]["position"]["y"] += pos[0]

                    color = 0.1 * np.random.randn(3)
                    color = color + world_dict["body"][i]["color"]
                    color = np.maximum(0, np.minimum(1, color))
                    world_dict["body"][i]["color"] = list(color)
                    break

        return world_dict

    def prepare_world(self, world_id=None, world_dict=None):
        if world_id is not None:
            self.world_id = world_id

        self.choose_worldfile(world_id)
        self.object_names = self.world_object_names[self.world_id]

        if world_dict is None:
            world_dict = Sim.loadWorldJson(self.world_file)
            world_dict = self.randomize_objects(world_dict)

        return world_dict


    def set_world(self, world_id=None, world_dict=None):

        world_dict = self.prepare_world(world_id, world_dict)
        self.sim = Sim(world_dict=world_dict)


    def reset(self, world_id=None, world_dict=None):

        if world_id is not None:
            self.world_id = world_id

        self.set_world(self.world_id, world_dict)

        if self.renderer is not None:
            self.renderer.reset()

        return self.sim_step(np.zeros(self.num_joints))

    def render_init(self, mode):
        if self.renderer is not None:
            self.renderer.close()
        if mode == "human":
            self.renderer = self.rendererType(
                self,
                xlim=self.taskspace_xlim,
                ylim=self.taskspace_ylim,
                figsize=self.renderer_figsize,
            )
        elif mode == "offline":
            self.renderer = self.rendererType(
                self,
                xlim=self.taskspace_xlim,
                ylim=self.taskspace_ylim,
                offline=True,
                figsize=self.renderer_figsize,
            )
        else:
            self.renderer = None

    def render_check(self, mode):
        if self.renderer is None:
            self.render_init(mode)

    def render(self, mode="human"):
        self.render_check(mode)
        self.renderer.step()


class ArmSimOneArmOneEyeEnv(ArmSimOneArmEnv):
    def __init__(self, *args, **kargs):

        self.t = 0
        self.init_salience_filters()

        self.fovea_width = 4
        self.fovea_height = 4
        self.fovea_pixel_side = 10

        super(ArmSimOneArmOneEyeEnv, self).__init__(*args, **kargs)

        self.rendererType = TestPlotterOneEye
        self.eye_pos = [0, 0]

    def init_worlds(self):

        self.world_files = [
            pkg_resources.resource_filename("ArmSim", "models/unreachable.json"),
            pkg_resources.resource_filename("ArmSim", "models/still.json"),
            pkg_resources.resource_filename("ArmSim", "models/movable.json"),
            pkg_resources.resource_filename("ArmSim", "models/controllable.json"),
            pkg_resources.resource_filename("ArmSim", "models/noobject.json"),
        ]

        self.worlds = {
            "unreachable": 0,
            "still": 1,
            "movable": 2,
            "controllable": 3,
            "noobject": 4,
        }

        self.world_object_names = {
            0: ["unreachable"],
            1: ["still"],
            2: ["movable"],
            3: ["controllable"],
            4: [],
        }

        self.object_names = self.world_object_names[self.worlds["unreachable"]]

    def reset(self, *args, **kargs):

        observation = super(ArmSimOneArmOneEyeEnv, self).reset(*args, **kargs)
        self.set_taskspace(self.taskspace_xlim, self.taskspace_ylim)
        return observation

    def set_world(self, world=None, world_dict=None):
        super(ArmSimOneArmOneEyeEnv, self).set_world(world, world_dict)
        self.set_taskspace(self.taskspace_xlim, self.taskspace_ylim)
        self.bground.reset(self.sim)
        self.fovea.reset(self.sim)

    def set_taskspace(self, xlim, ylim):

        self.taskspace_xlim = xlim
        self.taskspace_ylim = ylim

        self.bground_width = np.diff(self.taskspace_xlim)[0]
        self.bground_height = np.diff(self.taskspace_ylim)[0]
        self.bground_pixel_side = int(self.bground_width)
        self.bground = VisualSensor(
            self.sim,
            shape=(self.bground_pixel_side, self.bground_pixel_side),
            rng=(self.bground_width, self.bground_height),
        )

        self.bground_ratio = np.array(
            [
                self.bground_width / self.bground_pixel_side,
                self.bground_height / self.bground_pixel_side,
            ]
        )

        self.fovea = VisualSensor(
            self.sim,
            shape=(self.fovea_pixel_side, self.fovea_pixel_side),
            rng=(self.fovea_width, self.fovea_height),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "JOINT_POSITIONS": gym.spaces.Box(
                    -np.inf, np.inf, [self.num_joint_positions], dtype=float
                ),
                "TOUCH_SENSORS": gym.spaces.Box(
                    0, np.inf, [self.num_touch_sensors], dtype=float
                ),
                "VISUAL_SENSORS": gym.spaces.Box(
                    0, np.inf, self.fovea.shape + [3], dtype=float
                ),
                "OBJ_POSITION": gym.spaces.Box(-np.inf, np.inf, [2], dtype=float),
            }
        )

    def get_salient_points(self, bground):
        pass

    def sim_step(self, action):

        self.set_action(action)
        joints, sensors, obj_pos = self.get_observation()

        self.bground_img = self.bground.step(
            [
                self.bground_height / 2 + self.taskspace_ylim[0],
                self.bground_width / 2 + self.taskspace_xlim[0],
            ]
        )
        saliency = None

        # visual
        if self.t % 5 == 0:
            saliency = self.filter(self.bground_img)
            self.visual = self.fovea.step(self.eye_pos)

        observation = {
            "JOINT_POSITIONS": joints,
            "TOUCH_SENSORS": sensors,
            "VISUAL_SENSORS": self.visual,
            "OBJ_POSITION": obj_pos,
        }

        self.t += 1

        return observation
