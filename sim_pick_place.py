#!/usr/bin/env python3
"""
Pick & Place dla TIAGo Pro – ROS 2 / MoveIt 2
Wersja z pełnym planowaniem OMPL dla obu ramion
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import time
import math

from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (PlanningScene, CollisionObject, Constraints,
                              JointConstraint, RobotState, PositionConstraint, 
                              OrientationConstraint, BoundingVolume)
from shape_msgs.msg import SolidPrimitive

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────
R = math.radians

HOME_JOINTS = [0.15, R(-176), R(-53), R(95), R(31), R(-7), R(-69), R(12)]
LEFT_HOME_JOINTS = [R(145), R(-87), R(-35), R(-68), R(-38), R(-26), R(10)]
HANDOVER_JOINTS = [0.15, R(-74), R(-78), R(117), R(-117), R(191), R(-91), R(0)]

GRASP_X, GRASP_Y, GRASP_Z = 0.80, 0.00, 1.0
APPROACH_Z = GRASP_Z + 0.10
ORI_X, ORI_Y, ORI_Z, ORI_W = 0.0, 1.0, 0.0, 0.0

BOX_ID, BOX_SIZE, BOX_CENTER = 'sim_table', (0.80, 1.43, 0.03), (0.9, 0.00, 0.735)

ARM_GROUP = 'arm_right_torso'
LEFT_ARM_GROUP = 'arm_left'  # Grupa dla lewego ramienia
BASE_FRAME = 'base_footprint'
ARM_JOINTS = ['torso_lift_joint'] + [f'arm_right_{i}_joint' for i in range(1, 8)]
LEFT_ARM_JOINTS = [f'arm_left_{i}_joint' for i in range(1, 8)]
GRIPPER_JOINTS = ['gripper_right_finger_joint']
GRIPPER_OPEN, GRIPPER_CLOSED = [0.05], [0.80]

PLANNING_TIME, CART_STEP, CART_JUMP, MIN_FRACTION = 10.0, 0.005, 0.0, 0.90

class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')
        
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        
        self.action_clients = {
            'move': ActionClient(self, MoveGroup, '/move_action'),
            'arm': ActionClient(self, FollowJointTrajectory, '/arm_right_controller/follow_joint_trajectory'),
            'arm_left': ActionClient(self, FollowJointTrajectory, '/arm_left_controller/follow_joint_trajectory'),
            'torso': ActionClient(self, FollowJointTrajectory, '/torso_controller/follow_joint_trajectory'),
            'gripper': ActionClient(self, FollowJointTrajectory, '/gripper_right_controller/follow_joint_trajectory')
        }
        
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._joints = {}
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

    def _js_cb(self, msg):
        self._joints.update(dict(zip(msg.name, msg.position)))

    def _wait_for_joints(self, timeout=3.0):
        deadline = time.time() + timeout
        while len(self._joints) < 5 and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _spin_sleep(self, sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _send_action(self, client, goal, wait_for_result=True):
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh or not gh.accepted:
            return False
        if not wait_for_result:
            return gh 
        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        return res_future.result()

    def _robot_state(self) -> RobotState:
        rs = RobotState()
        rs.joint_state = JointState(name=list(self._joints.keys()), position=list(self._joints.values()))
        return rs

    def _make_pose(self, x, y, z) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = ORI_X, ORI_Y, ORI_Z, ORI_W
        return p

    def _build_traj_goal(self, joint_names, positions, duration_sec=3.0):
        traj = JointTrajectory()
        traj.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(seconds=duration_sec).to_msg()
        traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        return goal

    def _move_direct(self, client_key, joint_names, positions, duration=3.0):
        goal = self._build_traj_goal(joint_names, positions, duration)
        res = self._send_action(self.action_clients[client_key], goal)
        self._spin_sleep(0.5) 
        return res is not False

    def _ompl(self, joints: list, arm_group: str) -> bool:
        """Planowanie OMPL z automatycznym omijaniem kolizji"""
        goal = MoveGroup.Goal()
        goal.request.group_name = arm_group
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 10  # Zwiększono liczbę prób
        goal.request.planner_id = "RRTConnect"  # Jawne wskazanie plannera

        constraints = Constraints()
        target_joints = LEFT_ARM_JOINTS if arm_group == LEFT_ARM_GROUP else ARM_JOINTS
        
        for name, pos in zip(target_joints, joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        self.get_logger().info(f"Planowanie OMPL dla grupy: {arm_group}")
        res = self._send_action(self.action_clients['move'], goal)
        
        if not res:
            self.get_logger().error(f"OMPL [{arm_group}]: Cel odrzucony przez serwer")
            return False
            
        if res.result.error_code.val != 1:
            self.get_logger().error(f"OMPL [{arm_group}]: Błąd planowania. Kod: {res.result.error_code.val}")
            return False
            
        self.get_logger().info(f"OMPL [{arm_group}]: Planowanie zakończone sukcesem")
        return True

    def _ompl_pose(self, pose: Pose) -> bool:
        goal = MoveGroup.Goal()
        goal.request.group_name = ARM_GROUP
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 5
        goal.request.planner_id = "RRTConnect"

        constraints = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = 'arm_right_tool_link'
        pc.weight = 1.0
        
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]
        
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv_pose = Pose()
        bv_pose.position = pose.position
        bv_pose.orientation.w = 1.0
        bv.primitive_poses.append(bv_pose)
        pc.constraint_region = bv
        constraints.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = 'arm_right_tool_link'
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)
        
        goal.request.goal_constraints.append(constraints)
        res = self._send_action(self.action_clients['move'], goal)
        return res is not False and res.result.error_code.val == 1

    def _cartesian(self, end_pose: Pose) -> bool:
        self._wait_for_joints()
        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.group_name = ARM_GROUP
        req.link_name = 'arm_right_tool_link'
        req.waypoints = [end_pose]
        req.max_step = CART_STEP
        req.jump_threshold = CART_JUMP
        req.avoid_collisions = False
        req.start_state = self._robot_state()

        future = self.cart_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        res = future.result()

        self.get_logger().info(f"Cartesian: {res.fraction*100:.1f}% zaplanowane")
        if res.fraction < MIN_FRACTION:
            return False
        return self._execute_trajectory(res.solution.joint_trajectory)

    def _execute_trajectory(self, traj) -> bool:
        torso_names = ['torso_lift_joint']
        arm_names = [j for j in traj.joint_names if j not in torso_names]

        def filter_traj(names):
            idx = [traj.joint_names.index(j) for j in names if j in traj.joint_names]
            if not idx: return None
            new_traj = JointTrajectory()
            new_traj.header = traj.header
            new_traj.joint_names = [traj.joint_names[i] for i in idx]
            for pt in traj.points:
                npt = JointTrajectoryPoint()
                npt.positions = [pt.positions[i] for i in idx]
                npt.velocities = [pt.velocities[i] for i in idx] if pt.velocities else []
                npt.accelerations = [pt.accelerations[i] for i in idx] if pt.accelerations else []
                npt.time_from_start = pt.time_from_start
                new_traj.points.append(npt)
            return new_traj

        torso_traj = filter_traj(torso_names)
        if torso_traj:
            torso_goal = FollowJointTrajectory.Goal()
            torso_goal.trajectory = torso_traj
            self._send_action(self.action_clients['torso'], torso_goal, wait_for_result=False)

        arm_traj = filter_traj(arm_names)
        if arm_traj:
            arm_goal = FollowJointTrajectory.Goal()
            arm_goal.trajectory = arm_traj
            res = self._send_action(self.action_clients['arm'], arm_goal)
            return res is not False
        return True

    def _scene_add_box(self):
        obj = CollisionObject()
        obj.id = BOX_ID
        obj.header.frame_id = BASE_FRAME
        obj.operation = CollisionObject.ADD
        
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = list(BOX_SIZE)
        obj.primitives.append(prim)
        
        p = Pose()
        p.position.x, p.position.y, p.position.z = BOX_CENTER
        p.orientation.w = 1.0
        obj.primitive_poses.append(p)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        for _ in range(5):
            self.scene_pub.publish(scene)
            self._spin_sleep(0.1)
        self.get_logger().info(f"Stół zmapowany w MoveIt: {BOX_SIZE}")

    def run(self):
        self.get_logger().info("=== URUCHOMIENIE PEŁNEGO CYKLU OPERACYJNEGO ===")
        
        self.ik_client.wait_for_service()
        self.cart_client.wait_for_service()
        for c in self.action_clients.values(): 
            c.wait_for_server()
        self._wait_for_joints()
        self.get_logger().info("Wszystkie podsystemy połączone.")

        approach_pose = self._make_pose(GRASP_X, GRASP_Y, APPROACH_Z)
        grasp_pose = self._make_pose(GRASP_X, GRASP_Y, GRASP_Z)

        steps = [
            ("KROK 0: Dodawanie stołu", lambda: self._scene_add_box() or True),
            # UŻYCIE OMPL DLA LEWEGO RAMIENIA - automatyczne omijanie kolizji
            ("KROK 0.1: Lewe ramię HOME (OMPL)", lambda: self._ompl(LEFT_HOME_JOINTS, LEFT_ARM_GROUP)),
            ("KROK 0.2: Prawe ramię HOME (OMPL)", lambda: self._ompl(HOME_JOINTS, ARM_GROUP)),
            ("KROK 1: Dojazd nad stół (OMPL Pose)", lambda: self._ompl_pose(approach_pose)),
            ("KROK 2: Otwarcie chwytaka", lambda: self._move_direct('gripper', GRIPPER_JOINTS, GRIPPER_OPEN, 2.0)),
            ("KROK 3: Zjazd Cartesian (Grasp)", lambda: self._cartesian(grasp_pose)),
            ("KROK 4: Zamknięcie chwytaka", lambda: self._move_direct('gripper', GRIPPER_JOINTS, GRIPPER_CLOSED, 2.0)),
            ("KROK 5: Wjazd Cartesian (Approach)", lambda: self._cartesian(approach_pose) or self._ompl_pose(approach_pose)),
            ("KROK 6: Pozycja podania (Handover)", lambda: self._ompl(HANDOVER_JOINTS, ARM_GROUP)),
            ("KROK 7: Otwarcie chwytaka (Release)", lambda: self._move_direct('gripper', GRIPPER_JOINTS, GRIPPER_OPEN, 2.0)),
            ("KROK 8: Powrót do bazy (OMPL)", lambda: self._ompl(HOME_JOINTS, ARM_GROUP))
        ]

        for desc, action in steps:
            self.get_logger().info(desc)
            if not action():
                self.get_logger().error(f"PRZERWANIE na etapie: {desc}")
                return
            self._spin_sleep(0.5)

        self.get_logger().info("=== CYKL OPERACYJNY ZAKOŃCZONY SUKCESEM ===")

def main():
    rclpy.init()
    node = PickAndPlace()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()