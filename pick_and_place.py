#!/usr/bin/env python3
"""
Decision Pick & Place dla TIAGo Pro – ROS 2 / MoveIt 2
Wersja oparta WYŁĄCZNIE na danych z kamery, bez losowania.
Automatycznie wybiera lewą lub prawą rękę do chwytu w zależności od pozycji Y.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import time
import math

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (PlanningScene, CollisionObject, Constraints,
                             JointConstraint, RobotState, PositionConstraint,
                             OrientationConstraint, BoundingVolume)
from shape_msgs.msg import SolidPrimitive

import tf2_ros
from tf2_geometry_msgs import do_transform_pose

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA BAZOWA
# ─────────────────────────────────────────────────────────────────────────────
R = math.radians

# Pozycje startowe dla 7 stawów ramion (tułów całkowicie wyłączony)
RIGHT_HOME_JOINTS = [R(-176), R(-53), R(95), R(31), R(-7), R(-69), R(12)]
LEFT_HOME_JOINTS  = [R(145), R(-87), R(-35), R(-68), R(-38), R(-26), R(10)]

# Definicja stołu (wirtualnie obniżony środek, aby szerokie palce nie generowały kolizji)
BOX_ID = 'sim_table'
BOX_SIZE = (0.80, 1.43, 0.03)
BOX_CENTER = (0.90, 0.00, 0.735)

# Wysokość chwytania (TCP na końcówkach palców - zjeżdżamy tuż nad blat)
GRASP_Z = 0.95

BASE_FRAME = 'base_footprint'
PLANNING_TIME, CART_STEP, CART_JUMP, MIN_FRACTION = 10.0, 0.005, 0.0, 0.80

RIGHT_ARM_JOINTS = [f'arm_right_{i}_joint' for i in range(1, 8)]
LEFT_ARM_JOINTS = [f'arm_left_{i}_joint' for i in range(1, 8)]

# Konfiguracja Twoich chwytaków (0.0 = otwarty, 0.8 = zamknięty)
GRIPPER_OPEN, GRIPPER_CLOSED = [0.00], [0.80]

class DecisionPickAndPlace(Node):
    def __init__(self):
        super().__init__('decision_pick_and_place_node')
        
        # Inicjalizacja serwisów i akcji MoveIt
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        
        self.action_clients = {
            'move': ActionClient(self, MoveGroup, '/move_action'),
            'arm_right': ActionClient(self, FollowJointTrajectory, '/arm_right_controller/follow_joint_trajectory'),
            'arm_left': ActionClient(self, FollowJointTrajectory, '/arm_left_controller/follow_joint_trajectory'),
            'gripper_right': ActionClient(self, FollowJointTrajectory, '/gripper_right_controller/follow_joint_trajectory'),
            'gripper_left': ActionClient(self, FollowJointTrajectory, '/gripper_left_controller/follow_joint_trajectory')
        }
        
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._joints = {}
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.latest_detection = None
        self.create_subscription(PoseStamped, '/detected_tool_pose', self._det_cb, 10)
        self.get_logger().info("Węzeł uruchomiony. Czekam na dane z tematu /detected_tool_pose...")

    def _det_cb(self, msg):
        self.latest_detection = msg

    def _make_grasp_quaternion(self) -> tuple:
        pitch = math.pi
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        return 0.0, sp, 0.0, cp # Obrót chwytaka pionowo w dół

    def _transform_to_base(self, pose_stamped: PoseStamped) -> PoseStamped:
        try:
            if pose_stamped.header.frame_id == BASE_FRAME:
                return pose_stamped
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, pose_stamped.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=2.0))
            transformed_pose = do_transform_pose(pose_stamped.pose, transform)
            result = PoseStamped()
            result.header.frame_id = BASE_FRAME
            result.pose = transformed_pose
            return result
        except Exception as e:
            self.get_logger().error(f"Błąd transformacji układów TF2: {e}")
            return None

    def _js_cb(self, msg):
        self._joints.update(dict(zip(msg.name, msg.position)))

    def _wait_for_joints(self):
        while len(self._joints) < 5:
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

    def _build_traj_goal(self, joint_names, positions, duration_sec=2.0):
        traj = JointTrajectory()
        traj.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(seconds=duration_sec).to_msg()
        traj.points.append(pt)
        return FollowJointTrajectory.Goal(trajectory=traj)

    def _move_direct(self, client_key, joint_names, positions, duration=2.0):
        goal = self._build_traj_goal(joint_names, positions, duration)
        res = self._send_action(self.action_clients[client_key], goal)
        self._spin_sleep(0.2)
        return res is not False

    def _ompl(self, joints: list, arm_group: str) -> bool:
        goal = MoveGroup.Goal()
        goal.request.group_name = arm_group
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 10
        goal.request.planner_id = "RRTConnect"

        constraints = Constraints()
        target_joints = LEFT_ARM_JOINTS if arm_group == 'arm_left' else RIGHT_ARM_JOINTS
        for name, pos in zip(target_joints, joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above, jc.tolerance_below, jc.weight = 0.02, 0.02, 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        res = self._send_action(self.action_clients['move'], goal)
        return bool(res and res.result.error_code.val == 1)

    def _ompl_pose(self, pose: Pose, arm_group: str, tool_link: str) -> bool:
        goal = MoveGroup.Goal()
        goal.request.group_name = arm_group
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 10
        goal.request.planner_id = "RRTConnect"

        constraints = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = tool_link
        pc.weight = 1.0
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv_pose = Pose(position=pose.position, orientation=Pose().orientation)
        bv_pose.orientation.w = 1.0
        bv.primitive_poses.append(bv_pose)
        pc.constraint_region = bv
        constraints.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = tool_link
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)

        goal.request.goal_constraints.append(constraints)
        res = self._send_action(self.action_clients['move'], goal)
        return bool(res and res.result.error_code.val == 1)

    def _cartesian(self, end_pose: Pose, arm_group: str, tool_link: str) -> bool:
        self._wait_for_joints()
        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.group_name = arm_group
        req.link_name = tool_link
        req.waypoints = [end_pose]
        req.max_step = CART_STEP
        req.jump_threshold = CART_JUMP
        req.avoid_collisions = True
        req.start_state = self._robot_state()

        future = self.cart_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()

        if res.fraction < MIN_FRACTION:
            self.get_logger().warn(f"[{arm_group}] Cartesian zaplanował tylko {res.fraction*100:.1f}% trasy")
            return False
            
        client_key = 'arm_left' if arm_group == 'arm_left' else 'arm_right'
        arm_goal = FollowJointTrajectory.Goal()
        arm_goal.trajectory = res.solution.joint_trajectory
        return self._send_action(self.action_clients[client_key], arm_goal) is not False

    def _scene_add_box(self):
        obj = CollisionObject()
        obj.id = BOX_ID
        obj.header.frame_id = BASE_FRAME
        obj.operation = CollisionObject.ADD
        prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(BOX_SIZE))
        obj.primitives.append(prim)
        p = Pose()
        p.position.x, p.position.y, p.position.z = BOX_CENTER
        p.orientation.w = 1.0
        obj.primitive_poses.append(p)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        for _ in range(3):
            self.scene_pub.publish(scene)
            self._spin_sleep(0.1)

    def _run_sequence(self, target_pose, side):
        group = 'arm_right' if side == 'right' else 'arm_left'
        link = 'arm_right_tool_link' if side == 'right' else 'arm_left_tool_link'
        gripper = 'gripper_right' if side == 'right' else 'gripper_left'
        
        self.get_logger().info(f"==> Wykryto obiekt. Decyzja: Uruchamiam ramię {side.upper()}")
        
        qx, qy, qz, qw = self._make_grasp_quaternion()
        
        grasp_pose = Pose()
        grasp_pose.position.x = target_pose.position.x
        grasp_pose.position.y = target_pose.position.y
        grasp_pose.position.z = GRASP_Z
        grasp_pose.orientation.x = qx
        grasp_pose.orientation.y = qy
        grasp_pose.orientation.z = qz
        grasp_pose.orientation.w = qw

        app_pose = Pose()
        app_pose.position.x = grasp_pose.position.x
        app_pose.position.y = grasp_pose.position.y
        app_pose.position.z = grasp_pose.position.z + 0.15 # Pozycja dojazdowa 15 cm nad celem
        app_pose.orientation = grasp_pose.orientation

        steps = [
            (f"Dojazd nad obiekt ({side})", lambda: self._ompl_pose(app_pose, group, link)),
            (f"Otwarcie chwytaka ({side})", lambda: self._move_direct(gripper, [f'gripper_{side}_finger_joint'], GRIPPER_OPEN)),
            (f"Zjazd w dół ({side})", lambda: self._cartesian(grasp_pose, group, link) or self._ompl_pose(grasp_pose, group, link)),
            (f"Zamknięcie chwytaka ({side})", lambda: self._move_direct(gripper, [f'gripper_{side}_finger_joint'], GRIPPER_CLOSED)),
            (f"Podniesienie ze stołu ({side})", lambda: self._cartesian(app_pose, group, link) or self._ompl_pose(app_pose, group, link)),
            (f"Powrót do pozycji HOME ({side})", lambda: self._ompl(LEFT_HOME_JOINTS if side == 'left' else RIGHT_HOME_JOINTS, group))
        ]
        
        for desc, action in steps:
            self.get_logger().info(f"Status: {desc}")
            if not action():
                self.get_logger().error(f"❌ Przerwanie cyklu na etapie: {desc}. Awaryjny powrót do HOME.")
                self._ompl(LEFT_HOME_JOINTS if side == 'left' else RIGHT_HOME_JOINTS, group)
                break
        else:
            self.get_logger().info(f"✅ Chwyt ramieniem {side.upper()} wykonany pomyślnie!")

    def run(self):
        self._wait_for_joints()
        self._scene_add_box()
        self.get_logger().info("Ustawiam oba ramiona w bezpiecznych pozycjach startowych...")
        self._ompl(RIGHT_HOME_JOINTS, 'arm_right')
        self._ompl(LEFT_HOME_JOINTS, 'arm_left')
        self.get_logger().info("✅ Gotowy. Oczekuję na publikację współrzędnych celu na temat /detected_tool_pose...")
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_detection:
                det = self.latest_detection
                self.latest_detection = None # Czyszczenie bufora detekcji
                
                pose_stamped = self._transform_to_base(det)
                if not pose_stamped: 
                    continue
                
                # SYSTEM DECYZYJNY: Jeśli współrzędna Y jest dodatnia, cel leży po lewej stronie robota
                side = 'left' if pose_stamped.pose.position.y > 0 else 'right'
                
                self._run_sequence(pose_stamped.pose, side)
                self.get_logger().info("==============================================")
                self.get_logger().info("System wolny. Oczekuję na następny obiekt...")

def main():
    rclpy.init()
    node = DecisionPickAndPlace()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()