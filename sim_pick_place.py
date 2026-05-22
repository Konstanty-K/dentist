#!/usr/bin/env python3
"""
Pick & Place dla TIAGo Pro – ROS 2 / MoveIt 2
Wersja z obsługą statycznego lewego ramienia oraz zsynchronizowanym stołem

Cykl pracy robota:
  0. Inicjalizacja: Ustawienie LEWEGO ramienia w stałej pozycji oraz wstawienie stołu
  0.1 OMPL: Wyjście prawego ramienia do bezpiecznej pozycji startowej (HOME)
  1. OMPL: Dojazd nad stół (APPROACH)
  2. Otwarcie chwytaka (GRIPPER_OPEN)
  3. Cartesian: Zjazd pionowy pod chwyt (GRASP)
  4. Zamknięcie chwytaka na obiekcie (GRIPPER_CLOSED)
  5. Cartesian: Pionowe podniesienie obiektu ze stołu
  6. OMPL: Przejście do pozycji podania operatorowi (HANDOVER)
  7. Otwarcie chwytaka (Przekazanie narzędzia)
  8. OMPL: Powrót prawego ramienia do bezpiecznej pozycji startowej (HOME)
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
                              JointConstraint, RobotState,
                              PositionConstraint, OrientationConstraint,
                              BoundingVolume)
from shape_msgs.msg import SolidPrimitive

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA POZYCJI (Cykl Pracy)
# ─────────────────────────────────────────────────────────────────────────────

# 1. Pozycja startowa PRAWEGO ramienia (Wartości w radianach)
HOME_JOINTS = [
    0.15,                # torso_lift_joint (podniesiony dla bezpieczeństwa)
    math.radians(-176),  # arm_right_1_joint
    math.radians(-53),   # arm_right_2_joint
    math.radians(95),    # arm_right_3_joint
    math.radians(31),    # arm_right_4_joint
    math.radians(-7),    # arm_right_5_joint
    math.radians(-69),   # arm_right_6_joint
    math.radians(12)     # arm_right_7_joint
]

# STAŁA pozycja dla LEWEGO ramienia (Wartości w radianach)
LEFT_HOME_JOINTS = [
    math.radians(145),   # arm_left_1_joint
    math.radians(-87),   # arm_left_2_joint
    math.radians(-35),   # arm_left_3_joint
    math.radians(-68),   # arm_left_4_joint
    math.radians(-38),   # arm_left_5_joint
    math.radians(-26),   # arm_left_6_joint
    math.radians(10)     # arm_left_7_joint
]

# 2. Punkt chwytania na stole (Wycentrowany)
GRASP_X = 0.80
GRASP_Y = 0.00
GRASP_Z = 1.0   

# Punkt dojazdowy – 10 cm nad punktem chwytania
APPROACH_Z = GRASP_Z + 0.10   

# 3. Pozycja PODANIA narzędzia przez prawe ramię (Wartości w radianach)
HANDOVER_JOINTS = [
    0.15,                # torso_lift_joint
    math.radians(-74),   # arm_right_1_joint
    math.radians(-78),   # arm_right_2_joint
    math.radians(117),   # arm_right_3_joint
    math.radians(-117),  # arm_right_4_joint
    math.radians(191),   # arm_right_5_joint
    math.radians(-91),   # arm_right_6_joint
    math.radians(0)      # arm_right_7_joint
]

# Orientacja chwytaka: pionowo w dół dla wszystkich etapów
ORI_X = 0.0
ORI_Y = 1.0
ORI_Z = 0.0
ORI_W = 0.0

# Prostopadłościan (stół) w MoveIt - zsynchronizowany z Gazebo table.sdf
BOX_ID     = 'sim_table'
BOX_SIZE   = (0.80, 1.43, 0.03)
BOX_CENTER = (0.9, 0.00, 0.735)

# Nazwy stawów i grupy dla prawego ramienia
ARM_GROUP      = 'arm_right_torso'  
BASE_FRAME     = 'base_footprint'
ARM_JOINTS     = ['torso_lift_joint',
                  'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                  'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
                  'arm_right_7_joint']

# Definicja stawów dla lewego ramienia
LEFT_ARM_JOINTS = ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
                   'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint',
                   'arm_left_7_joint']

# Kalibracja sprzętowa chwytaka (Radiany)
GRIPPER_JOINTS = ['gripper_right_finger_joint']
GRIPPER_OPEN   = [0.05]  # Maksymalne rozwarcie (~46 stopni)
GRIPPER_CLOSED = [0.80]  # Pewny zacisk

PLANNING_TIME  = 10.0
CART_STEP      = 0.005   
CART_JUMP      = 0.0     
MIN_FRACTION   = 0.90    


class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')

        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        self.move_client    = ActionClient(self, MoveGroup, '/move_action')
        self.arm_client     = ActionClient(self, FollowJointTrajectory, '/arm_right_controller/follow_joint_trajectory')
        self.arm_left_client= ActionClient(self, FollowJointTrajectory, '/arm_left_controller/follow_joint_trajectory')
        self.torso_client   = ActionClient(self, FollowJointTrajectory, '/torso_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_right_controller/follow_joint_trajectory')

        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        self._joints = {}
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

    def _js_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._joints[name] = pos

    def _wait_for_joints(self, timeout=3.0):
        deadline = time.time() + timeout
        while len(self._joints) < 5 and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _robot_state(self) -> RobotState:
        js = JointState()
        js.name     = list(self._joints.keys())
        js.position = list(self._joints.values())
        rs = RobotState()
        rs.joint_state = js
        return rs

    def _make_pose(self, x, y, z) -> Pose:
        p = Pose()
        p.position.x    = float(x)
        p.position.y    = float(y)
        p.position.z    = float(z)
        p.orientation.x = ORI_X
        p.orientation.y = ORI_Y
        p.orientation.z = ORI_Z
        p.orientation.w = ORI_W
        return p

    def _active_sleep(self, duration_sec):
        end_time = time.time() + duration_sec
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

    # ─────────────────────────────────────────────────────────────────────────
    # KONTROLA LEWEGO RAMIENIA (BEZPOŚREDNIA / STATYCZNA)
    # ─────────────────────────────────────────────────────────────────────────
    def _set_left_arm_home(self):
        traj = JointTrajectory()
        traj.joint_names = LEFT_ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions       = LEFT_HOME_JOINTS
        pt.time_from_start = Duration(seconds=3).to_msg()
        traj.points.append(pt)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        self.get_logger().info("Wysyłam rozkaz ustawienia lewego ramienia...")
        future = self.arm_left_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if gh.accepted:
            rclpy.spin_until_future_complete(self, gh.get_result_async())
        self._active_sleep(1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # IK & OMPL & CARTESIAN
    # ─────────────────────────────────────────────────────────────────────────
    def _get_ik(self, pose: Pose):
        self._wait_for_joints()
        req = GetPositionIK.Request()
        req.ik_request.group_name               = ARM_GROUP
        req.ik_request.pose_stamped             = PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = BASE_FRAME
        req.ik_request.pose_stamped.pose        = pose
        req.ik_request.timeout.sec              = 5
        req.ik_request.robot_state              = self._robot_state()

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()

        if res.error_code.val != 1:
            self.get_logger().error(f"IK błąd {res.error_code.val}")
            return None
        names = res.solution.joint_state.name
        positions = res.solution.joint_state.position
        return [p for n, p in zip(names, positions) if 'arm_right' in n or 'torso' in n]

    def _ompl(self, joints: list, arm) -> bool:
        goal = MoveGroup.Goal()
        goal.request.group_name            = arm
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 5

        constraints = Constraints()
        if arm == "arm_left":
            arm_joints = LEFT_ARM_JOINTS
        else:
            arm_joints = ARM_JOINTS

        for name, pos in zip(arm_joints, joints):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        # MIEJSCE 1: Błąd odrzucenia celu przez serwer akcji
        if not gh.accepted:
            self.get_logger().error(f"OMPL [{arm}]: Serwer akcji MoveIt całkowicie odrzucił żądanie ruchu!")
            return False
        
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        
        error_code = rf.result().result.error_code.val
        
        # MIEJSCE 2: Błąd planowania lub wykonania (1 to sukces w MoveIt)
        if error_code != 1:
            self.get_logger().error(f"OMPL [{arm}]: Planowanie/Wykonanie zakończone błędem. Kod błędu MoveIt: {error_code}")
            return False
        return True

    def _ompl_pose(self, pose: Pose) -> bool:
        from shape_msgs.msg import SolidPrimitive as SP
        goal = MoveGroup.Goal()
        goal.request.group_name            = ARM_GROUP
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 5

        constraints = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name       = 'arm_right_tool_link'

        sphere = SP()
        sphere.type       = SP.SPHERE
        sphere.dimensions = [0.01]   
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv_pose = Pose()
        bv_pose.position    = pose.position
        bv_pose.orientation.w = 1.0
        bv.primitive_poses.append(bv_pose)
        pc.constraint_region = bv
        pc.weight = 1.0
        constraints.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id   = BASE_FRAME
        oc.link_name         = 'arm_right_tool_link'
        oc.orientation       = pose.orientation
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)
        goal.request.goal_constraints.append(constraints)

        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh.accepted:
            return False

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        return rf.result().result.error_code.val == 1

    def _cartesian(self, end_pose: Pose) -> bool:
        self._wait_for_joints()
        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.group_name       = ARM_GROUP
        req.link_name        = 'arm_right_tool_link'
        req.waypoints        = [end_pose]
        req.max_step         = CART_STEP
        req.jump_threshold   = CART_JUMP
        req.avoid_collisions = False
        req.start_state      = self._robot_state()

        future = self.cart_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        res = future.result()

        frac = res.fraction
        self.get_logger().info(f"Cartesian: {frac*100:.1f}% zaplanowane")
        if frac < MIN_FRACTION:
            return False

        return self._execute_trajectory(res.solution.joint_trajectory)

    def _execute_trajectory(self, traj) -> bool:
        torso_joints = ['torso_lift_joint']
        arm_joints   = [j for j in traj.joint_names if j not in torso_joints]
        has_torso    = any(j in torso_joints for j in traj.joint_names)

        arm_traj = JointTrajectory()
        arm_traj.header     = traj.header
        arm_traj.joint_names = arm_joints
        arm_idx = [traj.joint_names.index(j) for j in arm_joints]

        for pt in traj.points:
            npt = JointTrajectoryPoint()
            npt.positions     = [pt.positions[i]     for i in arm_idx]
            npt.velocities    = [pt.velocities[i]    for i in arm_idx] if pt.velocities    else []
            npt.accelerations = [pt.accelerations[i] for i in arm_idx] if pt.accelerations else []
            npt.time_from_start = pt.time_from_start
            arm_traj.points.append(npt)

        arm_goal = FollowJointTrajectory.Goal()
        arm_goal.trajectory = arm_traj

        torso_future = None
        if has_torso:
            torso_idx = [traj.joint_names.index(j) for j in torso_joints if j in traj.joint_names]
            torso_traj = JointTrajectory()
            torso_traj.header      = traj.header
            torso_traj.joint_names = [traj.joint_names[i] for i in torso_idx]

            for pt in traj.points:
                npt = JointTrajectoryPoint()
                npt.positions     = [pt.positions[i]     for i in torso_idx]
                npt.velocities    = [pt.velocities[i]    for i in torso_idx] if pt.velocities    else []
                npt.accelerations = [pt.accelerations[i] for i in torso_idx] if pt.accelerations else []
                npt.time_from_start = pt.time_from_start
                torso_traj.points.append(npt)

            torso_goal = FollowJointTrajectory.Goal()
            torso_goal.trajectory = torso_traj
            torso_future = self.torso_client.send_goal_async(torso_goal)

        arm_future = self.arm_client.send_goal_async(arm_goal)
        rclpy.spin_until_future_complete(self, arm_future)
        arm_gh = arm_future.result()
        if not arm_gh.accepted:
            return False

        if torso_future is not None:
            rclpy.spin_until_future_complete(self, torso_future)

        rf = arm_gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        return True

    def _gripper(self, position: list):
        traj = JointTrajectory()
        traj.joint_names = GRIPPER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions       = position
        pt.time_from_start = Duration(seconds=2).to_msg()
        traj.points.append(pt)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        future = self.gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if gh.accepted:
            rclpy.spin_until_future_complete(self, gh.get_result_async())
        self._active_sleep(1.0)

    def _scene_add_box(self):
        obj = CollisionObject()
        obj.id              = BOX_ID
        obj.header.frame_id = BASE_FRAME
        obj.operation       = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type       = SolidPrimitive.BOX
        prim.dimensions = list(BOX_SIZE)
        obj.primitives.append(prim)

        pose = Pose()
        pose.position.x    = BOX_CENTER[0]
        pose.position.y    = BOX_CENTER[1]
        pose.position.z    = BOX_CENTER[2]
        pose.orientation.w = 1.0
        obj.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        for _ in range(5):
            self.scene_pub.publish(scene)
            self._active_sleep(0.1)
        self.get_logger().info(f"Stół zmapowany w MoveIt: {BOX_SIZE}")

    # ─────────────────────────────────────────────────────────────────────────
    # GŁÓWNA SEKWENCJA CYKLU
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        self.get_logger().info("=== URUCHOMIENIE PEŁNEGO CYKLU OPERACYJNEGO ===")

        self.get_logger().info("Czekam na kontrolery i serwisy...")
        self.ik_client.wait_for_service()
        self.cart_client.wait_for_service()
        self.arm_client.wait_for_server()
        self.arm_left_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.move_client.wait_for_server()
        self._wait_for_joints()
        self.get_logger().info("Wszystkie podsystemy połączone.")

        approach_pose = self._make_pose(GRASP_X, GRASP_Y, APPROACH_Z)
        grasp_pose    = self._make_pose(GRASP_X, GRASP_Y, GRASP_Z)

        # KROK 0: Przygotowanie otoczenia i lewego ramienia
        self.get_logger().info("KROK 0: Dodawanie wirtualnego stołu do sceny planowania")
        self._scene_add_box()
        
        self.get_logger().info("KROK 0.1: Ustawianie LEWEGO ramienia w stałej pozycji spoczynkowej")
        if not self._ompl(LEFT_HOME_JOINTS, "arm_left"):
            self.get_logger().error("Lewe ramię nie osiągnęło pozycji HOME! Przerywam.")
            return
        self._active_sleep(1.0)

        self.get_logger().info("KROK 0.2: Wyjście PRAWEGO ramienia do pozycji HOME_JOINTS")
        if not self._ompl(HOME_JOINTS, "arm_right_torso"):
            self.get_logger().error("Prawe ramię nie osiągnęło pozycji HOME! Przerywam.")
            return
        self._active_sleep(1.0)

        # KROK 1: Ruch nad stół
        self.get_logger().info("KROK 1: OMPL → Dojazd nad punkt chwytu (Approach Pose)")
        if not self._ompl_pose(approach_pose):
            self.get_logger().error("Robot nie potrafi dojechać nad stół! Abort.")
            return
        self._active_sleep(1.0)

        # KROK 2: Przygotowanie chwytaka
        self.get_logger().info("KROK 2: Otwieranie chwytaka")
        self._gripper(GRIPPER_OPEN)

        # KROK 3: Precyzyjny zjazd kartezjański
        self.get_logger().info("KROK 3: Cartesian zjazd → Pobranie narzędzia ze stołu")
        if not self._cartesian(grasp_pose):
            self.get_logger().error("Zjazd liniowy zablokowany! Abort.")
            return
        self._active_sleep(0.5)

        # KROK 4: Chwyt obiektu
        self.get_logger().info("KROK 4: Zamykanie chwytaka na narzędziu")
        self._gripper(GRIPPER_CLOSED)

        # KROK 5: Wyjazd w górę
        self.get_logger().info("KROK 5: Cartesian wjazd → Podniesienie narzędzia w górę")
        if not self._cartesian(approach_pose):
            self.get_logger().warn("Wjazd liniowy przerwany – nadrabiam przez OMPL")
            self._ompl_pose(approach_pose)
        self._active_sleep(1.0)

        # KROK 6: Podanie przedmiotu (Handover)
        self.get_logger().info("KROK 6: OMPL → Ruch do pozycji PODANIA (HANDOVER_JOINTS)")
        if not self._ompl(HANDOVER_JOINTS, "arm_right_torso"):
            self.get_logger().error("Nie można wyznaczyć ścieżki do pozycji podania!")
            return
        self._active_sleep(1.0)

        # KROK 7: Przekazanie operatorowi
        self.get_logger().info("KROK 7: Otwarcie chwytaka – Przekazanie narzędzia")
        self._gripper(GRIPPER_OPEN)

        # KROK 8: Powrót prawego ramienia do bazy
        self.get_logger().info("KROK 8: Powrót prawego ramienia do pozycji bezpiecznej (HOME_JOINTS)")
        self._ompl(HOME_JOINTS)

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