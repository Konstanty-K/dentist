#!/usr/bin/env python3
"""
Pick & Place dla TIAGo Pro – ROS 2 / MoveIt 2

Sekwencja:
  1. OMPL: pozycja startowa → approach_pose (10 cm nad obiektem)
  2. Otwórz chwytak
  3. Cartesian: approach_pose → grasp_pose (zjazd w dół)
  4. Zamknij chwytak
  5. Cartesian: grasp_pose → approach_pose (wjazd w górę)
  6. OMPL: approach_pose → place_pose
  7. Otwórz chwytak

Współrzędne dobrane na podstawie testu IK:
  - z=0.88 (grasp) i z=0.98 (approach) są w zasięgu ramienia
  - orientacja (0, 0.7071, 0, 0.7071) = chwytak pionowo w dół
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import time

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
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

# Punkt chwytania – górna powierzchnia prostopadłościanu
GRASP_X = 0.55
GRASP_Y = -0.20
GRASP_Z = 0.88   # z=0.85+ działa wg testu IK

# Punkt dojazdowy – 10 cm nad punktem chwytania
APPROACH_Z = GRASP_Z + 0.10   # = 0.98

# Punkt odłożenia
PLACE_X = 0.55
PLACE_Y = -0.50
PLACE_Z = 0.98

# Orientacja chwytaka: pionowo w dół
# Źródło: move_to_target.cpp z pracy inżynierskiej tego samego robota
# (x=0, y=1, z=0, w=0) = obrót 180° wokół Y = chwytak skierowany w dół
ORI_X = 0.0
ORI_Y = 1.0
ORI_Z = 0.0
ORI_W = 0.0

# Prostopadłościan w MoveIt planning scene
BOX_ID     = 'target_box'
BOX_SIZE   = (0.10, 0.10, 0.80)           # x, y, z [m]
BOX_CENTER = (GRASP_X, GRASP_Y,
              GRASP_Z - BOX_SIZE[2] / 2)  # środek = góra - połowa wysokości

# Parametry MoveIt
ARM_GROUP      = 'arm_right_torso'  # z torsem = większy zasięg, lepsza konfiguracja
BASE_FRAME     = 'base_footprint'
ARM_JOINTS     = ['torso_lift_joint',
                  'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                  'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
                  'arm_right_7_joint']

# APPROACH_JOINTS będzie wyznaczone przez IK (arm_right_torso ma inną konfigurację)
# Używamy PoseGoal przez OMPL zamiast hardkodowanych stawów
APPROACH_JOINTS = None  # placeholder – nie używany przy arm_right_torso
GRIPPER_JOINTS = ['gripper_right_finger_joint']
GRIPPER_OPEN   = [0.02]
GRIPPER_CLOSED = [0.15]
PLANNING_TIME  = 10.0
CART_STEP      = 0.005   # krok interpolacji kartezjańskiej [m]
CART_JUMP      = 0.0     # wyłączony próg skoku
MIN_FRACTION   = 0.90    # minimalny % ścieżki kartezjańskiej


class PickAndPlace(Node):

    def __init__(self):
        super().__init__('pick_and_place_node')

        # Serwisy
        self.ik_client  = self.create_client(GetPositionIK, '/compute_ik')
        self.cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        # Akcje
        self.move_client    = ActionClient(self, MoveGroup, '/move_action')
        self.arm_client     = ActionClient(self, FollowJointTrajectory,
                                           '/arm_right_controller/follow_joint_trajectory')
        self.torso_client   = ActionClient(self, FollowJointTrajectory,
                                           '/torso_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory,
                                           '/gripper_right_controller/follow_joint_trajectory')

        # Publisher sceny
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # Stan stawów (potrzebny do IK i Cartesian)
        self._joints = {}
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

    def _js_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._joints[name] = pos

    def _wait_for_joints(self, timeout=3.0):
        """Czeka aż /joint_states zacznie publikować."""
        deadline = time.time() + timeout
        while len(self._joints) < 5 and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _robot_state(self) -> RobotState:
        """Zwraca aktualny RobotState z /joint_states."""
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

    # ─────────────────────────────────────────────────────────────────────────
    # IK
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ik(self, pose: Pose):
        """Zwraca listę pozycji stawów arm_right lub None."""
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
            self.get_logger().error(
                f"IK błąd {res.error_code.val} dla "
                f"({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
            )
            return None

        names = res.solution.joint_state.name
        positions = res.solution.joint_state.position
        return [p for n, p in zip(names, positions) if 'arm_right' in n or 'torso' in n]

    # ─────────────────────────────────────────────────────────────────────────
    # OMPL
    # ─────────────────────────────────────────────────────────────────────────

    def _ompl(self, joints: list) -> bool:
        """Planuje i wykonuje ruch do zadanej konfiguracji stawów przez OMPL."""
        goal = MoveGroup.Goal()
        goal.request.group_name            = ARM_GROUP
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 5

        constraints = Constraints()
        for name, pos in zip(ARM_JOINTS, joints):
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
        if not gh.accepted:
            self.get_logger().error("OMPL: cel odrzucony")
            return False

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        code = rf.result().result.error_code.val
        if code != 1:
            self.get_logger().error(f"OMPL: błąd kod {code}")
            return False
        return True

    def _ompl_pose(self, pose: Pose) -> bool:
        """
        Planuje ruch do zadanej pozycji przez OMPL używając PoseGoal.
        Lepsze niż JointGoal bo MoveIt dobiera konfigurację stawów
        zgodną z późniejszą ścieżką kartezjańską.
        """
        from shape_msgs.msg import SolidPrimitive as SP
        goal = MoveGroup.Goal()
        goal.request.group_name            = ARM_GROUP
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 5

        constraints = Constraints()

        # Constraint pozycji – sfera 1cm wokół celu
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name       = 'arm_right_tool_link'
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0

        sphere = SP()
        sphere.type       = SP.SPHERE
        sphere.dimensions = [0.01]   # tolerancja 1 cm
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv_pose = Pose()
        bv_pose.position    = pose.position
        bv_pose.orientation.w = 1.0
        bv.primitive_poses.append(bv_pose)
        pc.constraint_region = bv
        pc.weight = 1.0
        constraints.position_constraints.append(pc)

        # Constraint orientacji
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
            self.get_logger().error("OMPL pose: cel odrzucony")
            return False

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        code = rf.result().result.error_code.val
        if code != 1:
            self.get_logger().error(f"OMPL pose: błąd kod {code}")
            return False
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # CARTESIAN PATH
    # ─────────────────────────────────────────────────────────────────────────

    def _cartesian(self, end_pose: Pose) -> bool:
        """
        Ruch po linii prostej do end_pose.
        Trajektoria z arm_right_torso zawiera torso_lift_joint i stawy ramienia –
        rozdzielamy ją i wysyłamy do dwóch kontrolerów równolegle.
        """
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
            self.get_logger().error(f"Cartesian: za mała frakcja ({frac*100:.1f}%)")
            return False

        return self._execute_trajectory(res.solution.joint_trajectory)

    def _execute_trajectory(self, traj) -> bool:
        """
        Rozdziela trajektorię na część ramieniową i torsową,
        wysyła do odpowiednich kontrolerów i czeka na zakończenie obu.
        """
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        torso_joints = ['torso_lift_joint']
        arm_joints   = [j for j in traj.joint_names if j not in torso_joints]
        has_torso    = any(j in torso_joints for j in traj.joint_names)

        # ── Trajektoria ramienia ───────────────────────────────────────────────
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

        # ── Trajektoria torsu (jeśli istnieje) ───────────────────────────────
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

        # ── Wyślij i czekaj ───────────────────────────────────────────────────
        arm_future = self.arm_client.send_goal_async(arm_goal)

        rclpy.spin_until_future_complete(self, arm_future)
        arm_gh = arm_future.result()
        if not arm_gh.accepted:
            self.get_logger().error("Kontroler ramienia odrzucił trajektorię")
            return False

        if torso_future is not None:
            rclpy.spin_until_future_complete(self, torso_future)
            torso_gh = torso_future.result()
            if not torso_gh.accepted:
                self.get_logger().warn("Kontroler torsu odrzucił trajektorię (kontynuuję)")

        # Czekaj na zakończenie ramienia
        rf = arm_gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # CHWYTAK
    # ─────────────────────────────────────────────────────────────────────────

    def _gripper(self, position: list):
        traj = JointTrajectory()
        traj.joint_names = GRIPPER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions       = position
        pt.time_from_start = Duration(seconds=2).to_msg()
        traj.points.append(pt)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        # Wysyłamy asynchronicznie (Fire & Forget)
        self.gripper_client.send_goal_async(goal_msg)
        
        # Wymuszamy na Pythonie odczekanie dokładnie tyle, ile trwa fizyczny ruch palców w Gazebo.
        # Plus mały bufor (0.5s), żeby chwyt był pewny przed podniesieniem ramienia.
        time.sleep(2.5)

    # ─────────────────────────────────────────────────────────────────────────
    # PLANNING SCENE
    # ─────────────────────────────────────────────────────────────────────────

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
            time.sleep(0.1)
        self.get_logger().info(
            f"Box dodany: {BOX_SIZE} @ center {BOX_CENTER}"
        )

    def _scene_remove_box(self):
        obj = CollisionObject()
        obj.id              = BOX_ID
        obj.header.frame_id = BASE_FRAME
        obj.operation       = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        for _ in range(5):
            self.scene_pub.publish(scene)
            time.sleep(0.1)
        self.get_logger().info("Box usunięty ze sceny")

    # ─────────────────────────────────────────────────────────────────────────
    # GŁÓWNA SEKWENCJA
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        self.get_logger().info("=== START PICK & PLACE ===")

        # Czekaj na serwery
        self.get_logger().info("Czekam na kontrolery i serwisy...")
        self.ik_client.wait_for_service()
        self.cart_client.wait_for_service()
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.move_client.wait_for_server()
        self._wait_for_joints()
        self.get_logger().info(f"Gotowy. Stan stawów: {len(self._joints)} stawów")

        # Definicja pozycji
        approach_pose = self._make_pose(GRASP_X, GRASP_Y, APPROACH_Z)
        grasp_pose    = self._make_pose(GRASP_X, GRASP_Y, GRASP_Z)
        place_pose    = self._make_pose(PLACE_X,  PLACE_Y,  PLACE_Z)

        # ── KROK 1: IK dla approach_pose ─────────────────────────────────────
        self.get_logger().info(
            f"KROK 1: IK dla approach_pose ({GRASP_X}, {GRASP_Y}, {APPROACH_Z})"
        )
        approach_joints = self._get_ik(approach_pose)
        if approach_joints is None:
            self.get_logger().error("Brak IK dla approach_pose! Abort.")
            return

        # ── KROK 2: OMPL → approach_pose przez PoseGoal ─────────────────────
        self.get_logger().info("KROK 2: OMPL → approach_pose")
        if not self._ompl_pose(approach_pose):
            self.get_logger().warn("PoseGoal nie powiódł się – próbuję JointGoal")
            if approach_joints and not self._ompl(approach_joints):
                self.get_logger().error("OMPL nie powiódł się! Abort.")
                return
        time.sleep(1.0)

        # ── KROK 3: Otwórz chwytak ────────────────────────────────────────────
        self.get_logger().info("KROK 3: Otwieram chwytak")
        self._gripper(GRIPPER_OPEN)
        time.sleep(0.5)

        # ── KROK 4: Cartesian zjazd → grasp_pose ─────────────────────────────
        self.get_logger().info(
            f"KROK 4: Cartesian zjazd → grasp_pose ({GRASP_X}, {GRASP_Y}, {GRASP_Z})"
        )
        if not self._cartesian(grasp_pose):
            self.get_logger().error("Cartesian zjazd nie powiódł się! Abort.")
            return
        time.sleep(0.5)

        # ── KROK 5: Zamknij chwytak ───────────────────────────────────────────
        self.get_logger().info("KROK 5: Zamykam chwytak")
        self._gripper(GRIPPER_CLOSED)
        time.sleep(2.0)

        # ── KROK 6: Cartesian wjazd → approach_pose ──────────────────────────
        self.get_logger().info("KROK 6: Cartesian wjazd → approach_pose")
        # Odczekaj żeby joint_states zaktualizował stan po zamknięciu chwytaka
        time.sleep(1.0)
        rclpy.spin_once(self, timeout_sec=0.2)
        if not self._cartesian(approach_pose):
            self.get_logger().warn("Cartesian wjazd niepełny – próbuję OMPL PoseGoal")
            self._ompl_pose(approach_pose)
        time.sleep(0.5)

        # ── KROK 7: OMPL → place_pose ─────────────────────────────────────────
        self.get_logger().info(
            f"KROK 7: OMPL → place_pose ({PLACE_X}, {PLACE_Y}, {PLACE_Z})"
        )
        place_joints = self._get_ik(place_pose)
        if place_joints is None:
            self.get_logger().error("Brak IK dla place_pose!")
            return
        self._ompl(place_joints)
        time.sleep(0.5)

        # ── KROK 8: Otwórz chwytak ────────────────────────────────────────────
        self.get_logger().info("KROK 8: Otwieram chwytak – odkładam obiekt")
        self._gripper(GRIPPER_OPEN)

        self.get_logger().info("=== MISJA ZAKOŃCZONA ===")


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