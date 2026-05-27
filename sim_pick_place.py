#!/usr/bin/env python3
"""
Smart Pick & Place dla TIAGo Pro – ROS 2 / MoveIt 2
Wersja nasłuchująca pozycji obiektu z topicu /detected_tool_pose
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

import tf2_ros
from tf2_geometry_msgs import do_transform_pose


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────
R = math.radians

HOME_JOINTS = [0.15, R(-176), R(-53), R(95), R(31), R(-7), R(-69), R(12)]
LEFT_HOME_JOINTS = [R(145), R(-87), R(-35), R(-68), R(-38), R(-26), R(10)]
HANDOVER_JOINTS = [0.15, R(-74), R(-78), R(117), R(-117), R(191), R(-91), R(0)]

BOX_ID = 'sim_table'
BOX_SIZE = (0.80, 1.43, 0.03)
BOX_CENTER = (0.9, 0.00, 0.735)

# Stała wysokość chwytania - obiekt leży na stole
TABLE_TOP_Z = BOX_CENTER[2] + BOX_SIZE[2]/2  # 0.75 m
TOOL_HEIGHT_ABOVE_TABLE = 0.20               # 5 cm nad blatem
GRASP_Z = TABLE_TOP_Z + TOOL_HEIGHT_ABOVE_TABLE

ARM_GROUP = 'arm_right_torso'
LEFT_ARM_GROUP = 'arm_left'
BASE_FRAME = 'base_footprint'
ARM_JOINTS = ['torso_lift_joint'] + [f'arm_right_{i}_joint' for i in range(1, 8)]
LEFT_ARM_JOINTS = [f'arm_left_{i}_joint' for i in range(1, 8)]
GRIPPER_JOINTS = ['gripper_right_finger_joint']
GRIPPER_OPEN, GRIPPER_CLOSED = [0.05], [0.80]

PLANNING_TIME, CART_STEP, CART_JUMP, MIN_FRACTION = 10.0, 0.005, 0.0, 0.90

# Temat z percepcji
DETECTED_POSE_TOPIC = '/detected_tool_pose'
DETECTION_TIMEOUT = 30.0  # sekundy czekania na detekcję


class SmartPickAndPlace(Node):
    def __init__(self):
        super().__init__('smart_pick_and_place_node')

        # ── MoveIt ──
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

        # ── TF2 ──
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subskrypcja na detekcje z kamery ──
        self.latest_detection = None
        self.detection_sub = self.create_subscription(
            PoseStamped,
            DETECTED_POSE_TOPIC,
            self._detection_cb,
            10
        )
        self.get_logger().info(f"Nasłuchuję detekcji na: {DETECTED_POSE_TOPIC}")

    # ─────────────────── CALLBACK DETEKCJI ───────────────────
    def _detection_cb(self, msg: PoseStamped):
        """Zapisuje najnowszą detekcję. Węzeł kamery publikuje tu ciągle."""
        self.latest_detection = msg
        self.get_logger().info(
            f"Detekcja: frame={msg.header.frame_id} "
            f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})",
            throttle_duration_sec=2.0
        )

    def _wait_for_detection(self, timeout=DETECTION_TIMEOUT):
        """Blokuje węzeł do momentu otrzymania detekcji lub timeoutu."""
        self.get_logger().info(f"Czekam na detekcję obiektu (max {timeout}s)...")
        deadline = time.time() + timeout
        while self.latest_detection is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_detection

    def _transform_to_base(self, pose_stamped: PoseStamped) -> PoseStamped:
        """Transformuje PoseStamped do układu base_footprint."""
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                pose_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=2.0)
            )
            
            # W ROS 2 do_transform_pose oczekuje obiektu Pose, a nie PoseStamped!
            transformed_pose = do_transform_pose(pose_stamped.pose, transform)
            
            # Pakujemy wynik z powrotem do PoseStamped
            result = PoseStamped()
            result.header.frame_id = BASE_FRAME
            result.header.stamp = pose_stamped.header.stamp
            result.pose = transformed_pose
            return result
            
        except Exception as e:
            self.get_logger().error(f"Błąd transformacji TF2: {e}")
            return None

    def _extract_yaw(self, quat) -> float:
        """Wyciąga kąt yaw (rotacja wokół Z) z kwaterniona."""
        # Wzór na yaw z kwaterniona
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _make_grasp_quaternion(self, yaw: float) -> tuple:
        """
        Tworzy kwaternion dla chwytaka:
        - roll = 0
        - pitch = π (chwytak pionowo w dół)
        - yaw = z detekcji kamery (obrót wokół osi narzędzia)
        """
        roll = 0.0
        pitch = math.pi  # gripper pointing down

        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def _process_detection(self, detection: PoseStamped):
        """
        Przetwarza detekcję z kamery na finalną pozycję chwytania w base_footprint.
        
        Kroki:
        1. Wyciągnij yaw z kwaterniona kamery
        2. Przelicz pozycję na Z = GRASP_Z (stała wysokość nad stołem)
        3. Transformuj do base_footprint
        4. Skonstruuj nowy kwaternion (pitch=π + yaw)
        """
        # 1. Wyciągamy yaw z detekcji kamery
        yaw_camera = self._extract_yaw(detection.pose.orientation)
        self.get_logger().info(f"Kąt z kamery: {math.degrees(yaw_camera):.1f}°")

        # 2. Modyfikujemy pozycję - Z = stała wysokość nad stołem
        adjusted = PoseStamped()
        adjusted.header = detection.header
        adjusted.pose.position.x = detection.pose.position.x
        adjusted.pose.position.y = detection.pose.position.y
        adjusted.pose.position.z = GRASP_Z  # Nadpisujemy Z na stałą wysokość!
        adjusted.pose.orientation.w = 1.0   # Tymczasowa orientacja

        # 3. Transformacja do base_footprint
        base_pose = self._transform_to_base(adjusted)
        if base_pose is None:
            return None

        # 4. Budujemy właściwy kwaternion (pitch=π, yaw z kamery)
        qx, qy, qz, qw = self._make_grasp_quaternion(yaw_camera)

        final = PoseStamped()
        final.header.frame_id = BASE_FRAME
        final.pose.position = base_pose.pose.position
        final.pose.orientation.x = qx
        final.pose.orientation.y = qy
        final.pose.orientation.z = qz
        final.pose.orientation.w = qw

        self.get_logger().info(
            f"Finalny cel w {BASE_FRAME}:\n"
            f"  Pozycja: ({final.pose.position.x:.3f}, {final.pose.position.y:.3f}, {final.pose.position.z:.3f})\n"
            f"  Yaw: {math.degrees(yaw_camera):.1f}°"
        )
        return final

    # ─────────────────── MOVEIT UTILS ───────────────────
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
        rs.joint_state = JointState(
            name=list(self._joints.keys()),
            position=list(self._joints.values())
        )
        return rs

    def _get_ik(self, pose: Pose, max_attempts: int = 5):
        """IK z pętlą retry (w ROS 2 brak pola 'attempts')."""
        self._wait_for_joints()
        for attempt in range(max_attempts):
            req = GetPositionIK.Request()
            req.ik_request.group_name = ARM_GROUP
            req.ik_request.pose_stamped = PoseStamped()
            req.ik_request.pose_stamped.header.frame_id = BASE_FRAME
            req.ik_request.pose_stamped.pose = pose
            req.ik_request.timeout.sec = 2
            req.ik_request.robot_state = self._robot_state()

            future = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            res = future.result()

            if res.error_code.val == 1:
                return [p for n, p in zip(res.solution.joint_state.name,
                                          res.solution.joint_state.position)
                        if 'arm_right' in n or 'torso' in n]
        return None

    def _validate_reachability(self, pose: Pose) -> bool:
        ik = self._get_ik(pose)
        if ik is None:
            self.get_logger().error(
                f"Pozycja NIEOSIĄGALNA: "
                f"({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
            )
            return False
        return True

    def _build_traj_goal(self, joint_names, positions, duration_sec=3.0):
        traj = JointTrajectory()
        traj.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(seconds=duration_sec).to_msg()
        traj.points.append(pt)
        return FollowJointTrajectory.Goal(trajectory=traj)

    def _move_direct(self, client_key, joint_names, positions, duration=3.0):
        goal = self._build_traj_goal(joint_names, positions, duration)
        res = self._send_action(self.action_clients[client_key], goal)
        self._spin_sleep(0.5)
        return res is not False

    def _ompl(self, joints: list, arm_group: str) -> bool:
        goal = MoveGroup.Goal()
        goal.request.group_name = arm_group
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 10
        goal.request.planner_id = "RRTConnect"

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

        res = self._send_action(self.action_clients['move'], goal)
        if not res or res.result.error_code.val != 1:
            self.get_logger().error(f"OMPL [{arm_group}] Błąd: {res.result.error_code.val if res else 'Odrzucono'}")
            return False
        return True

    def _ompl_pose(self, pose: Pose) -> bool:
        goal = MoveGroup.Goal()
        goal.request.group_name = ARM_GROUP
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.num_planning_attempts = 10
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
        if not res or res.result.error_code.val != 1:
            self.get_logger().error(f"OMPL Pose Błąd: {res.result.error_code.val if res else 'Odrzucono'}")
            return False
        return True

    def _cartesian(self, end_pose: Pose) -> bool:
        self._wait_for_joints()
        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.group_name = ARM_GROUP
        req.link_name = 'arm_right_tool_link'
        req.waypoints = [end_pose]
        req.max_step = CART_STEP
        req.jump_threshold = CART_JUMP
        req.avoid_collisions = True
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
            if not idx:
                return None
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

    # ─────────────────── GŁÓWNA SEKWENCJA ───────────────────
    def run(self):
        self.get_logger().info("=== URUCHOMIENIE SMART PICK & PLACE ===")
        
        # Czekamy na serwisy
        self.ik_client.wait_for_service()
        self.cart_client.wait_for_service()
        for c in self.action_clients.values():
            c.wait_for_server()
        
        self._wait_for_joints()
        self.get_logger().info("Wszystkie podsystemy połączone.")
        
        # Czekamy na TF
        self.get_logger().info("Czekam na drzewo TF2...")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                self.tf_buffer.lookup_transform(BASE_FRAME, 'head_front_camera_color_optical_frame', rclpy.time.Time())
                break
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.2)
        
        # ── KROK 3: Przygotowanie (tylko raz na starcie) ──
        self._scene_add_box()
        if not self._ompl(LEFT_HOME_JOINTS, LEFT_ARM_GROUP):
            self.get_logger().error("Lewe ramię nie osiągnęło HOME!")
            return
        if not self._ompl(HOME_JOINTS, ARM_GROUP):
            self.get_logger().error("Prawe ramię nie osiągnęło HOME!")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # GŁÓWNA PĘTLA - czekaj na detekcję → wykonaj cykl → powtórz
        # ═══════════════════════════════════════════════════════════════════
        self.get_logger().info("🔄 Wchodzę w pętlę oczekiwania na detekcje...")
        
        while rclpy.ok():
            # Resetuj stan - czekamy na NOWĄ detekcję
            self.latest_detection = None
            
            # ── KROK 1: Odbierz detekcję z kamery ──
            detection = self._wait_for_detection()
            if detection is None:
                self.get_logger().warn("Timeout - brak detekcji. Kontynuuję nasłuchiwanie...")
                continue
            
            # ── KROK 2: Przetwórz detekcję na pozycję w base_footprint ──
            final_pose = self._process_detection(detection)
            if final_pose is None:
                self.get_logger().error("Nie udało się przetworzyć detekcji. Pomijam...")
                continue
            
            grasp_pose = final_pose.pose
            grasp_pose.position.z = GRASP_Z
            
            # Approach = 15 cm nad grasp
            approach_pose = Pose()
            approach_pose.position.x = grasp_pose.position.x
            approach_pose.position.y = grasp_pose.position.y
            approach_pose.position.z = grasp_pose.position.z + 0.15
            approach_pose.orientation = grasp_pose.orientation
            
            # Walidacja osiągalności
            if not self._validate_reachability(approach_pose):
                self.get_logger().error("Wykryty obiekt jest poza zasięgiem ramienia. Pomijam...")
                continue
            
            # ── KROK 4: Cykl Pick & Place ──
            steps = [
                ("Dojazd nad obiekt (OMPL Pose)", lambda: self._ompl_pose(approach_pose)),
                ("Otwarcie chwytaka", lambda: self._move_direct('gripper', GRIPPER_JOINTS, GRIPPER_OPEN, 2.0)),
                ("Zjazd Cartesian (Grasp)", lambda: self._cartesian(grasp_pose) or self._ompl_pose(grasp_pose)),
                ("Zamknięcie chwytaka", lambda: self._move_direct('gripper', GRIPPER_JOINTS, GRIPPER_CLOSED, 2.0)),
                ("Wjazd Cartesian (Approach)", lambda: self._cartesian(approach_pose) or self._ompl_pose(approach_pose)),
                ("Pozycja podania (Handover)", lambda: self._ompl(HANDOVER_JOINTS, ARM_GROUP)),
                ("Otwarcie chwytaka (Release)", lambda: self._move_direct('gripper', GRIPPER_JOINTS, GRIPPER_OPEN, 2.0)),
                ("Powrót do bazy", lambda: self._ompl(HOME_JOINTS, ARM_GROUP))
            ]
            
            cycle_success = True
            for desc, action in steps:
                self.get_logger().info(desc)
                if not action():
                    self.get_logger().error(f"❌ PRZERWANIE na etapie: {desc}")
                    cycle_success = False
                    break
                self._spin_sleep(0.5)
            
            if cycle_success:
                self.get_logger().info("✅ === CYKL ZAKOŃCZONY SUKCESEM === Czekam na kolejną detekcję...")
            else:
                self.get_logger().warn("⚠️ Cykl przerwany. Próbuję wrócić do HOME i kontynuuję...")
                self._ompl(HOME_JOINTS, ARM_GROUP)  # Bezpieczny powrót
        
        self.get_logger().info("=== ZAKOŃCZENIE PĘTLI ===")


def main():
    rclpy.init()
    node = SmartPickAndPlace()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()