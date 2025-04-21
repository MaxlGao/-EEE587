import mujoco
import mujoco.viewer
import numpy as np
import cvxpy as cp
import time
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 

XML_PATH = "cube.xml"
SIM_DURATION = 25
CONTROL_TIMESTEP = 0.009
SLEEP_PER_STEP = 0.001

RELATIVE_PATH_VELOCITY = 0.10
FORWARD_DISTANCE = 0.2 
TURN_DIRECTION = 'right'
TURN_DISTANCE = 0.2
WAYPOINT_TOLERANCE = 0.03 

MU_P = 0.7
MU_G = 0.35
CUBE_MASS = 0.1
GRAVITY = 9.81

APPROACH_SPEED = 0.10
CONTACT_THRESHOLD = 0.01
FN_TO_INWARD_VEL_GAIN = 0.1
pusher_z_height = 0.05

MPC_HORIZON = 50
n_states = 6
n_controls = 2

Q = np.diag([
    400.0, 400.0, # x, y position error cost
    50.0,         # theta orientation error cost
    1.0,          # vx velocity error cost
    50.0,         # vy velocity error cost
    0.5           # vtheta angular velocity error cost
])
Q_N = Q * 20 # Terminal state cost multiplier (scales with Q)

R_cost = np.diag([0.01, 0.01])

max_fn = 5.0

ORIENTATION_ERROR_THRESHOLD = np.radians(40.0)

SITES_Y_PUSH = {"center": "site_back_center", "left_turn": "site_back_right", "right_turn": "site_back_left"}
SITES_X_PUSH = {"center": "site_left_center", "left_turn": "site_left_down", "right_turn": "site_left_up"}
ALL_SITE_NAMES = list(SITES_Y_PUSH.values()) + list(SITES_X_PUSH.values())
DESIRED_MARKER_NAME = "desired_pos_marker"

initial_cube_pos_xy = np.zeros(2)
initial_cube_theta = 0.0
initial_cube_rot_mat = np.eye(3)

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_relative_path_target(current_time, initial_pos, initial_theta):
    global initial_cube_rot_mat
    time_for_forward = FORWARD_DISTANCE / RELATIVE_PATH_VELOCITY if RELATIVE_PATH_VELOCITY > 1e-6 else float('inf')
    time_for_turn = TURN_DISTANCE / RELATIVE_PATH_VELOCITY if RELATIVE_PATH_VELOCITY > 1e-6 else float('inf')
    forward_dir_body = np.array([0., 1., 0.])
    forward_dir = initial_cube_rot_mat @ forward_dir_body; forward_dir = forward_dir[:2]
    norm_fwd = np.linalg.norm(forward_dir)
    if norm_fwd > 1e-6: forward_dir /= norm_fwd
    else: forward_dir = np.array([0., 1.])
    if TURN_DIRECTION.lower() == 'right':
        turn_angle_change = -np.pi / 2; turn_dir = np.array([forward_dir[1], -forward_dir[0]])
    else:
        turn_angle_change = np.pi / 2; turn_dir = np.array([-forward_dir[1], forward_dir[0]])
    final_theta = normalize_angle(initial_theta + turn_angle_change)
    if current_time < 0: current_time = 0
    if current_time < time_for_forward:
        dist_moved = current_time * RELATIVE_PATH_VELOCITY; pos = initial_pos + forward_dir * dist_moved
        theta = initial_theta; vel = forward_dir * RELATIVE_PATH_VELOCITY; vel_theta = 0.0
    elif current_time < time_for_forward + time_for_turn:
        time_in_turn_seg = current_time - time_for_forward; dist_moved_turn = time_in_turn_seg * RELATIVE_PATH_VELOCITY
        corner_pos = initial_pos + forward_dir * FORWARD_DISTANCE; pos = corner_pos + turn_dir * dist_moved_turn
        theta = final_theta; vel = turn_dir * RELATIVE_PATH_VELOCITY; vel_theta = 0.0
    else:
        corner_pos = initial_pos + forward_dir * FORWARD_DISTANCE; pos = corner_pos + turn_dir * TURN_DISTANCE
        theta = final_theta; vel = np.array([0.0, 0.0]); vel_theta = 0.0
    return pos, vel, theta, vel_theta

def calculate_reference_trajectory(current_sim_time, dt, N, initial_pos, initial_theta):
    ref_traj = np.zeros((n_states, N + 1))
    for i in range(N + 1):
        t_predict = current_sim_time + i * dt
        pos_ref, vel_ref, theta_ref, vtheta_ref = get_relative_path_target(t_predict, initial_pos, initial_theta)
        ref_traj[:, i] = [pos_ref[0], pos_ref[1], theta_ref, vel_ref[0], vel_ref[1], vtheta_ref]
    return ref_traj

def calculate_limit_surface_matrix(mu_g, mass, cube_size):
    f_max = mu_g * mass * GRAVITY
    if f_max <= 1e-6: return np.diag([1e-9]*3)
    half_width = cube_size[0]; half_height = cube_size[1]
    characteristic_radius = np.sqrt(half_width**2 + half_height**2)
    m_max = f_max * characteristic_radius
    if m_max <= 1e-9: L_inv_sq_diag = np.array([1/(f_max**2), 1/(f_max**2), 1e18])
    else: L_inv_sq_diag = np.array([1/(f_max**2), 1/(f_max**2), 1/(m_max**2)])
    L_diag = 1.0 / np.maximum(L_inv_sq_diag, 1e-12); L = np.diag(L_diag)
    return L

def get_contact_jacobian(contact_pos_body):
    px, py = contact_pos_body[0], contact_pos_body[1]
    Jc = np.array([[1., 0., -py],[0., 1.,  px]])
    return Jc

def get_contact_normal_tangent(site_name):
    if "back" in site_name: cn_body = np.array([0., -1.]); ct_body = np.array([1., 0.])
    elif "left" in site_name: cn_body = np.array([-1., 0.]); ct_body = np.array([0., 1.])
    else: cn_body = np.array([0., -1.]); ct_body = np.array([1., 0.])
    return cn_body, ct_body

def get_force_map_matrix_B(Jc, site_name):
    cn_body, ct_body = get_contact_normal_tangent(site_name)
    wrench_n_comp = Jc.T @ cn_body; wrench_t_comp = Jc.T @ ct_body
    B_map = np.vstack((wrench_n_comp, wrench_t_comp)).T
    return B_map

def get_linearized_dynamics_force_based(current_state, B_map_current, L_matrix, dt, mass, inertia_zz):
    A_d, B_d_back, B_d_side = get_kinematic_model_matrices(dt)
    return A_d, B_d_back 


def get_kinematic_model_matrices(dt):
     VEL_DECAY = 0.95; ROT_VEL_DECAY = 0.90;
     A_kin = np.array([
        [1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt],
        [0,0,0,VEL_DECAY,0,0], [0,0,0,0,VEL_DECAY,0], [0,0,0,0,0,ROT_VEL_DECAY]])
     FN_VX_GAIN = 0.1; FT_VTH_GAIN = 0.05
     B_kin_back_push = np.array([
        [0,0], [0,0], [0,0], [FN_VX_GAIN*dt, 0], [0, 0], [0, FT_VTH_GAIN*dt]])
     FN_VY_GAIN_SIDE = -0.01; FT_VY_GAIN_SIDE = 0.02
     B_kin_side_push = np.array([
        [0,0], [0,0], [0,0], [FN_VX_GAIN*dt, 0],
        [FN_VY_GAIN_SIDE*dt, FT_VY_GAIN_SIDE*dt], [0, FT_VTH_GAIN*dt]])
     return A_kin, B_kin_back_push, B_kin_side_push

class MPController_ForceBased:
    def __init__(self, Q, R, QN, u_min_fn, u_max_fn, mu_p, N):
        self.Q = Q; self.R = R; self.QN = QN; self.u_min_fn = u_min_fn; self.u_max_fn = u_max_fn
        self.mu_p = mu_p; self.N = N; self.n_states = Q.shape[0]; self.n_controls = R.shape[0]
        self.u_k = cp.Variable((self.n_controls, self.N), name="u")
        self.x_k = cp.Variable((self.n_states, self.N + 1), name="x")
        self.x_init = cp.Parameter(self.n_states, name="x_init")
        self.x_ref = cp.Parameter((self.n_states, self.N + 1), name="x_ref")
        self.A_param = cp.Parameter((self.n_states, self.n_states), name="A_d")
        self.B_param = cp.Parameter((self.n_states, self.n_controls), name="B_d")
        objective = 0; constraints = [self.x_k[:, 0] == self.x_init]
        q_diag = np.diag(self.Q); qn_diag = np.diag(self.QN)
        q_theta = q_diag[2]; qn_theta = qn_diag[2]; q_vy = q_diag[4]; qn_vy = qn_diag[4]
        state_indices_flat = [0, 1, 3, 5]; Q_flat = np.diag(q_diag[state_indices_flat]); QN_flat = np.diag(qn_diag[state_indices_flat])
        for k in range(self.N):
            state_diff = self.x_k[:, k] - self.x_ref[:, k]; state_diff_flat = state_diff[state_indices_flat]
            objective += cp.quad_form(state_diff_flat, Q_flat) + q_theta * cp.power(state_diff[2], 2) + q_vy * cp.power(state_diff[4], 2) + cp.quad_form(self.u_k[:, k], self.R)
            constraints += [self.x_k[:, k+1] == self.A_param @ self.x_k[:, k] + self.B_param @ self.u_k[:, k]]
            fn_k = self.u_k[0, k]; ft_k = self.u_k[1, k]
            constraints += [fn_k >= self.u_min_fn, fn_k <= self.u_max_fn, cp.abs(ft_k) <= self.mu_p * fn_k]
        term_state_diff = self.x_k[:, self.N] - self.x_ref[:, self.N]; term_state_diff_flat = term_state_diff[state_indices_flat]
        objective += cp.quad_form(term_state_diff_flat, QN_flat) + qn_theta * cp.power(term_state_diff[2], 2) + qn_vy * cp.power(term_state_diff[4], 2)
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        self.solver_options={'solver':cp.CLARABEL,'verbose':False, 'tol_gap_abs': 1e-4, 'tol_gap_rel': 1e-4}

    def update_dynamics_and_solve(self, current_state, ref_trajectory, A_d, B_d):
        self.A_param.value = A_d; self.B_param.value = B_d
        current_state_norm = current_state.copy(); current_state_norm[2] = normalize_angle(current_state_norm[2])
        ref_trajectory_norm = ref_trajectory.copy()
        angles_to_unwrap = np.concatenate(([current_state_norm[2]], ref_trajectory_norm[2, :]))
        unwrapped_angles = np.unwrap(angles_to_unwrap); ref_trajectory_norm[2, :] = unwrapped_angles[1:]
        self.x_init.value = current_state_norm; self.x_ref.value = ref_trajectory_norm
        try:
            solution_cost = self.problem.solve(**self.solver_options); status = self.problem.status
            if status == cp.OPTIMAL or status == cp.OPTIMAL_INACCURATE:
                if self.u_k.value is not None: return self.u_k[:, 0].value, status
                else: return np.zeros(self.n_controls), "Solver Optimal, Value None"
            else: return np.zeros(self.n_controls), status
        except Exception as e: print(f"MPC Solver Error: {e}"); return np.zeros(self.n_controls), "Solver Exception"

try:
    print(f"Loading model from: {XML_PATH}"); model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model); print("Model loaded successfully.")
except Exception as e: print(f"Error loading MuJoCo model: {e}"); exit()

CUBE_INERTIA_IZZ = 0.001; cube_geom_size = np.array([0.05, 0.05, 0.05])
try:
    cube_body_id = model.body('cube').id
    if np.sum(model.body_inertia[cube_body_id]) > 1e-9:
        CUBE_INERTIA_IZZ = model.body_inertia[cube_body_id][2]; print(f"Using Izz from model: {CUBE_INERTIA_IZZ:.4f}")
        cube_geom_id = model.body_geomadr[cube_body_id]; cube_geom_size = model.geom_size[cube_geom_id].copy()
    else:
        print("Inertia not found/valid in model, calculating from geom size.")
        cube_geom_id = model.body_geomadr[cube_body_id]; cube_geom_type = model.geom_type[cube_geom_id]
        if cube_geom_type == mujoco.mjtGeom.mjGEOM_BOX:
             cube_geom_size = model.geom_size[cube_geom_id].copy(); w = cube_geom_size[0] * 2; h = cube_geom_size[1] * 2
             CUBE_INERTIA_IZZ = CUBE_MASS / 12.0 * (w**2 + h**2); print(f"Calculated Izz (approx): {CUBE_INERTIA_IZZ:.4f}")
        else: print("Warning: Cube geometry not a box, inertia calc inaccurate.")
except Exception as e: print(f"Warning: Could not get cube geom/inertia: {e}")
L_matrix = calculate_limit_surface_matrix(MU_G, CUBE_MASS, cube_geom_size); print(f"L Matrix Diag: {np.diag(L_matrix)}")
try:
    cube_jnt_id = model.joint('cube_joint').id; cube_jnt_adr = model.jnt_qposadr[cube_jnt_id]; cube_jnt_veladr = model.jnt_dofadr[cube_jnt_id]
    print(f"Found joint 'cube_joint': qpos_adr={cube_jnt_adr}, qvel_adr={cube_jnt_veladr}")
except Exception as e: print("Error: Joint 'cube_joint' not found."); cube_jnt_adr = 0; cube_jnt_veladr = 0
try:
    pusher_body_id = model.body('pusher_body').id
    if model.body_mocapid[pusher_body_id] >= 0: pusher_mocap_id = model.body_mocapid[pusher_body_id]
    else: raise ValueError("No mocap ID found for pusher_body")
    print(f"Found pusher_mocap_id: {pusher_mocap_id}")
except Exception as e: print(f"Error finding pusher mocap ID: {e}. Using 0."); pusher_mocap_id = 0
site_data = {}
try:
    print("Loading site data:")
    for site_name in ALL_SITE_NAMES:
         site_id = model.site(site_name).id; site_pos_body = model.site_pos[site_id].copy()
         site_data[site_name] = {'id': site_id, 'pos': site_pos_body}; print(f" - Site '{site_name}': ID={site_id}, Pos={site_pos_body}")
except Exception as e: print(f"Error getting site data: {e}.")
try: desired_marker_id = model.site(DESIRED_MARKER_NAME).id; print(f"Found desired marker site '{DESIRED_MARKER_NAME}': ID={desired_marker_id}")
except Exception as e: print(f"Warning: Marker site '{DESIRED_MARKER_NAME}' not found."); desired_marker_id = -1

controller = MPController_ForceBased(Q, R_cost, Q_N, 0.0, max_fn, MU_P, MPC_HORIZON)
sim_time = 0.0; last_control_time = -CONTROL_TIMESTEP
optimal_force_cmd = np.zeros(n_controls); pusher_velocity_cmd = np.zeros(3)
current_segment_index = 0; selected_site_name = SITES_Y_PUSH["center"]

mujoco.mj_forward(model, data)
initial_cube_pos_3d = data.qpos[cube_jnt_adr:cube_jnt_adr+3].copy(); initial_cube_quat = data.qpos[cube_jnt_adr+3:cube_jnt_adr+7].copy()
initial_cube_pos_xy = initial_cube_pos_3d[:2].copy(); initial_cube_rot_mat_flat = np.zeros(9)
mujoco.mju_quat2Mat(initial_cube_rot_mat_flat, initial_cube_quat); initial_cube_rot_mat = initial_cube_rot_mat_flat.reshape((3, 3))
r_init = R.from_matrix(initial_cube_rot_mat); initial_cube_theta = r_init.as_euler('xyz', degrees=False)[2]; initial_cube_theta = normalize_angle(initial_cube_theta)
print(f"Stored Initial State: Pos={initial_cube_pos_xy}, Theta={np.degrees(initial_cube_theta):.1f} deg")
if selected_site_name in site_data:
    initial_target_site_pos_body = site_data[selected_site_name]['pos']; initial_target_site_pos_world = initial_cube_pos_3d + initial_cube_rot_mat @ initial_target_site_pos_body
    start_offset = np.array([0, -0.05, 0]); initial_pusher_pos = initial_target_site_pos_world + start_offset; initial_pusher_pos[2] = pusher_z_height
    data.mocap_pos[pusher_mocap_id] = initial_pusher_pos; data.mocap_quat[pusher_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
    print(f"Initial pusher pos set to: {initial_pusher_pos}")
else: print("Warning: Initial site name not found. Pusher positioning skipped.")

times = []; actual_xs = []; actual_ys = []; desired_xs = []; desired_ys = []; actual_thetas = []; desired_thetas = []

A_kin, B_kin_back, B_kin_side = get_kinematic_model_matrices(CONTROL_TIMESTEP)

print("\nStarting Simulation...")
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_sim_loop = time.time(); step_count = 0
        forward_dir_body = np.array([0., 1., 0.]); forward_dir = initial_cube_rot_mat @ forward_dir_body; forward_dir = forward_dir[:2]
        norm_fwd = np.linalg.norm(forward_dir);
        if norm_fwd > 1e-6: forward_dir /= norm_fwd
        else: forward_dir = np.array([0., 1.])
        path_corner_pos = initial_cube_pos_xy + forward_dir * FORWARD_DISTANCE
        if TURN_DIRECTION.lower() == 'right': turn_dir = np.array([forward_dir[1], -forward_dir[0]])
        else: turn_dir = np.array([-forward_dir[1], forward_dir[0]])
        path_end_point = path_corner_pos + turn_dir * TURN_DISTANCE
        print(f"Path Info: Corner@{path_corner_pos}, End@{path_end_point}")

        while viewer.is_running() and sim_time < SIM_DURATION:
            step_start = time.time(); first_print_this_step = False

            if sim_time >= last_control_time + CONTROL_TIMESTEP:
                last_control_time += CONTROL_TIMESTEP; first_print_this_step = True

                actual_cube_pos_3d = data.qpos[cube_jnt_adr:cube_jnt_adr+3].copy(); actual_pos = actual_cube_pos_3d[:2]
                actual_quat = data.qpos[cube_jnt_adr+3:cube_jnt_adr+7].copy(); actual_cube_rot_mat_flat = np.zeros(9)
                mujoco.mju_quat2Mat(actual_cube_rot_mat_flat, actual_quat); actual_cube_rot_mat = actual_cube_rot_mat_flat.reshape((3,3))
                r = R.from_matrix(actual_cube_rot_mat); actual_theta = r.as_euler('xyz', degrees=False)[2]; actual_theta = normalize_angle(actual_theta)
                actual_cube_vel = data.qvel[cube_jnt_veladr:cube_jnt_veladr+6].copy(); actual_vel_xy = actual_cube_vel[:2]; actual_vel_theta = actual_cube_vel[5]
                current_cube_state = np.array([actual_pos[0], actual_pos[1], actual_theta, actual_vel_xy[0], actual_vel_xy[1], actual_vel_theta])

                des_pos, des_vel, des_theta, des_vtheta = get_relative_path_target(sim_time, initial_cube_pos_xy, initial_cube_theta); des_theta = normalize_angle(des_theta)
                if desired_marker_id != -1: data.site_xpos[desired_marker_id] = [des_pos[0], des_pos[1], 0.01]

                times.append(sim_time); actual_xs.append(actual_pos[0]); actual_ys.append(actual_pos[1])
                desired_xs.append(des_pos[0]); desired_ys.append(des_pos[1])
                actual_thetas.append(actual_theta); desired_thetas.append(des_theta)

                segment_switched = False
                if current_segment_index == 0:
                    disp_vec = actual_pos - initial_cube_pos_xy
                    dist_along_fwd = np.dot(disp_vec, forward_dir)
                    if dist_along_fwd >= FORWARD_DISTANCE - WAYPOINT_TOLERANCE:
                        current_segment_index = 1; segment_switched = True
                        if first_print_this_step: print(f"\n>>> Switched to Turn Segment (Dist Trigger) at {sim_time:.2f}s <<<\n")
                elif current_segment_index == 1:
                    disp_vec = actual_pos - path_corner_pos
                    dist_along_turn = np.dot(disp_vec, turn_dir)
                    if dist_along_turn >= TURN_DISTANCE - WAYPOINT_TOLERANCE:
                        current_segment_index = 2; segment_switched = True
                        if first_print_this_step: print(f"\n>>> Switched to Finished Segment (Dist Trigger) at {sim_time:.2f}s <<<\n")

                orientation_error = normalize_angle(des_theta - actual_theta); site_selection_reason = "N/A"
                if current_segment_index < 2:
                    if abs(orientation_error) < ORIENTATION_ERROR_THRESHOLD:
                        selected_site_name = SITES_Y_PUSH["center"] if current_segment_index == 0 else SITES_X_PUSH["center"]
                        site_selection_reason = f"Seg {current_segment_index}, Small Err -> Center Push"
                    else:
                        if current_segment_index == 0:
                            selected_site_name = SITES_Y_PUSH["left_turn"] if orientation_error > 0 else SITES_Y_PUSH["right_turn"]
                            site_selection_reason = f"Seg 0, Large Err {np.degrees(orientation_error):.1f} -> Correction Push"
                        else:
                            selected_site_name = SITES_X_PUSH["center"]
                            site_selection_reason = f"Seg 1, Large Err {np.degrees(orientation_error):.1f} -> Center Push (No Correction)"
                else:
                    selected_site_name = SITES_Y_PUSH["center"]; site_selection_reason = "Path Finished"

                ref_trajectory = calculate_reference_trajectory(sim_time, CONTROL_TIMESTEP, MPC_HORIZON, initial_cube_pos_xy, initial_cube_theta)

                optimal_force_cmd = np.zeros(n_controls); pusher_velocity_cmd = np.zeros(3); solver_status = "Not Run"; mpc_success = False
                if current_segment_index < 2:
                    if selected_site_name in site_data:
                         site_pos_body = site_data[selected_site_name]['pos']
                         current_B_kin = B_kin_side if current_segment_index == 1 else B_kin_back
                         force_cmd_temp, solver_status = controller.update_dynamics_and_solve(
                              current_cube_state.copy(), ref_trajectory.copy(), A_kin, current_B_kin)
                         optimal_force_cmd = force_cmd_temp if force_cmd_temp is not None else np.zeros(n_controls)
                         mpc_success = (solver_status is not None) and \
                                       ('optimal' in solver_status.lower() or 'inaccurate' in solver_status.lower()) and \
                                       (force_cmd_temp is not None)
                    else:
                         if first_print_this_step: print(f"Error: Site '{selected_site_name}' not found. Skipping MPC.")

                if mpc_success and selected_site_name in site_data:
                    current_target_site_pos_body = site_data[selected_site_name]['pos']
                    target_contact_pos_world = actual_cube_pos_3d + actual_cube_rot_mat @ current_target_site_pos_body
                    current_pusher_pos = data.mocap_pos[pusher_mocap_id].copy()
                    vector_to_target_site = target_contact_pos_world - current_pusher_pos
                    site_vel_body = get_contact_jacobian(current_target_site_pos_body) @ actual_cube_vel[[0,1,5]]
                    site_vel_world = actual_cube_rot_mat[:2,:2] @ site_vel_body
                    vector_to_target_site_xy = vector_to_target_site[:2]
                    correction_vel = vector_to_target_site_xy / CONTROL_TIMESTEP
                    fn_cmd = optimal_force_cmd[0]
                    cn_body, _ = get_contact_normal_tangent(selected_site_name)
                    contact_normal_world = actual_cube_rot_mat[:2,:2] @ cn_body
                    inward_vel = -contact_normal_world * FN_TO_INWARD_VEL_GAIN * fn_cmd
                    target_vel_xy = site_vel_world + correction_vel + inward_vel
                    speed = np.linalg.norm(target_vel_xy)
                    max_push_speed = RELATIVE_PATH_VELOCITY * 2.0
                    if speed > max_push_speed: target_vel_xy = (target_vel_xy / speed) * max_push_speed
                    pusher_velocity_cmd[:2] = target_vel_xy
                else:
                    pusher_velocity_cmd = np.zeros(3)

                pusher_velocity_cmd[2] = 0.0

            current_mocap_pos = data.mocap_pos[pusher_mocap_id].copy()
            target_mocap_pos = current_mocap_pos + pusher_velocity_cmd * CONTROL_TIMESTEP
            target_mocap_pos[2] = pusher_z_height
            data.mocap_pos[pusher_mocap_id, :] = target_mocap_pos[:]
            data.mocap_quat[pusher_mocap_id, :] = np.array([1., 0., 0., 0.])

            try:
                 mujoco.mj_step(model, data); sim_time += model.opt.timestep; step_count += 1
            except Exception as e: print(f"MuJoCo step error: {e}"); break

            viewer.sync()
            time_elapsed_this_step = time.time() - step_start
            time_to_sleep = model.opt.timestep - time_elapsed_this_step + SLEEP_PER_STEP
            if time_to_sleep > 0: time.sleep(time_to_sleep)

    end_sim_loop = time.time()
    print(f"\nSimulation Finished."); print(f"Total Sim Time: {sim_time:.3f} s / {SIM_DURATION}s"); print(f"Total Steps: {step_count}")
    print(f"Real Time Elapsed: {end_sim_loop - start_sim_loop:.3f} s")
    avg_step_time = (end_sim_loop - start_sim_loop) / step_count if step_count > 0 else 0
    print(f"Average Real Time Per Step: {avg_step_time*1000:.3f} ms")

except Exception as e: print(f"\nAn error occurred: {e}"); import traceback; traceback.print_exc()
finally:
    if 'viewer' in locals() and viewer.is_running(): viewer.close()
    print("Viewer closed.")

# --- Plotting Results ---
if times:
    times = np.array(times); actual_xs = np.array(actual_xs); actual_ys = np.array(actual_ys)
    desired_xs = np.array(desired_xs); desired_ys = np.array(desired_ys)
    actual_thetas = np.array(actual_thetas); desired_thetas = np.array(desired_thetas)

    pos_error = np.sqrt((actual_xs - desired_xs)**2 + (actual_ys - desired_ys)**2)
    theta_error_raw = actual_thetas - desired_thetas
    theta_error = np.array([normalize_angle(e) for e in theta_error_raw])

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Trajectory
    axs[0].plot(desired_xs, desired_ys, 'r--', label='Desired Trajectory')
    axs[0].plot(actual_xs, actual_ys, 'b-', label='Actual Trajectory')
    axs[0].set_xlabel("X Position (m)"); axs[0].set_ylabel("Y Position (m)")
    axs[0].set_title("Cube Trajectory"); axs[0].legend(); axs[0].grid(True); axs[0].axis('equal')

    # Plot 2: Position Error 
    axs[1].plot(times, pos_error, 'g-', label='Position Error (m)')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Position Error (m)") 
    axs[1].set_title("Position Error over Time") 
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(); plt.show()
else:
    print("No data recorded for plotting.")

