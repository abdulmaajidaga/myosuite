"""
IK solver and mocap alignment functions
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from typing import Dict, NamedTuple, Optional, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class IKResult(NamedTuple):
    qpos: np.ndarray
    err_norm: float
    steps: int
    success: bool

def solve_ik_multi_site(sim: Any,  # MyoSuite sim object
                       site_targets: Dict[str, np.ndarray],
                       tol: float = 1e-5,
                       max_steps: int = 500,
                       regularization_strength: float = 0.01,
                       rcond: float = 1e-8) -> Optional[IKResult]:
    """
    Solve IK for multiple sites using damped least squares with adaptive regularization.
    """
    model = sim.model.ptr
    data = sim.data.ptr
    
    # Get site IDs and stack target positions
    site_ids, target_pos_vec = [], []
    for name, pos in site_targets.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid == -1:
            logger.error(f"Site {name} not found in model")
            return None
        site_ids.append(sid)
        target_pos_vec.append(pos)
    target_pos_vec = np.concatenate(target_pos_vec)
    
    # Main IK loop
    reg = regularization_strength
    for i in range(max_steps):
        current_pos = np.concatenate([data.site_xpos[sid] for sid in site_ids])
        
        # Stack Jacobians
        jac_stack = np.zeros((3 * len(site_ids), model.nv))
        for idx, sid in enumerate(site_ids):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, sid)
            jac_stack[3*idx:3*idx+3, :] = jacp
            
        pos_error = target_pos_vec - current_pos
        err_norm = np.linalg.norm(pos_error)
        
        if err_norm < tol:
            return IKResult(qpos=data.qpos.copy(), 
                          err_norm=err_norm,
                          steps=i,
                          success=True)
        
        # Damped least squares with adaptive regularization
        try:
            JTJ = jac_stack.T @ jac_stack
            rhs = jac_stack.T @ pos_error
            dq = np.linalg.solve(JTJ + reg * np.eye(model.nv), rhs)
        except np.linalg.LinAlgError:
            # If solve fails, try least squares with increased regularization
            logger.warning(f"Matrix solve failed, using lstsq with reg={reg}")
            dq = np.linalg.lstsq(jac_stack.T @ jac_stack + reg * np.eye(model.nv),
                                jac_stack.T @ pos_error,
                                rcond=rcond)[0]
            reg *= 2  # Increase regularization for next iteration
            
        # Update position with step limits
        step_size = min(1.0, 0.5 / (np.linalg.norm(dq) + 1e-6))
        mujoco.mj_integratePos(model, data.qpos, dq * step_size, 1.0)
        mujoco.mj_forward(model, data)
    
    return IKResult(qpos=data.qpos.copy(),
                   err_norm=err_norm,
                   steps=max_steps,
                   success=False)

def align_mocap_to_model(sim: Any,  # MyoSuite sim object
                        mocap_markers: Dict[str, np.ndarray],
                        alignment_frames: int = 1) -> Tuple[Rotation, np.ndarray]:
    """
    Compute optimal alignment between mocap and model coordinates.
    Uses Procrustes analysis on the first few frames of data.
    """
    # Get model reference positions
    model_points = {}
    for name in ['V_Shoulder', 'V_Elbow', 'V_Wrist']:
        site_id = mujoco.mj_name2id(sim.model.ptr, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id == -1:
            raise ValueError(f"Required site {name} not found in model")
        model_points[name] = sim.data.ptr.site_xpos[site_id].copy()
    
    # Stack model points
    model_pts = np.vstack([model_points[k] for k in ['V_Shoulder', 'V_Elbow', 'V_Wrist']])
    
    # Average first few mocap frames and reshape coordinates
    mocap_pts = np.zeros((3, 3))
    for i, name in enumerate(['V_Shoulder', 'V_Elbow', 'V_Wrist']):
        pts = mocap_markers[name][:alignment_frames]
        avg_pt = np.mean(pts, axis=0)
        # Convert mocap convention [x, y, z] to model [x, -z, y]
        mocap_pts[i] = [avg_pt[0], -avg_pt[2], avg_pt[1]]
    
    # Center both point sets
    model_centroid = np.mean(model_pts, axis=0)
    mocap_centroid = np.mean(mocap_pts, axis=0)
    
    model_centered = model_pts - model_centroid
    mocap_centered = mocap_pts - mocap_centroid
    
    # Find optimal rotation
    rotation, _ = Rotation.align_vectors(model_centered, mocap_centered)
    
    return rotation, model_centroid - rotation.apply(mocap_centroid)