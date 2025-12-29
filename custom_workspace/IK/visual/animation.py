import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
import numpy as np
import plotly.colors as pc
from scipy.signal import butter, filtfilt, find_peaks

def butter_lowpass_filter(data, cutoff=6, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def get_phases(xyz, fs=100):
    vel = np.diff(xyz, axis=0) * fs
    speed = np.linalg.norm(vel, axis=1)
    cutoff = int(len(speed) * 0.05) or 1
    roi = speed[cutoff:-cutoff]
    denom = roi.max() - roi.min()
    norm = (roi - roi.min()) / denom if denom > 0 else np.zeros_like(roi)
    inv = 1 - norm
    peaks, props = find_peaks(inv, distance=len(speed)*0.1, prominence=0.05)
    idxs = peaks + cutoff
    if len(idxs) >= 3:
        top = np.argsort(props['prominences'])[::-1][:3]
        stops = sorted(idxs[top])
    elif len(idxs) == 2:
        s = sorted(idxs)
        stops = sorted(s + [s[1] + (len(speed)-s[1])//2])
    else:
        L = len(speed)
        stops = [int(L*0.25), int(L*0.50), int(L*0.75)]
    return stops

def resample_simple(traj, target_length):
    orig_idx = np.linspace(0, 1, len(traj))
    targ_idx = np.linspace(0, 1, target_length)
    res = np.zeros((target_length, 3))
    for i in range(3): res[:,i] = np.interp(targ_idx, orig_idx, traj[:,i])
    return res

def resample_phased(traj, stops, phase_len=250):
    s1, s2, s3 = stops
    segs = [traj[:s1], traj[s1:s2], traj[s2:s3], traj[s3:]]
    return np.vstack([resample_simple(s, phase_len) for s in segs])

def interactive_animated_phased():
    data_dir = 'data/kinematic/Stroke/'
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files: return
    marker, N_total, anim_step = 'WRA', 1000, 10
    data_std, data_phs = {}, {}
    for file_path in files:
        fname = os.path.basename(file_path)
        try:
            df_raw = pd.read_csv(file_path, header=None, nrows=2)
            header_l0 = df_raw.iloc[0].ffill().astype(str).str.strip().values
            header_l1 = df_raw.iloc[1].values
            cols = pd.MultiIndex.from_arrays([header_l0, header_l1])
            df = pd.read_csv(file_path, header=None, skiprows=2, names=cols)
            if marker in df.columns.get_level_values(0):
                xyz_raw = df[marker][['X', 'Y', 'Z']].values
                std = resample_simple(xyz_raw, N_total)
                data_std[fname] = std - std[0]
                xyz_filt = butter_lowpass_filter(xyz_raw)
                stops = get_phases(xyz_filt)
                phs = resample_phased(xyz_filt, stops, N_total//4)
                data_phs[fname] = phs - phs[0]
        except Exception: pass
    target, outlier = 'S1_12_1.csv', 'S1_12_2.csv'
    if target in data_std and outlier in data_std:
        data_std[outlier] += data_std[target][-1] - data_std[outlier][-1]
    if target in data_phs and outlier in data_phs:
        data_phs[outlier] += data_phs[target][-1] - data_phs[outlier][-1]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top View (X vs Y)", "Side View (X vs Z)"), horizontal_spacing=0.1)
    filenames = sorted(data_std.keys())
    subject_colors = {}
    palette = pc.qualitative.Dark24 + pc.qualitative.Alphabet
    traces_std_indices, traces_phs_indices = [], []
    current_idx = 0
    def add_group(data_map, visible=True):
        nonlocal current_idx
        indices = []
        for fname in filenames:
            xyz, sub = data_map[fname], fname.split('_')[0]
            if sub not in subject_colors: subject_colors[sub] = palette[len(subject_colors)%len(palette)]
            col = subject_colors[sub]
            fig.add_trace(go.Scatter(x=xyz[:,0], y=xyz[:,1], mode='lines', line=dict(color=col, width=1), opacity=0.3, name=sub, legendgroup=sub, showlegend=(current_idx==0), visible=visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=xyz[:,0], y=xyz[:,2], mode='lines', line=dict(color=col, width=1), opacity=0.3, name=sub, legendgroup=sub, showlegend=False, visible=visible), row=1, col=2)
            fig.add_trace(go.Scatter(x=[xyz[0,0]], y=[xyz[0,1]], mode='markers', marker=dict(color=col, size=6), name=sub, legendgroup=sub, showlegend=False, text=f"{fname}", hoverinfo='text', visible=visible), row=1, col=1)
            fig.add_trace(go.Scatter(x=[xyz[0,0]], y=[xyz[0,2]], mode='markers', marker=dict(color=col, size=6), name=sub, legendgroup=sub, showlegend=False, text=f"{fname}", hoverinfo='text', visible=visible), row=1, col=2)
            indices.extend(range(current_idx, current_idx+4))
            current_idx += 4
        return indices
    traces_std_indices = add_group(data_std, True)
    traces_phs_indices = add_group(data_phs, False)
    frames = []
    dot_indices = [i for i in range(len(traces_std_indices) + len(traces_phs_indices)) if i % 4 >= 2]
    for k in range(0, N_total, anim_step):
        frame_traces = []
        for fname in filenames:
            pos = data_std[fname][k]
            frame_traces.append(go.Scatter(x=[pos[0]], y=[pos[1]]))
            frame_traces.append(go.Scatter(x=[pos[0]], y=[pos[2]]))
        for fname in filenames:
            pos = data_phs[fname][k]
            frame_traces.append(go.Scatter(x=[pos[0]], y=[pos[1]]))
            frame_traces.append(go.Scatter(x=[pos[0]], y=[pos[2]]))
        frames.append(go.Frame(data=frame_traces, traces=dot_indices, name=str(k)))
    fig.frames = frames
    fig.update_layout(title="Interactive Stroke Analysis", height=600, updatemenus=[
        dict(type="buttons", direction="left", x=0.1, y=0, buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])]),
        dict(type="dropdown", direction="down", x=0.3, y=1.15, showactive=True, buttons=[
            dict(label="Mode: Standard Time", method="update", args=[{"visible": [i < len(traces_std_indices) for i in range(current_idx)]}]),
            dict(label="Mode: Phase Normalized", method="update", args=[{"visible": [i >= len(traces_std_indices) for i in range(current_idx)]}])])],
        sliders=[{"steps": [{"args": [[str(k)], {"frame": {"duration":0,"redraw":True},"mode":"immediate"}], "label": str(k), "method": "animate"} for k in range(0, N_total, anim_step)], "x": 0.1, "len": 0.9}])
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.write_html('animation.html', auto_play=False)

if __name__ == "__main__":
    interactive_animated_phased()
