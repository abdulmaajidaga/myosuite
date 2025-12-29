import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.colors as mcolors

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

def resample_segment(segment, target_length):
    if len(segment) < 2: return np.zeros((target_length, 3))
    orig_idx = np.linspace(0, 1, len(segment))
    target_idx = np.linspace(0, 1, target_length)
    res = np.zeros((target_length, 3))
    for i in range(3): res[:, i] = np.interp(target_idx, orig_idx, segment[:, i])
    return res

def animate_filtered_phased():
    data_dir = 'data/kinematic/Stroke/'
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files: return
    marker, N_phase = 'WRA', 250
    data_map = {}
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
                xyz_filt = butter_lowpass_filter(xyz_raw)
                s1, s2, s3 = get_phases(xyz_filt)
                phs = np.vstack([resample_segment(xyz_filt[:s1], N_phase), resample_segment(xyz_filt[s1:s2], N_phase), resample_segment(xyz_filt[s2:s3], N_phase), resample_segment(xyz_filt[s3:], N_phase)])
                data_map[fname] = phs - phs[0]
        except Exception: pass
    target, outlier = 'S1_12_1.csv', 'S1_12_2.csv'
    if target in data_map and outlier in data_map:
        data_map[outlier] += data_map[target][-1] - data_map[outlier][-1]
    trajectories, filenames = [], []
    for fname in sorted(data_map.keys()):
        trajectories.append(data_map[fname])
        filenames.append(fname)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    all_p = np.vstack(trajectories)
    mx, My, Mz = all_p.max(axis=0); mi = all_p.min(axis=0)
    for ax, t, x, y in [(ax1, "Top", 0, 1), (ax2, "Side", 0, 2)]:
        ax.set_title(t); ax.set_xlim(mi[0]-50, mx+50); ax.set_ylim(mi[y]-50, all_p[:,y].max()+50); ax.grid(True, alpha=0.3)
    color_p = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    sub_col = {}
    dots1, trails1, dots2, trails2 = [], [], [], []
    for i, fname in enumerate(filenames):
        sid = fname.split('_')[0]
        if sid not in sub_col: sub_col[sid] = color_p[len(sub_col)%len(color_p)]
        col = sub_col[sid]
        lbl = sid if sid not in [h.get_label() for h in trails1] else ""
        d1, = ax1.plot([], [], 'o', ms=3, color=col, zorder=5)
        t1, = ax1.plot([], [], '-', lw=1.5, alpha=0.5, color=col, zorder=5, label=lbl)
        d2, = ax2.plot([], [], 'o', ms=3, color=col, zorder=5)
        t2, = ax2.plot([], [], '-', lw=1.5, alpha=0.5, color=col, zorder=5)
        dots1.append(d1); trails1.append(t1); dots2.append(d2); trails2.append(t2)
    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    txt = fig.text(0.5, 0.02, '', ha='center')
    def update(f):
        txt.set_text(f'Progress: {f/10:.1f}%')
        for i, traj in enumerate(trajectories):
            idx = min(f, len(traj)-1); pos = traj[idx]; hist = traj[:idx+1]
            dots1[i].set_data([pos[0]], [pos[1]]); trails1[i].set_data(hist[:,0], hist[:,1])
            dots2[i].set_data([pos[0]], [pos[2]]); trails2[i].set_data(hist[:,0], hist[:,2])
        return dots1+trails1+dots2+trails2+[txt]
    ani = animation.FuncAnimation(fig, update, frames=1000, blit=True, interval=20)
    ani.save('animated_stroke.mp4', writer=animation.FFMpegWriter(fps=60))

if __name__ == "__main__":
    animate_filtered_phased()
