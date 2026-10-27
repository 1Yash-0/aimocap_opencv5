import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists(): print('ABORT exists', dst); sys.exit(2)
z = np.load(src)
P = np.asarray(z['skeleton3d'], float) * 100.0  # m -> cm, Y-up
fps = float(z['fps']); T = len(P)
for j in range(P.shape[1]):
    for c in range(3):
        v = P[:, j, c]; ok = np.isfinite(v)
        if ok.any() and not ok.all(): P[:, j, c] = np.interp(np.arange(T), np.flatnonzero(ok), v[ok])
        elif not ok.any(): P[:, j, c] = 0.0
I = dict(nose=0, lsho=5, rsho=6, lelb=7, relb=8, lwri=9, rwri=10, lhip=11, rhip=12, lknee=13, rknee=14, lank=15, rank=16, lbtoe=17, rbtoe=20)
X = {k: P[:, i] for k, i in I.items()}
X['root'] = 0.5 * (X['lhip'] + X['rhip']); X['neck'] = 0.5 * (X['lsho'] + X['rsho'])
def nrm(v): return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
def frame(xax, up):
    x = nrm(xax); zz = nrm(np.cross(x, up)); y = np.cross(zz, x)
    return R.from_matrix(np.stack([x, y, zz], axis=-1))
# skeleton: name -> (parent, aim child or None, kind)
parent = {'neck': 'root', 'nose': 'neck', 'lhip': 'root', 'rhip': 'root', 'lsho': 'neck', 'rsho': 'neck',
          'lelb': 'lsho', 'lwri': 'lelb', 'relb': 'rsho', 'rwri': 'relb', 'lknee': 'lhip', 'lank': 'lknee', 'rknee': 'rhip', 'rank': 'rknee', 'lbtoe': 'lank', 'rbtoe': 'rank'}
aim = {'lsho': 'lelb', 'lelb': 'lwri', 'rsho': 'relb', 'relb': 'rwri', 'lhip': 'lknee', 'lknee': 'lank', 'rhip': 'rknee', 'rknee': 'rank', 'lank': 'lbtoe', 'rank': 'rbtoe', 'nose': None, 'lwri': None, 'rwri': None, 'lbtoe': None, 'rbtoe': None}
children = {}
for c, p in parent.items(): children.setdefault(p, []).append(c)
G = {'root': frame(X['rhip'] - X['lhip'], X['neck'] - X['root']), 'neck': frame(X['rsho'] - X['lsho'], X['nose'] - X['neck'])}
# rest offsets in parent frame (median)
off = {c: np.median(G_inv_apply, axis=0) for c, G_inv_apply in []}
rest_dir = {}
ORDER = ['root', 'neck', 'nose', 'lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip', 'lknee', 'lank', 'lbtoe', 'rhip', 'rknee', 'rank', 'rbtoe']
OFF = {}
def swing(u, v):
    ax = np.cross(u, v); s = np.linalg.norm(ax, axis=-1); c = np.sum(u * v, -1)
    ang = np.arctan2(s, c); ax = ax / np.maximum(s, 1e-9)[:, None]
    return R.from_rotvec(ax * ang[:, None])
for n in ORDER[1:]:
    p = parent[n]
    if p not in G: raise SystemExit(f'parent {p} not ready for {n}')
    OFF[n] = np.median(G[p].inv().apply(X[n] - X[p]), axis=0)
    if n in G: continue
    a = aim.get(n)
    if a is None:
        G[n] = G[p]; continue
    cur = nrm(X[a] - X[n])
    rd = nrm(np.median(G[p].inv().apply(cur), axis=0))[None].repeat(T, 0)
    world_rest = G[p].apply(rd)
    G[n] = swing(world_rest, cur) * G[p]
END = {n: np.median(G[n].inv().apply(X[a2] - X[n]), axis=0) if False else None for n in []}
lines = ['HIERARCHY']; order = []
def emit(n, d, lines, order):
    ind = '  ' * d
    o = np.zeros(3) if n == 'root' else OFF[n]
    lines.append(f'{ind}{"ROOT" if n == "root" else "JOINT"} {n}'); lines.append(ind + '{')
    lines.append(f'{ind}  OFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}')
    lines.append(f'{ind}  CHANNELS ' + ('6 Xposition Yposition Zposition ' if n == 'root' else '3 ') + 'Zrotation Xrotation Yrotation')
    order.append(n)
    kids = children.get(n, [])
    for k in kids: emit(k, d + 1, lines, order)
    if not kids:
        e = np.array([0.0, 5.0, 0.0]) if n == 'nose' else np.array([0.0, -3.0, 0.0])
        lines += [f'{ind}  End Site', ind + '  {', f'{ind}    OFFSET {e[0]:.1f} {e[1]:.1f} {e[2]:.1f}', ind + '  }']
    lines.append(ind + '}')
emit('root', 0, lines, order)
lines += ['MOTION', f'Frames: {T}', f'Frame Time: {1.0 / fps:.6f}']
E = {}
for n in order:
    loc = G[n] if n == 'root' else G[parent[n]].inv() * G[n]
    E[n] = loc.as_euler('ZXY', degrees=True)
for t in range(T):
    vals = [f'{v:.4f}' for v in X['root'][t]]
    for n in order: vals += [f'{v:.4f}' for v in E[n][t]]
    lines.append(' '.join(vals))
dst.write_text('\n'.join(lines) + '\n', encoding='ascii')
# FK check: reconstruct wrist/ankle positions and compare with points
def fk_pos(n):
    chain = []; m = n
    while m != 'root': chain.append(m); m = parent[m]
    pos = X['root'].copy(); g = G['root']
    for m in reversed(chain):
        pos = pos + G[parent[m]].apply(np.tile(OFF[m], (T, 1)))
    return pos
for n in ['lwri', 'rwri', 'lank', 'rank', 'nose']:
    e = np.linalg.norm(fk_pos(n) - X[n], axis=1)
    print(f'FK vs points {n}: p50 {np.percentile(e, 50):.2f} p95 {np.percentile(e, 95):.2f} cm (bone lengths fixed at medians)')
print('wrote', dst, 'joints', len(order), 'frames', T, 'fps', round(fps, 3))