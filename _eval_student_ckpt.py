import sys, numpy as np, torch, torch.nn.functional as F
from student_vision import DinoCPGStudent, make_windows, DATA
from distill_student import split_logits

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/student_vision/dino_cpg_ep5.pt"
d = np.load(DATA)
frames, acts, ep_start = d["frames"], d["acts"], d["ep_start"]
x_all = torch.from_numpy(frames); y_all = torch.from_numpy(acts)
n = len(frames); ep_bounds = np.flatnonzero(ep_start)
n_val = max(1, n // 10)
perm = np.random.RandomState(0).permutation(n)
val_idx = perm[:n_val]
net = DinoCPGStudent()
net.load_state_dict(torch.load(ckpt, map_location="cpu"))
net.eval().cuda()
correct = np.zeros(4)
with torch.no_grad():
    for b in range(0, len(val_idx), 16):
        idx = val_idx[b:b+16]
        widx = make_windows(idx, ep_bounds, 16, 2)
        x = x_all[torch.from_numpy(widx)].permute(0,1,4,2,3).float().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x)
        for k, lg in enumerate(split_logits(pred)):
            correct[k] += (lg.argmax(1).cpu() == y_all[idx, k]).sum()
acc = correct / len(val_idx)
print(f"{ckpt}: move {acc[0]:.3f} aim {acc[1]:.3f} shoot {acc[2]:.3f} super {acc[3]:.3f}")
