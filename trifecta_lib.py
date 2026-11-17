"""3連単の確率計算(trifecta_v4.py と同じ式)。当日予想用に切り出したもの。"""
import itertools

import numpy as np

PERMS = np.array(list(itertools.permutations(range(6), 3)))  # 120通り
PA, PB, PC = PERMS.T
COMBO = [f"{a+1}-{b+1}-{c+1}" for a, b, c in PERMS]


def tri_probs(pk, w, t3, tab, bw, bt, g1, g2):
    """1レースの120通りの3連単確率。pk=決まり手確率(5), w=1着率(6), t3=3連対率(6)"""
    PAt, PBt, PCt = tab
    uA = (w / bw) ** g1
    uT = (t3 / bt) ** g2
    pa = PAt * uA
    pa = pa / pa.sum(1, keepdims=True)
    pb = PBt * uT
    pb = pb / pb.sum(2, keepdims=True)
    pc = PCt * uT
    pc = pc / pc.sum(3, keepdims=True)
    return (pk[:, None] * pa[:, PA] * pb[:, PA, PB] * pc[:, PA, PB, PC]).sum(0)
