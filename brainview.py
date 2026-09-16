"""Live 3D view of the connectome while it plays.

Every dot is one neuron at its real measured position. A dot lights up on the
frame its neuron spikes and fades over ~200 ms, so what you see is the actual
traffic driving the game, not an animation of it.

Cost control: the dim background is re-projected only every REPROJ frames,
while the spiking subset — a few thousand neurons — is re-projected every
frame. Pixels are written with bincount into a numpy buffer, never with
per-point draw calls.
"""
from __future__ import annotations

import os
import numpy as np
import pygame

DATA = os.environ.get("FLY_DATA", os.path.expanduser("~/fly-data"))
REPROJ = 6                    # frames between background rebuilds

REGIONS = [
    ("optic L", "#38bdf8"),
    ("optic R", "#4f46e5"),
    ("central brain", "#fb923c"),
    ("VNC", "#14b8a6"),
    ("visual proj.", "#ef4444"),
    ("descending", "#fde047"),
    ("ascending", "#c084fc"),
]


def _hex(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


class BrainView:
    def __init__(self, w, h, data=DATA):
        z = np.load(os.path.join(data, "brain.npz"), allow_pickle=True)
        pos = z["positions"]
        sc = z["superclass"].astype(str)
        side = z["side"].astype(str)
        ct = z["cell_type"].astype(str)

        ok = ~np.isnan(pos).any(1)
        self.idx = np.flatnonzero(ok)           # neuron ids that can be drawn
        self.n_drawn = len(self.idx)
        # display axes: x = left-right, y = anterior-posterior, z = up
        P = np.column_stack([pos[ok, 0], pos[ok, 2], -pos[ok, 1]]).astype(np.float32)
        P -= (P.max(0) + P.min(0)) / 2
        P /= np.abs(P).max()                    # roughly -1..1

        s = sc[ok]
        sd = side[ok]
        reg = np.full(self.n_drawn, 6, np.int8)
        ol = np.char.startswith(s, "ol_")
        reg[ol & (sd == "L")] = 0
        reg[ol & (sd == "R")] = 1
        reg[np.char.startswith(s, "cb_")] = 2
        reg[np.char.startswith(s, "vnc_")] = 3
        reg[np.char.startswith(s, "visual_projection")] = 4
        reg[s == "visual_centrifugal"] = 4
        reg[s == "ascending_neuron"] = 6
        reg[s == "descending_neuron"] = 5
        self.reg = reg
        self.colors = np.stack([_hex(c) for _, c in REGIONS])       # (7,3)
        self.pt_col = self.colors[reg]                               # (n,3)

        # map full-brain neuron id -> row in the drawn arrays
        self.row_of = np.full(len(pos), -1, np.int64)
        self.row_of[self.idx] = np.arange(self.n_drawn)

        # the cells the pilot actually touches, for labelled markers
        self.marks = {}
        for name, types in (("LC4", ["LC4"]), ("LPLC2", ["LPLC2"]),
                            ("LC10a", ["LC10a"]), ("DNp01", ["DNp01"]),
                            ("DNa02", ["DNa02"])):
            r = self.row_of[np.flatnonzero(np.isin(ct, types))]
            self.marks[name] = r[r >= 0]

        self.P = P
        self.w, self.h = w, h
        # one scale that fits the cloud at ANY rotation
        self.radius = float(np.sqrt((P ** 2).sum(1)).max())
        self.scale = 0.46 * min(w, h) / self.radius
        self.yaw = -0.9
        self.pitch = 0.22
        self.spin = 0.0055
        self.frame = 0
        self.bg = np.zeros((h, w, 3), np.float32)
        self.heat = np.zeros((h, w, 3), np.float32)
        self.surf = pygame.Surface((w, h))
        self._px = None
        self._py = None
        self.rates = np.zeros(len(REGIONS), np.float32)
        self.reg_size = np.array([(reg == i).sum() for i in range(len(REGIONS))], np.float32)
        self.reg_size[self.reg_size == 0] = 1.0

    # ---------------------------------------------------------------- math
    def _project(self, Q):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        x = Q[:, 0] * cy + Q[:, 1] * sy
        yy = -Q[:, 0] * sy + Q[:, 1] * cy
        y = yy * cp - Q[:, 2] * sp
        depth = yy * sp + Q[:, 2] * cp
        scale = self.scale
        px = (x * scale + self.w * 0.5).astype(np.int32)
        py = (y * scale + self.h * 0.46).astype(np.int32)
        np.clip(px, 0, self.w - 1, out=px)
        np.clip(py, 0, self.h - 1, out=py)
        return px, py, depth

    def _splat(self, buf, px, py, col, gain=1.0):
        """gain is a scalar or one value per point — never a column vector, or
        the weights broadcast into an n x n array."""
        flat = py.astype(np.int64) * self.w + px
        n = self.w * self.h
        for c in range(3):
            buf[..., c] += np.bincount(flat, weights=col[:, c] * gain,
                                       minlength=n).reshape(self.h, self.w)

    # ---------------------------------------------------------------- frame
    def update(self, fired):
        """fired: indices into the full 166,700 neuron array."""
        self.frame += 1
        self.yaw += self.spin

        if self._px is None or self.frame % REPROJ == 0:
            px, py, depth = self._project(self.P)
            self._px, self._py = px, py
            d = (depth - depth.min()) / max(1e-6, float(depth.max() - depth.min()))
            self.bg[:] = 0.0
            self._splat(self.bg, px, py, self.pt_col, 0.030 + 0.045 * d)
            np.clip(self.bg, 0, 42, out=self.bg)

        rows = self.row_of[fired] if len(fired) else np.zeros(0, np.int64)
        rows = rows[rows >= 0]
        self.heat *= 0.74
        if len(rows):
            self._splat(self.heat, self._px[rows], self._py[rows],
                        self.pt_col[rows], 0.34)
            np.clip(self.heat, 0, 600, out=self.heat)
            counts = np.bincount(self.reg[rows], minlength=len(REGIONS))
        else:
            counts = np.zeros(len(REGIONS))
        # spikes per neuron per second, smoothed
        self.rates = 0.7 * self.rates + 0.3 * (counts / self.reg_size / (1 / 30))

    def draw(self, screen, x0, y0, font, fonts):
        # soft tone curve: dense regions stay coloured instead of clipping white
        img = (255.0 * (1.0 - np.exp(-(self.bg + self.heat) / 120.0))).astype(np.uint8)
        pygame.surfarray.blit_array(self.surf, np.transpose(img, (1, 0, 2)))
        screen.blit(self.surf, (x0, y0))

        marks = []
        for name, rows in self.marks.items():
            if not len(rows):
                continue
            mx = int(self._px[rows].mean())
            my = int(self._py[rows].mean())
            hot = self.heat[max(0, my - 6):my + 7, max(0, mx - 6):mx + 7].max()
            marks.append([name, mx, my, hot > 25])
        # labels would pile up on each other: stack them down the right side
        marks.sort(key=lambda m: m[2])
        ly = None
        for name, mx, my, on in marks:
            col = (255, 255, 255) if on else (128, 140, 160)
            r = 9 if on else 6
            pygame.draw.circle(screen, (8, 10, 14), (x0 + mx, y0 + my), r + 2, 3)
            pygame.draw.circle(screen, col, (x0 + mx, y0 + my), r, 2)
            ty = my if ly is None else max(my, ly + 15)
            ly = ty
            tx = mx + r + 16
            pygame.draw.line(screen, (60, 68, 84), (x0 + mx + r, y0 + my),
                             (x0 + tx - 4, y0 + ty), 1)
            lab = fonts.render(name, True, col)
            sh = fonts.render(name, True, (8, 10, 14))
            screen.blit(sh, (x0 + tx + 1, y0 + ty - 6))
            screen.blit(lab, (x0 + tx, y0 + ty - 7))

        y = y0 + self.h - len(REGIONS) * 15 - 14
        strip = pygame.Surface((self.w, len(REGIONS) * 15 + 14), pygame.SRCALPHA)
        strip.fill((9, 11, 15, 216))
        screen.blit(strip, (x0, y - 7))
        for i, (name, hexc) in enumerate(REGIONS):
            c = tuple(int(v) for v in self.colors[i])
            pygame.draw.rect(screen, c, (x0 + 10, y + 4, 7, 7))
            screen.blit(fonts.render(name, True, (150, 160, 178)), (x0 + 22, y))
            bw = int(min(1.0, self.rates[i] / 12.0) * 90)
            pygame.draw.rect(screen, (32, 38, 50), (x0 + 118, y + 3, 90, 8))
            if bw:
                pygame.draw.rect(screen, c, (x0 + 118, y + 3, bw, 8))
            screen.blit(fonts.render(f"{self.rates[i]:5.1f} Hz", True, (150, 160, 178)),
                        (x0 + 214, y))
            y += 15
