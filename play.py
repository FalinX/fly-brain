"""Boxhead 2Play — run it.

    python play.py                     you (P1) + the fly brain (P2)
    python play.py --p1 fly --p2 none  watch the fly alone
    python play.py --p1 script --p2 fly   baseline vs fly
    python play.py --p1 fly --p2 none --record clip.mp4 --seconds 60

P1 controls: WASD move, mouse aim, left click / space fire, Q swap weapon.
Ammo is unlimited.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from game import Game, DT, ARENA_W, ARENA_H, WALL, WEAPONS
from pilots import FlyPilot, ScriptPilot, HumanPilot
import versions
from brainview import BrainView

SCALE = 1.0
PANEL = 300
BRAINW = 460
W, H = int(ARENA_W + BRAINW + PANEL), int(ARENA_H)

BG = (13, 15, 20)
FLOOR = (22, 26, 34)
GRID = (30, 35, 46)
WALLC = (52, 60, 76)
INK = (228, 233, 241)
MUTED = (126, 136, 154)
P1C = (88, 190, 255)
P2C = (255, 206, 84)
ZC = (118, 178, 96)
DEVC = (226, 92, 88)
BUL = (255, 240, 180)
BARC = (196, 110, 52)
BOOMC = (255, 176, 66)
BRAINC = (120, 226, 190)
HANDC = (150, 158, 176)


VERSION = None


def make_pilot(kind):
    if kind == "fly":
        v = VERSION or versions.get("")
        return FlyPilot(readout=v["readout"], encoder=v["encoder"],
                        barrel_gate=v.get("barrel_gate", "none"),
                        escape_mode=v.get("escape_mode", "hand"),
                        move_mode=v.get("move_mode", "reverse"), label=v["id"])
    if kind == "script":
        return ScriptPilot()
    if kind == "human":
        return HumanPilot()
    return None


PILOT_COLOR = {'script': (150, 158, 176), 'human': (88, 190, 255)}
FLY_COLOR = (255, 206, 84)


def draw_arena(s, g, t, pilots=None):
    s.fill(BG)
    pygame.draw.rect(s, FLOOR, (0, 0, ARENA_W, ARENA_H))
    for x in range(0, int(ARENA_W), 48):
        pygame.draw.line(s, GRID, (x, 0), (x, ARENA_H))
    for y in range(0, int(ARENA_H), 48):
        pygame.draw.line(s, GRID, (0, y), (ARENA_W, y))
    pygame.draw.rect(s, WALLC, (0, 0, ARENA_W, ARENA_H), int(WALL))

    for b in g.barrels:
        c = (255, 120, 60) if b.fuse >= 0 else BARC
        pygame.draw.rect(s, c, (b.x - 11, b.y - 11, 22, 22), border_radius=3)
        pygame.draw.rect(s, (18, 20, 26), (b.x - 11, b.y - 11, 22, 22), 2, border_radius=3)

    for z in g.zombies:
        c = DEVC if z.kind == "devil" else ZC
        if z.hit_flash > 0:
            c = (255, 255, 255)
        r = 13 if z.kind == "walker" else 12
        pygame.draw.rect(s, c, (z.x - r, z.y - r, 2 * r, 2 * r), border_radius=2)
        pygame.draw.rect(s, (12, 14, 18), (z.x - r, z.y - r, 2 * r, 2 * r), 2, border_radius=2)

    for b in g.bullets:
        pygame.draw.line(s, BUL, (b.x, b.y), (b.x - b.vx * 0.014, b.y - b.vy * 0.014), 2)

    for e in g.booms:
        k = e.t / 0.35
        r = int(e.r * (0.45 + 0.55 * k))
        surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*BOOMC, int(170 * (1 - k))), (r, r), r)
        s.blit(surf, (e.x - r, e.y - r))

    for i, p in enumerate(g.players):
        nm = pilots[i].name if pilots and pilots[i] else ''
        c = PILOT_COLOR.get(nm, FLY_COLOR if nm.startswith('v') or nm.startswith('fly') else P1C)
        if not p.alive:
            c = (70, 74, 84)
        if p.hurt_flash > 0:
            c = (255, 130, 130)
        pygame.draw.rect(s, c, (p.x - 14, p.y - 14, 28, 28), border_radius=3)
        pygame.draw.rect(s, (12, 14, 18), (p.x - 14, p.y - 14, 28, 28), 2, border_radius=3)
        if p.alive:
            pygame.draw.line(s, c, (p.x, p.y),
                             (p.x + math.cos(p.aim) * 30, p.y + math.sin(p.aim) * 30), 4)
            pygame.draw.circle(s, (255, 255, 255),
                               (p.x + math.cos(p.aim) * 30, p.y + math.sin(p.aim) * 30), 3)


def bar(s, f, x, y, w, h, frac, col, label, val, tag=None):
    pygame.draw.rect(s, (30, 34, 44), (x, y, w, h), border_radius=2)
    fw = int(w * max(0.0, min(1.0, frac)))
    if fw:
        pygame.draw.rect(s, col, (x, y, fw, h), border_radius=2)
    s.blit(f.render(label, True, MUTED), (x, y - 15))
    txt = f.render(val, True, INK)
    s.blit(txt, (x + w - txt.get_width(), y - 15))
    if tag:
        tg = f.render(tag, True, BRAINC if tag == "BRAIN" else HANDC)
        s.blit(tg, (x + w + 8, y - 1))


def draw_panel(s, fonts, g, pilots, fps, brain_ms):
    f, fb, fs = fonts
    x0 = int(ARENA_W) + BRAINW + 18
    pygame.draw.rect(s, (17, 20, 26), (ARENA_W + BRAINW, 0, PANEL, ARENA_H))
    pygame.draw.line(s, (36, 42, 54), (ARENA_W + BRAINW, 0), (ARENA_W + BRAINW, ARENA_H))

    y = 18
    s.blit(fb.render("BOXHEAD 2PLAY", True, INK), (x0, y)); y += 22
    s.blit(fs.render("unlimited ammo", True, MUTED), (x0, y)); y += 24
    s.blit(f.render(f"wave {g.wave}   {g.t:5.1f}s", True, INK), (x0, y)); y += 20
    s.blit(fs.render(f"{len(g.zombies)} zombies alive", True, MUTED), (x0, y)); y += 26

    for i, p in enumerate(g.players):
        nm = pilots[i].name if pilots[i] else "-"
        c = PILOT_COLOR.get(nm, FLY_COLOR if nm.startswith('v') or nm.startswith('fly') else P1C)
        s.blit(f.render(f"P{i+1}  {nm}", True, c), (x0, y)); y += 18
        pygame.draw.rect(s, (30, 34, 44), (x0, y, 240, 9), border_radius=2)
        pygame.draw.rect(s, c if p.alive else (90, 40, 40),
                         (x0, y, int(240 * max(0, p.hp) / 100), 9), border_radius=2)
        y += 15
        acc = 100 * p.hits / max(1, p.shots)
        s.blit(fs.render(f"{p.kills} kills   {p.score} pts   {acc:4.1f}% hit",
                         True, MUTED), (x0, y)); y += 13
        s.blit(fs.render(f"weapon: {WEAPONS[p.weapon].name}", True, MUTED), (x0, y))
        y += 22

    fly = next((p for p in pilots if isinstance(p, FlyPilot)), None)
    if fly is None:
        return
    y += 8
    pygame.draw.line(s, (36, 42, 54), (x0, y), (x0 + 240, y)); y += 12
    s.blit(fb.render("THE FLY BRAIN", True, INK), (x0, y)); y += 17
    s.blit(fs.render(f"{fly.n:,} neurons · {len(fly.weights)//1000000}.6M synapses",
                     True, MUTED), (x0, y)); y += 13
    s.blit(fs.render(f"{brain_ms:4.1f} ms / frame   budget {DT*1000:.0f} ms" + (f"   {fps:.0f} fps" if fps > 1 else ""),
                     True, BRAINC if brain_ms < DT * 1000 else DEVC), (x0, y)); y += 24

    s.blit(fs.render("WHAT IT SEES   voltage injected", True, MUTED), (x0, y)); y += 30
    d = fly.see
    for key, lab in (("loom", "LPLC2 looming"), ("threat", "LC4 threat"),
                     ("chase", "LC10a target")):
        l, r = d.get(key + "_L", 0.0), d.get(key + "_R", 0.0)
        pygame.draw.rect(s, (30, 34, 44), (x0, y, 240, 10), border_radius=2)
        pygame.draw.rect(s, (90, 150, 220), (x0 + 120 - int(120 * l / 0.8), y,
                                             int(120 * l / 0.8), 10))
        pygame.draw.rect(s, (220, 140, 90), (x0 + 120, y, int(120 * r / 0.8), 10))
        pygame.draw.line(s, (60, 66, 80), (x0 + 120, y), (x0 + 120, y + 10))
        s.blit(fs.render(lab, True, MUTED), (x0, y - 13))
        s.blit(fs.render("L | R", True, (70, 76, 92)), (x0 + 205, y - 13))
        y += 30

    y += 10
    s.blit(fs.render("WHAT IT DOES   descending neurons", True, MUTED), (x0, y)); y += 32
    t = fly.trace
    bar(s, fs, x0, y, 170, 11, abs(t["steer_R"] - t["steer_L"]) / 6.0,
        BRAINC, "DNa02 steer", f"{t['steer_R']-t['steer_L']:+.1f} Hz", "BRAIN"); y += 32
    bar(s, fs, x0, y, 170, 11, t["escape"] / 20.0,
        (255, 120, 110) if fly.fleeing else BRAINC, "DNp01 escape",
        f"{t['escape']:.1f} Hz", "BRAIN"); y += 32
    bar(s, fs, x0, y, 170, 11, t["forward"] / 20.0, (70, 76, 92),
        "DNg100 forward", f"{t['forward']:.1f} Hz", "dead"); y += 32
    bar(s, fs, x0, y, 170, 11, t["punch"] / 20.0, (70, 76, 92),
        "DNg11 punch", f"{t['punch']:.1f} Hz", "noise"); y += 34

    s.blit(fs.render("BRAIN = read from spikes", True, BRAINC), (x0, y)); y += 13
    s.blit(fs.render("dead / noise = not wired to a key", True, HANDC), (x0, y)); y += 13
    s.blit(fs.render("walk + trigger are hand rules", True, HANDC), (x0, y)); y += 20


def human_cmd(p, keys, mouse, fire):
    wx = (keys[pygame.K_d] - keys[pygame.K_a])
    wy = (keys[pygame.K_s] - keys[pygame.K_w])
    aim = math.atan2(mouse[1] - p.y, mouse[0] - p.x)
    da = (aim - p.aim + math.pi) % (2 * math.pi) - math.pi
    ca, sa = math.cos(p.aim), math.sin(p.aim)
    fx = wx * ca + wy * sa
    fy = -wx * sa + wy * ca
    return {"move": (fx, fy), "turn": da / DT, "fire": fire, "swap": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default="human", choices=["human", "fly", "script"])
    ap.add_argument("--p2", default="fly", choices=["fly", "script", "human", "none"])
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--record", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--speed", type=float, default=1.0, help="wall-clock multiplier")
    ap.add_argument("--fly", default="", help="fly version id, e.g. v1 v2 v3 v4")
    ap.add_argument("--barrels", type=int, default=0,
                    help="explosive barrels on the map (0 = none, benchmarks used 7)")
    a = ap.parse_args()

    global VERSION
    VERSION = versions.get(a.fly)
    print(f"fly version: {VERSION['id']} — {VERSION['headline']}")
    n = 1 if a.p2 == "none" else 2
    g = Game(n_players=n, seed=a.seed, n_barrels=a.barrels)
    pilots = [make_pilot(a.p1)] + ([make_pilot(a.p2)] if n == 2 else [])

    pygame.init()
    pygame.display.set_caption("Boxhead 2Play — driven by a real fruit-fly connectome")
    flags = pygame.HIDDEN if a.record else 0
    screen = pygame.display.set_mode((W, H), flags)
    clock = pygame.time.Clock()
    fonts = (pygame.font.SysFont("consolas", 15),
             pygame.font.SysFont("consolas", 16, bold=True),
             pygame.font.SysFont("consolas", 12))

    writer = None
    if a.record:
        import subprocess
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(int(1 / DT)), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", a.record],
            stdin=subprocess.PIPE)

    bv = None
    if any(isinstance(p, FlyPilot) for p in pilots):
        bv = BrainView(BRAINW, int(ARENA_H))

    fire_held = False
    brain_ms = 0.0
    running = True
    frames = int(a.seconds / DT) if a.seconds else 10 ** 9
    i = 0
    while running and i < frames:
        i += 1
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                if ev.key == pygame.K_q and isinstance(pilots[0], HumanPilot):
                    g.players[0].cmd_swap = True
                if ev.key == pygame.K_SPACE:
                    fire_held = True
            elif ev.type == pygame.KEYUP and ev.key == pygame.K_SPACE:
                fire_held = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                fire_held = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                fire_held = False

        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pos()

        for pid, pilot in enumerate(pilots):
            p = g.players[pid]
            if not p.alive or pilot is None:
                continue
            obs = g.observe(pid)
            if isinstance(pilot, HumanPilot):
                cmd = human_cmd(p, keys, mouse, fire_held)
            else:
                t0 = time.perf_counter()
                cmd = pilot.act(obs)
                if isinstance(pilot, FlyPilot):
                    brain_ms = 0.85 * brain_ms + 0.15 * (time.perf_counter() - t0) * 1000
            swap = cmd["swap"] or p.cmd_swap
            p.cmd_move, p.cmd_turn = cmd["move"], cmd["turn"]
            p.cmd_fire, p.cmd_swap = cmd["fire"], swap

        g.step()
        for p in g.players:
            p.cmd_swap = False

        draw_arena(screen, g, g.t, pilots)
        if bv is not None:
            pygame.draw.rect(screen, (9, 11, 15), (ARENA_W, 0, BRAINW, ARENA_H))
            fly0 = next((p for p in pilots if isinstance(p, FlyPilot)), None)
            bv.update(fly0.fired if fly0 is not None else [])
            bv.draw(screen, int(ARENA_W), 0, fonts[0], fonts[2])
            screen.blit(fonts[1].render("LIVE CONNECTOME", True, INK),
                        (ARENA_W + 14, 12))
            screen.blit(fonts[2].render(
                f"{bv.n_drawn:,} neurons - each lights up on the frame it spikes",
                True, MUTED), (ARENA_W + 14, 31))
            pygame.draw.line(screen, (36, 42, 54), (ARENA_W, 0), (ARENA_W, ARENA_H))
        draw_panel(screen, fonts, g, pilots, clock.get_fps(), brain_ms)
        if g.over:
            f = pygame.font.SysFont("consolas", 40, bold=True)
            t = f.render("GAME OVER", True, (255, 120, 120))
            screen.blit(t, (ARENA_W / 2 - t.get_width() / 2, ARENA_H / 2 - 20))
        pygame.display.flip()

        if writer is not None:
            arr = pygame.surfarray.array3d(screen).swapaxes(0, 1)
            writer.stdin.write(arr.tobytes())
        else:
            clock.tick(int(1 / DT * a.speed))

        if g.over and (a.seconds or writer):
            break

    if writer is not None:
        writer.stdin.close()
        writer.wait()
        print("wrote", a.record)
    for pid, p in enumerate(g.players):
        print(f"P{pid+1} {pilots[pid].name if pilots[pid] else '-':7s} "
              f"survived {g.t:5.1f}s  wave {g.wave}  kills {p.kills}  "
              f"score {p.score}  acc {100*p.hits/max(1,p.shots):4.1f}%")
    pygame.quit()


if __name__ == "__main__":
    main()
