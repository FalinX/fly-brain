"""Boxhead 2Play — pure game logic, no rendering, no input.

Fixed timestep so a brain loop can be tick-locked to it: every call to step()
advances exactly DT seconds regardless of how long the caller took. That is the
single thing the fly.ai SSH Fighter post-mortem says decides whether a slow
controller is usable at all.

Ammo is unlimited for every weapon.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

DT = 1.0 / 30.0
ARENA_W, ARENA_H = 960.0, 640.0
WALL = 18.0

# ---------------------------------------------------------------- weapons
@dataclass(frozen=True)
class Weapon:
    name: str
    cooldown: float      # seconds between shots
    pellets: int
    spread: float        # radians, half-angle
    speed: float
    damage: float
    life: float          # seconds a bullet stays alive


WEAPONS = [
    Weapon("pistol", 0.26, 1, 0.010, 620.0, 34.0, 1.1),
    Weapon("shotgun", 0.62, 7, 0.200, 520.0, 22.0, 0.55),
    Weapon("uzi", 0.075, 1, 0.055, 700.0, 15.0, 1.0),
]


@dataclass
class Bullet:
    x: float; y: float; vx: float; vy: float
    dmg: float; life: float; owner: int


@dataclass
class Zombie:
    x: float; y: float
    hp: float
    speed: float
    kind: str            # "walker" | "devil"
    hit_flash: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Barrel:
    x: float; y: float
    hp: float = 30.0
    fuse: float = -1.0   # >=0 once ignited


@dataclass
class Boom:
    x: float; y: float; t: float; r: float


@dataclass
class Player:
    pid: int
    x: float; y: float
    aim: float = 0.0          # radians, facing direction
    hp: float = 100.0
    weapon: int = 0
    cd: float = 0.0
    score: int = 0
    kills: int = 0
    shots: int = 0
    hits: int = 0
    dmg_barrel: float = 0.0     # damage taken from barrel blasts
    dmg_contact: float = 0.0    # damage taken from zombies touching it
    barrel_shots: int = 0       # own bullets that hit a barrel
    alive: bool = True
    hurt_flash: float = 0.0
    speed: float = 185.0
    # what the controller asked for this tick
    cmd_move: tuple[float, float] = (0.0, 0.0)
    cmd_turn: float = 0.0
    cmd_fire: bool = False
    cmd_swap: bool = False


# ---------------------------------------------------------------- world
class Game:
    def __init__(self, n_players: int = 2, seed: int = 7, start_wave: int = 0,
                 n_barrels: int = 7):
        """start_wave lets a collector drop straight into a crowded board.
        Escaping only decides anything once there are enough zombies to be
        surrounded by, and those frames are rare if every run starts at wave 1."""
        self.rng = random.Random(seed)
        self.t = 0.0
        self.wave = max(0, start_wave - 1)
        self.wave_timer = 2.0
        self.spawn_queue = 0
        self.spawn_cd = 0.0
        self.players: list[Player] = []
        pos = [(ARENA_W * 0.36, ARENA_H * 0.5), (ARENA_W * 0.64, ARENA_H * 0.5)]
        for i in range(n_players):
            self.players.append(Player(i, *pos[i], aim=0.0 if i else math.pi))
        self.zombies: list[Zombie] = []
        self.bullets: list[Bullet] = []
        self.barrels: list[Barrel] = []
        self.booms: list[Boom] = []
        self.over = False
        self.n_barrels = n_barrels
        self._seed_barrels()

    # ------------------------------------------------------------ setup
    def _seed_barrels(self):
        for _ in range(self.n_barrels):
            self.barrels.append(Barrel(
                self.rng.uniform(WALL + 60, ARENA_W - WALL - 60),
                self.rng.uniform(WALL + 60, ARENA_H - WALL - 60)))

    def _start_wave(self):
        self.wave += 1
        self.spawn_queue = 5 + int(self.wave * 2.6)
        self.spawn_cd = 0.0
        while len(self.barrels) < self.n_barrels:
            self.barrels.append(Barrel(
                self.rng.uniform(WALL + 60, ARENA_W - WALL - 60),
                self.rng.uniform(WALL + 60, ARENA_H - WALL - 60)))

    def _spawn_one(self):
        edge = self.rng.randrange(4)
        if edge == 0:
            x, y = self.rng.uniform(WALL, ARENA_W - WALL), WALL + 6
        elif edge == 1:
            x, y = self.rng.uniform(WALL, ARENA_W - WALL), ARENA_H - WALL - 6
        elif edge == 2:
            x, y = WALL + 6, self.rng.uniform(WALL, ARENA_H - WALL)
        else:
            x, y = ARENA_W - WALL - 6, self.rng.uniform(WALL, ARENA_H - WALL)
        devil = self.wave >= 3 and self.rng.random() < min(0.12 + 0.05 * self.wave, 0.45)
        if devil:
            self.zombies.append(Zombie(x, y, 55.0 + 4 * self.wave, 92.0, "devil"))
        else:
            self.zombies.append(Zombie(x, y, 38.0 + 3 * self.wave, 52.0, "walker"))

    # ------------------------------------------------------------ helpers
    def alive_players(self):
        return [p for p in self.players if p.alive]

    def nearest_player(self, x, y):
        best, bd = None, 1e18
        for p in self.alive_players():
            d = (p.x - x) ** 2 + (p.y - y) ** 2
            if d < bd:
                best, bd = p, d
        return best

    def _explode(self, x, y, radius=118.0, dmg=95.0, chain=True):
        self.booms.append(Boom(x, y, 0.0, radius))
        for z in list(self.zombies):
            if (z.x - x) ** 2 + (z.y - y) ** 2 <= radius * radius:
                z.hp -= dmg
                z.hit_flash = 0.12
        for p in self.alive_players():
            d2 = (p.x - x) ** 2 + (p.y - y) ** 2
            if d2 <= radius * radius:
                p.hp -= 26.0
                p.dmg_barrel += 26.0
                p.hurt_flash = 0.22
        if chain:
            for b in self.barrels:
                if b.fuse < 0 and (b.x - x) ** 2 + (b.y - y) ** 2 <= (radius * 1.15) ** 2:
                    b.fuse = 0.12

    # ------------------------------------------------------------ step
    def step(self):
        """Advance exactly DT seconds."""
        if self.over:
            return
        self.t += DT

        # -- waves
        if self.spawn_queue <= 0 and not self.zombies:
            self.wave_timer -= DT
            if self.wave_timer <= 0:
                self._start_wave()
                self.wave_timer = 3.0
        if self.spawn_queue > 0:
            self.spawn_cd -= DT
            if self.spawn_cd <= 0:
                self._spawn_one()
                self.spawn_queue -= 1
                self.spawn_cd = max(0.10, 0.5 - 0.03 * self.wave)

        # -- players
        for p in self.players:
            if not p.alive:
                continue
            p.hurt_flash = max(0.0, p.hurt_flash - DT)
            if p.cmd_swap:
                p.weapon = (p.weapon + 1) % len(WEAPONS)
            p.aim = (p.aim + p.cmd_turn * DT) % (2 * math.pi)
            # cmd_move is in the player's own frame: (forward, strafe-right)
            # along where it is facing. A fly walks along its body axis, so the
            # brain's steering output is what decides direction.
            fx, fy = p.cmd_move
            m = math.hypot(fx, fy)
            if m > 1e-6:
                fx, fy = fx / m, fy / m
                ca, sa = math.cos(p.aim), math.sin(p.aim)
                mx = fx * ca - fy * sa
                my = fx * sa + fy * ca
                p.x += mx * p.speed * DT
                p.y += my * p.speed * DT
                p.x = min(max(p.x, WALL + 12), ARENA_W - WALL - 12)
                p.y = min(max(p.y, WALL + 12), ARENA_H - WALL - 12)
            p.cd = max(0.0, p.cd - DT)
            if p.cmd_fire and p.cd <= 0.0:
                w = WEAPONS[p.weapon]
                p.cd = w.cooldown
                p.shots += 1
                for _ in range(w.pellets):
                    a = p.aim + self.rng.uniform(-w.spread, w.spread)
                    self.bullets.append(Bullet(
                        p.x + math.cos(a) * 20, p.y + math.sin(a) * 20,
                        math.cos(a) * w.speed, math.sin(a) * w.speed,
                        w.damage, w.life, p.pid))

        # -- bullets
        for b in list(self.bullets):
            b.life -= DT
            b.x += b.vx * DT
            b.y += b.vy * DT
            if (b.life <= 0 or b.x < WALL or b.x > ARENA_W - WALL
                    or b.y < WALL or b.y > ARENA_H - WALL):
                self.bullets.remove(b)
                continue
            hit = False
            for z in self.zombies:
                if (z.x - b.x) ** 2 + (z.y - b.y) ** 2 <= 17 * 17:
                    z.hp -= b.dmg
                    z.hit_flash = 0.1
                    z.vx += b.vx * 0.045
                    z.vy += b.vy * 0.045
                    self.players[b.owner].hits += 1
                    hit = True
                    break
            if hit:
                self.bullets.remove(b)
                continue
            for bar in self.barrels:
                if bar.fuse < 0 and (bar.x - b.x) ** 2 + (bar.y - b.y) ** 2 <= 16 * 16:
                    bar.hp -= b.dmg
                    self.players[b.owner].barrel_shots += 1
                    if bar.hp <= 0:
                        bar.fuse = 0.05
                    self.bullets.remove(b)
                    break

        # -- barrels
        for bar in list(self.barrels):
            if bar.fuse >= 0:
                bar.fuse -= DT
                if bar.fuse <= 0:
                    self.barrels.remove(bar)
                    self._explode(bar.x, bar.y)

        # -- zombies
        for z in list(self.zombies):
            z.hit_flash = max(0.0, z.hit_flash - DT)
            tgt = self.nearest_player(z.x, z.y)
            if tgt is not None:
                a = math.atan2(tgt.y - z.y, tgt.x - z.x)
                z.vx += math.cos(a) * z.speed * 6.0 * DT
                z.vy += math.sin(a) * z.speed * 6.0 * DT
            sp = math.hypot(z.vx, z.vy)
            if sp > z.speed:
                z.vx *= z.speed / sp
                z.vy *= z.speed / sp
            z.vx *= 0.90
            z.vy *= 0.90
            z.x += z.vx * DT
            z.y += z.vy * DT
            z.x = min(max(z.x, WALL), ARENA_W - WALL)
            z.y = min(max(z.y, WALL), ARENA_H - WALL)
            if z.hp <= 0:
                self.zombies.remove(z)
                near = self.nearest_player(z.x, z.y)
                if near:
                    near.kills += 1
                    near.score += 120 if z.kind == "devil" else 50
                continue
            if tgt is not None and (tgt.x - z.x) ** 2 + (tgt.y - z.y) ** 2 <= 24 * 24:
                dmg = (26.0 if z.kind == "devil" else 16.0) * DT
                tgt.hp -= dmg
                tgt.dmg_contact += dmg
                tgt.hurt_flash = 0.18

        # -- explosions fade
        for e in list(self.booms):
            e.t += DT
            if e.t > 0.35:
                self.booms.remove(e)

        # -- deaths
        for p in self.players:
            if p.alive and p.hp <= 0:
                p.alive = False
        if not self.alive_players():
            self.over = True

    # ------------------------------------------------------------ view
    def observe(self, pid: int):
        """Everything a controller is allowed to know. Angles are relative to
        where the player is facing: negative = to its left, positive = right."""
        p = self.players[pid]
        out = []
        for z in self.zombies:
            dx, dy = z.x - p.x, z.y - p.y
            dist = math.hypot(dx, dy) + 1e-6
            rel = (math.atan2(dy, dx) - p.aim + math.pi) % (2 * math.pi) - math.pi
            closing = -(dx * z.vx + dy * z.vy) / dist     # px/s, + = coming at me
            out.append({"rel": rel, "dist": dist, "closing": closing, "kind": z.kind})
        bars = []
        for b in self.barrels:
            dx, dy = b.x - p.x, b.y - p.y
            dist = math.hypot(dx, dy) + 1e-6
            rel = (math.atan2(dy, dx) - p.aim + math.pi) % (2 * math.pi) - math.pi
            bars.append({"rel": rel, "dist": dist, "fuse": b.fuse})
        return {"self": p, "zombies": out, "barrels": bars,
                "wave": self.wave, "t": self.t}
