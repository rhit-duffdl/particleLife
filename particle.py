import random
import pygame as pg
import numpy as np
import multiprocessing
import os
import logging
from constants import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    COLORS,
    MAX_VELOCITY,
    NUM_PARTICLES,
    PARTICLE_RADIUS,
)


logging.basicConfig(level=logging.DEBUG)


class Particle:
    velocity: np.ndarray = np.array([0.0, 0.0], dtype=float)
    acceleration: np.ndarray = np.array([0.0, 0.0], dtype=float)
    position: np.ndarray = np.array([0.0, 0.0], dtype=float)
    attraction_radius = 50

    def __init__(
        self,
        x,
        y,
        color,
        radius=PARTICLE_RADIUS,
    ) -> None:
        self.position = np.array([x, y])
        self.color = color
        self.radius = radius
        self.velocity = np.array(
            [
                random.randrange(int(np.sqrt(MAX_VELOCITY))),
                random.randrange(int(np.sqrt(MAX_VELOCITY))),
            ]
        ).astype("float64")
        self.acceleration = np.array(
            [
                random.randrange(int(np.sqrt(MAX_VELOCITY))),
                random.randrange(int(np.sqrt(MAX_VELOCITY))),
            ]
        ).astype("float64")

    def draw(self, screen) -> None:
        # Draw particle
        pg.draw.circle(
            screen, self.color, (self.position[0], self.position[1]), self.radius
        )

        # Draw debug outline

    #        pg.draw.circle(
    #            screen,
    #            self.color,
    #            (self.position[0], self.position[1]),
    #            self.attraction_radius,
    #            5,
    #        )

    def cleanup(self):
        if self.position[0] > SCREEN_WIDTH:
            self.position[0] = SCREEN_WIDTH
            self.velocity[0] = abs(self.velocity[0]) * -1
        elif self.position[0] < 0:
            self.position[0] = 0
            self.velocity[0] = abs(self.velocity[0])

        if self.position[1] > SCREEN_HEIGHT:
            self.position[1] = SCREEN_HEIGHT
            self.velocity[1] = abs(self.velocity[1]) * -1
        elif self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] = abs(self.velocity[1])

        magnitude = np.linalg.norm(self.velocity)
        if magnitude > MAX_VELOCITY:
            self.velocity *= MAX_VELOCITY / magnitude


class Particles:
    colors: dict[str, list[Particle]] = {}
    attraction_dict: dict[tuple[str, str], float] = {}
    num_processes: int = os.cpu_count() - 1 or 4
    particles_chunk_queue: multiprocessing.Queue = multiprocessing.Queue()
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    processes: list[multiprocessing.Process] = []

    def __init__(
        self,
        colors: list[str],
        color_distribution: dict[str, int] = {},
        num_particles: int = NUM_PARTICLES,
    ) -> None:
        self.num_particles = num_particles
        if (
            color_distribution == {}
            or sum(color_distribution.values()) != num_particles
        ):
            color_distribution = {
                color: int(num_particles / len(colors)) for color in colors
            }

        for color in colors:
            particles = [
                Particle(
                    random.randrange(SCREEN_WIDTH),
                    random.randrange(SCREEN_HEIGHT),
                    color,
                )
                for _ in range(color_distribution[color])
            ]
            self.colors[color] = particles

            for other_color in colors:
                if (
                    other_color != color
                    and (other_color, color) not in self.attraction_dict
                ):
                    self.attraction_dict[(other_color, color)] = int(
                        random.random() * (MAX_VELOCITY**2) * random.choice([1, -1])
                    )
        self.processes = [
            multiprocessing.Process(
                target=self.calculate_forces_worker,
                args=(self.particles_chunk_queue, self.result_queue),
            )
            for _ in range(self.num_processes)
        ]

        logging.debug("initialized processes, starting now")

        for p in self.processes:
            p.start()

    def calculate_forces_worker(self, particles_chunk_queue, result_queue):
        while True:
            particles_chunk, other_particles_chunk, dt = particles_chunk_queue.get(
                timeout=2
            )
            partial_forces = []
            for p1 in particles_chunk:
                net_force = np.array([0.0, 0.0], dtype=float)
                for p2 in other_particles_chunk:
                    if p1.color == p2.color:
                        continue
                    force_magnitude = self.attraction_dict[(p1.color, p2.color)]
                    distance = np.linalg.norm(p2.position - p1.position)
                    if distance > p1.attraction_radius:
                        continue

                    direction = ((p2.position - p1.position) / max(1, distance)).astype(
                        "float64"
                    )
                    force = (direction * force_magnitude / max(1, distance)).astype(
                        "float64"
                    )
                    net_force += force
                partial_forces.append(net_force)
            result_queue.put(partial_forces)

    def update_particles_multiprocess(self, dt):
        all_pairs = []
        for color, particles in self.colors.items():
            for other_color, other_particles in self.colors.items():
                if color == other_color:
                    continue
                for p1 in particles:
                    for p2 in other_particles:
                        distance = np.linalg.norm(p2.position - p1.position)
                        if distance > p1.attraction_radius:
                            continue
                        all_pairs.append((p1, p2))

        chunk_size = len(all_pairs) // self.num_processes
        chunks = (
            []
            if chunk_size == 0
            else [
                all_pairs[i : i + chunk_size]
                for i in range(0, len(all_pairs), chunk_size)
            ]
        )
        for chunk in chunks:
            particles_chunk = [particle for (particle, _) in chunk]
            other_particles_chunk = [other_particle for (_, other_particle) in chunk]
            self.particles_chunk_queue.put((particles_chunk, other_particles_chunk, dt))

        for chunk in chunks:
            particles_chunk = [particle for (particle, _) in chunk]
            other_particles_chunk = [other_particle for (_, other_particle) in chunk]
            forces = self.result_queue.get()
            for particle, force in zip(particles_chunk, forces):
                acceleration = np.array(force, dtype="float64")
                particle.velocity += acceleration * dt

        for _, particles in self.colors.items():
            for particle in particles:
                particle.position = (particle.position + particle.velocity * dt).astype(
                    "float64"
                )
                particle.cleanup()

    def draw_particles(self, screen):
        for _, particles in self.colors.items():
            for particle in particles:
                particle.draw(screen)
