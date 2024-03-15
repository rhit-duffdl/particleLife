import pygame as pg
from particle import Particles
import logging
from constants import SCREEN_HEIGHT, SCREEN_WIDTH, COLORS, MAX_VELOCITY, NUM_PARTICLES 

logging.basicConfig(level=logging.DEBUG)

pg.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption("Particle Life")


def main():
    running = True
    dt = 0

    particles = Particles(COLORS)
    print(particles.attraction_dict)
    logging.debug("starting game")
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        screen.fill(BLACK)

        particles.update_particles_multiprocess(dt)
        particles.draw_particles(screen)

        pg.display.flip()

        dt = pg.time.Clock().tick(60) / 100

    for p in particles.processes:
        p.terminate()
        p.join()


if __name__ == "__main__":
    main()
