import pygame
import random
import sys
from pygame.locals import *

# Initialization
pygame.init()
infoObject = pygame.display.Info()
screen_width, screen_height = infoObject.current_w, infoObject.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption('LCD Dead Pixel Simulation Test')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
COLORS = [WHITE, RED, GREEN, BLUE, YELLOW]  # Include WHITE for the dot color

def draw_random_dots():
    screen.fill(BLACK)
    correct_positions = []
    for _ in range(random.randint(1, 5)):
        x = random.randint(0, screen_width-1)
        y = random.randint(0, screen_height-1)
        color = random.choice(COLORS)
        pygame.draw.circle(screen, color, (x, y), 1)
        correct_positions.append((x, y, color))
    pygame.display.flip()
    return correct_positions

def main():
    correct_answers = 0
    try:
        for i in range(10):
            correct_positions = draw_random_dots()
            running = True
            start_ticks = pygame.time.get_ticks()  # Start timer
            while running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == MOUSEBUTTONDOWN and event.button == 1:
                        x, y = event.pos
                        for correct_x, correct_y, dot_color in correct_positions:
                            # Check if the click is within 4 pixels of the correct dot
                            if abs(x - correct_x) <= 4 and abs(y - correct_y) <= 4:
                                print(f"Correct! Coordinates: ({correct_x}, {correct_y})")
                                correct_answers += 1
                                break
                        else:
                            print(f"Incorrect. Correct coordinates: {correct_positions}")
                        running = False
                seconds = (pygame.time.get_ticks() - start_ticks) / 1000
                if seconds > 60:  # 15 seconds passed
                    print(f"Time's up. Correct coordinates: {correct_positions}")
                    running = False
    finally:
        pygame.quit()
        print(f"Test completed. Correct answers: {correct_answers}/10")

if __name__ == "__main__":
    main()
