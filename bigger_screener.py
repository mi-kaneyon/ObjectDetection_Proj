import pygame
import tkinter as tk
import random
import sys

# Initialize pygame and tkinter
pygame.init()
root = tk.Tk()
root.attributes('-fullscreen', True)

screen_width, screen_height = 1920, 1080
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("Dot Screener Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
COLORS = [BLACK, RED, GREEN, BLUE, YELLOW]

# Game variables
correct_answers = 0
total_questions = 10
dot_positions = []

def draw_dot():
    color = random.choice(COLORS)
    size = random.randint(1, 10)  # ドットのサイズもランダムに決定
    x = random.randint(0, screen_width - size)
    y = random.randint(0, screen_height - size)
    pygame.draw.rect(screen, color, (x, y, size, size))
    return (x, y), color, size

def game_loop():
    global correct_answers
    for _ in range(total_questions):
        screen.fill(WHITE)
        dot_position, dot_color, dot_size = draw_dot()
        dot_positions.append((dot_position, dot_color, dot_size))
        pygame.display.update()
        
        found = False
        start_ticks = pygame.time.get_ticks()
        while not found:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    # ドットの範囲内にクリックがあったかどうかをチェック
                    if dot_position[0] <= mouse_x <= dot_position[0] + dot_size and dot_position[1] <= mouse_y <= dot_position[1] + dot_size:
                        correct_answers += 1
                        found = True
                    else:
                        found = True
            if (pygame.time.get_ticks() - start_ticks) / 1000 > 60:
                break
        
        if found:
            print(f"Found at {dot_position}. Size: {dot_size}, Color: {dot_color}.")
        else:
            print(f"Time's up. The correct position was {dot_position}. Size: {dot_size}, Color: {dot_color}.")

def main():
    try:
        game_loop()
        accuracy = (correct_answers / total_questions) * 100
        print(f"Game Over. Accuracy: {accuracy}%. Correct answers: {correct_answers} out of {total_questions}.")
        for pos, color, size in dot_positions:
            print(f"Dot position: {pos}, Size: {size}, Color: {color}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
