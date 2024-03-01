import pygame
import random
import sys
from pygame.locals import *

# 初期設定
pygame.init()
infoObject = pygame.display.Info()
screen_width, screen_height = infoObject.current_w, infoObject.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption('ドット抜け疑似表示テスト')

# 色定義
BLACK = (0, 0, 0)
RED = (255, 0, 0)

def draw_random_dot():
    x = random.randint(0, screen_width-1)
    y = random.randint(0, screen_height-1)
    screen.fill(BLACK)
    pygame.draw.circle(screen, RED, (x, y), 1)
    pygame.display.flip()
    return x, y

def main():
    correct_answers = 0
    try:
        for i in range(10):
            correct_x, correct_y = draw_random_dot()
            running = True
            start_ticks = pygame.time.get_ticks() # タイマー開始
            while running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == MOUSEBUTTONDOWN and event.button == 1:
                        x, y = event.pos
                        if abs(x - correct_x) <= 1 and abs(y - correct_y) <= 1:
                            print(f"正解！ 座標: ({correct_x}, {correct_y})")
                            correct_answers += 1
                        else:
                            print(f"不正解。正解座標: ({correct_x}, {correct_y})")
                        running = False
                seconds = (pygame.time.get_ticks() - start_ticks) / 1000
                if seconds > 15: # 15秒経過
                    print(f"時間切れ。正解座標: ({correct_x}, {correct_y})")
                    running = False
    finally:
        pygame.quit()
        print(f"テスト終了。正解数: {correct_answers}/10")

if __name__ == "__main__":
    main()
