import pygame
import sys

# Parameters
n = 10  # number of nodes in the 1D grid
target1 = 1
target2 = 2
cell_size = 30
margin = 5
font_size = 16

# Pygame init
pygame.init()
font = pygame.font.SysFont(None, font_size)
screen_width = n * (cell_size + margin)
screen_height = 100
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("1D OT Cost Delta Visualization")


def compute_deltas(n, t1, t2):
    #return [abs(i - t1) - abs(i - t2) for i in range(n)]
    return [(i - t1)**2 - (i - t2)**2 for i in range(n)]


def draw_grid(n, t1, t2):
    screen.fill((255, 255, 255))
    deltas = compute_deltas(n, t1, t2)
    for i in range(n):
        x = i * (cell_size + margin)
        y = 20
        rect = pygame.Rect(x, y, cell_size, cell_size)
        color = (200, 200, 255)
        if i == t1:
            color = (255, 100, 100)
        elif i == t2:
            color = (100, 255, 100)
        pygame.draw.rect(screen, color, rect)

        delta = deltas[i]
        text = font.render(str(delta), True, (0, 0, 0))
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)

    pygame.display.flip()


def main():
    global target1, target2
    clock = pygame.time.Clock()

    while True:
        clock.tick(30)
        draw_grid(n, target1, target2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and target1 > 0:
                    target1 -= 1
                elif event.key == pygame.K_RIGHT and target1 < n - 1:
                    target1 += 1
                elif event.key == pygame.K_UP and target2 > 0:
                    target2 -= 1
                elif event.key == pygame.K_DOWN and target2 < n - 1:
                    target2 += 1


if __name__ == "__main__":
    main()
