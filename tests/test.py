import pygame

# Initialize pygame
pygame.init()

# Create a window
screen = pygame.display.set_mode((600, 600))

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)


# Fill the screen with white
screen.fill(WHITE)

# Draw a red line from (50, 50) to (450, 450) with thickness 5
pygame.draw.rect(screen, RED, (50, 50, 10, 10))
pygame.draw.circle(screen, RED, (200, 200), 10)

# Update display
pygame.display.flip()

# Keep the window open for a few seconds
pygame.time.wait(3000)

# Quit pygame
pygame.quit()
