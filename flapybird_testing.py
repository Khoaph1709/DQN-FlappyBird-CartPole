import flappy_bird_gymnasium
import gymnasium
import pygame 

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
clock = pygame.time.Clock()

obs, _ = env.reset() 
running = True
while running:
    action = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1
    
    obs, reward, terminated, _, info = env.step(action)
    
    clock.tick(60)
    
    if terminated:
        running = False

env.close()
pygame.quit()