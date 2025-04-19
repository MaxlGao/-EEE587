import numpy as np
import pygame
import time

def rect_edge_point(width, height, phi):
    """
    Computes the (x, y) coordinates of the intersection of a ray from the center of a rectangle
    with angle phi (which can be a column vector of angles) and the rectangle's edge.

    The function computes the distances to the rectangle's edges using trigonometric functions
    and returns the coordinates of the intersection point, as well as the normal and tangential
    vectors at those points.

    :param width: Width of the rectangle.
    :param height: Height of the rectangle.
    :param phi: Array or scalar of angles (in radians).
    :return: (x, y) coordinates of the intersection points, normal vectors, tangential vectors.
    """
    # Convert phi to a numpy array for broadcasting
    phi = np.asarray(phi)
    
    # Compute cos and sin for all phi values
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Compute distances to the rectangle's edges
    dx = np.where(cos_phi != 0, np.abs(width / 2 / cos_phi), np.inf)
    dy = np.where(sin_phi != 0, np.abs(height / 2 / sin_phi), np.inf)
    
    # Find the minimum distances (scale factor) for each angle
    t = np.minimum(dx, dy)
    
    # Compute the intersection points
    x = t * cos_phi
    y = t * sin_phi
    
    # Determine normal and tangential direction based on the edge hit
    normal_x = np.where(dx < dy, np.copysign(1, -cos_phi), 0)
    normal_y = np.where(dx >= dy, np.copysign(1, -sin_phi), 0)
    
    tangential_x = np.where(dx < dy, 0, np.copysign(1, sin_phi))
    tangential_y = np.where(dx >= dy, 0, np.copysign(1, -cos_phi))
    
    normal = np.stack((normal_x, normal_y), axis=-1)
    tangential = np.stack((tangential_x, tangential_y), axis=-1)
    
    return x, y, normal, tangential

def transform2(x,y,angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    new_x = x * cos_angle - y * sin_angle
    new_y = x * sin_angle + y * cos_angle
    return new_x, new_y

def playback(x_trajectory, u_trajectory, x_ref_values, B_WIDTH, B_HEIGHT, dt, speed):
    # Pygame animation
    # Initialize pygame
    pygame.init()

    # Screen dimensions
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Rectangle Animation")

    # Colors
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLACK = (0,0,0)

    # Clock for controlling frame rate
    clock = pygame.time.Clock()

    trajectory_len = len(u_trajectory)

    running = True
    index = 0
    while running:
        # This entire script assumes the origin is at TOP LEFT

        start_time = time.time()
        screen.fill(WHITE)  # Clear screen
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get current position and angle from the trajectory
        bx = x_trajectory[index][0]
        by = x_trajectory[index][1]
        angle = x_trajectory[index][2]
        # print(angle)
        target_bx = x_ref_values[0, index]
        target_by = x_ref_values[1, index]
        target_angle = x_ref_values[2, index]
        # print(target_angle)
        phi1 = x_trajectory[index][3]
        phi2 = x_trajectory[index][4]
        norm1 = u_trajectory[index][0]
        norm2 = u_trajectory[index][1]
        tang1 = u_trajectory[index][2]
        tang2 = u_trajectory[index][3]

        # print("x trajectory:", x_trajectory[index])
        # print("u trajectory:", u_trajectory[index])

        # Positions of point pushers on the body frame
        base_p1x, base_p1y, nor_dir1, tan_dir1 = rect_edge_point(B_WIDTH, B_HEIGHT, phi1)
        base_p2x, base_p2y, nor_dir2, tan_dir2 = rect_edge_point(B_WIDTH, B_HEIGHT, phi2)

        # Force vectors
        norm1 = norm1 * nor_dir1
        norm2 = norm2 * nor_dir2
        tang1 = tang1 * tan_dir1
        tang2 = tang2 * tan_dir2

        # Converting pusher position to base frame
        p1x, p1y = transform2(base_p1x, base_p1y, angle)
        p1x, p1y = (float(p1x + bx), float(p1y + by)) 
        p2x, p2y = transform2(base_p2x, base_p2y, angle)
        p2x, p2y = (float(p2x + bx), float(p2y + by)) 

        # print("p1 pos:", p1x, p1y)
        # print("p2 pos:", p2x, p2y)

        # Converting pusher forces to base frame
        norm1 = (float(norm1[0,0] * np.cos(angle) - norm1[0,1] * np.sin(angle)),
                 float(norm1[0,0] * np.sin(angle) + norm1[0,1] * np.cos(angle)))
        norm2 = (float(norm2[0,0] * np.cos(angle) - norm2[0,1] * np.sin(angle)),
                 float(norm2[0,0] * np.sin(angle) + norm2[0,1] * np.cos(angle)))
        tang1 = (float(tang1[0,0] * np.cos(angle) - tang1[0,1] * np.sin(angle)),
                 float(tang1[0,0] * np.sin(angle) + tang1[0,1] * np.cos(angle)))
        tang2 = (float(tang2[0,0] * np.cos(angle) - tang2[0,1] * np.sin(angle)),
                 float(tang2[0,0] * np.sin(angle) + tang2[0,1] * np.cos(angle)))
        

        # Create a rotated rectangle
        rect = pygame.Surface((1000*B_WIDTH, 1000*B_HEIGHT), pygame.SRCALPHA)
        rect.fill(BLUE)
        rotated_rect = pygame.transform.rotate(rect, -angle*180/np.pi)
        rect_rect = rotated_rect.get_rect(center=(1000*bx, 1000*by))
        
        # Draw the rotated rectangle
        screen.blit(rotated_rect, rect_rect.topleft)

        # Create two points
        pygame.draw.circle(screen, RED, (1000*p1x, 1000*p1y), 5)
        pygame.draw.circle(screen, RED, (1000*p2x, 1000*p2y), 5)

        # Draw vectors
        scale = 0.1
        pygame.draw.line(screen, GREEN, (1000*p1x, 1000*p1y),
                                        (1000*(p1x + norm1[0]*scale),
                                         1000*(p1y + norm1[1]*scale)), 5)
        pygame.draw.line(screen, GREEN, (1000*p1x, 1000*p1y),
                                        (1000*(p1x + tang1[0]*scale),
                                         1000*(p1y + tang1[1]*scale)), 5)
        pygame.draw.line(screen, GREEN, (1000*p2x, 1000*p2y),
                                        (1000*(p2x + norm2[0]*scale),
                                         1000*(p2y + norm2[1]*scale)), 5)
        pygame.draw.line(screen, GREEN, (1000*p2x, 1000*p2y),
                                        (1000*(p2x + tang2[0]*scale),
                                         1000*(p2y + tang2[1]*scale)), 5)
        
        # Draw past trajectory
        if index > 1:
            bx, by = x_trajectory[0][:2]
            points = [(1000*arr[0][0], 1000*arr[1][0]) for arr in x_trajectory[:index]]
            pygame.draw.lines(screen, BLACK, False, points, 2)
            pygame.draw.circle(screen, BLACK, (int(1000*bx), int(1000*by)), 5)

        # Draw the target
        rect_points = np.array([[-500*B_WIDTH, -500*B_HEIGHT], [500*B_WIDTH, -500*B_HEIGHT], 
                                [500*B_WIDTH, 500*B_HEIGHT], [-500*B_WIDTH, 500*B_HEIGHT]])
        rot_matrix = np.array([[np.cos(-target_angle), -np.sin(-target_angle)], 
                               [np.sin(-target_angle), np.cos(-target_angle)]])
        rect_points = rect_points @ rot_matrix + np.array([1000*target_bx, 1000*target_by])
        pygame.draw.polygon(screen, BLACK, rect_points, 2)

        index += 1
        if index >= trajectory_len:
            index = 0  # Loop back to the first point

        if index % 10 == 0:
            print(index)
        # Refresh display
        pygame.display.flip()
        elapsed = time.time() - start_time
        time.sleep(max(0, dt - elapsed) / speed)
        clock.tick(60)  # Limit to 60 FPS

    pygame.quit()