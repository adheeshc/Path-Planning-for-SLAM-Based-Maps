import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('SLAM_MAP_THRESH.png')

imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

for i in contours : 
    for j in i :
        cv2.circle(imgray,(j[0][0],j[0][1]),16,(0,0,0),-1)
        

cv2.imshow('MAP', imgray)

x = imgray.shape[1]
y = imgray.shape[0]
xv, yv = np.meshgrid(range(x), range(y))
xv = xv.reshape(-1)
yv = yv.reshape(-1)

Z = np.zeros((y,x))

for i in range(0,len(xv)):
        if imgray[yv[i]][xv[i]] < 30:
            Z[yv[i]][xv[i]] = False
        else:
            Z[yv[i]][xv[i]] = True
            
empty = np.zeros([np.shape(Z)[0],np.shape(Z)[1],3], dtype = np.uint8)
empty[:] = 255
empty[Z==False]=[200,200,0]
            
#cv2.imshow('Z',Z)


def diff_constraint(ul, ur, x, y, dt, theta):
    r = 0.038
    L = 0.3175
    ul = (ul*2*np.pi)/60
    ur = (ur*2*np.pi)/60
    step = dt/50
    
    dt_int = np.arange(0, dt+step, step)
    
    theta_dot = (r/L)*(ur-ul)
    theta_new = theta_dot*dt_int + theta
      
    x_dot = (r/2)*(ul+ur)*np.cos(theta_new)
    y_dot = (r/2)*(ul+ur)*np.sin(theta_new)
    
    x_new = x_dot*dt_int + x
    y_new = y_dot*dt_int + y

    total_cost = []
    
    x_cost = x
    y_cost = y
           
    for _ in range(len(x_new)):
        if Z[int(y_new[_])][int(x_new[_])]== False:
            return False
        
    for i in range(len(x_new)):
        total_cost.append(cost(x_cost, y_cost, x_new[i], y_new[i]))
        x_cost = x_new[i]
        y_cost = y_new[i]
        
    cost_return = np.sum(total_cost)
    
    return [round(x_new[-1],1), round(y_new[-1],1), round(theta_new[-1],1), round(cost_return,1)]

def getMin(costG, visited,Q, angles, rpms_robot):
    t = None
    for v in Q:
        if v in costG:
            if t==None:
                t=v
            elif (costG[v]<costG[t]):
               t = v 
       
    return visited[t], t, angles[t], rpms_robot[t]


def cost(x1,y1,x2,y2):
    return round(np.sqrt((y2-y1)**2+(x2-x1)**2),1)


def heuristic(x1,y1,x2,y2):
    return round(2*np.sqrt((y2-y1)**2+(x2-x1)**2),1)

def getPath(dest,start):        
    path = []
    while dest != start:
        path.append(parent_sons[dest])
        dest = parent_sons[dest]
    return path

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter('D_1.avi', fourcc, 20.0, (800,600))

start_x = 50
start_y = 550

goal_x = 500
goal_y = 50

start = (start_x,start_y)    
dest = (goal_x,goal_y)

ini_theta = 0


"""A* Algorithm plus initial variables/sets"""

visited = {}
cost_go = {}
angles = {}
parent_sons = {}
rpms_robot = {}
Q = [(start_x,start_y)]
visited[(start_x,start_y)] = 0
cost_go[(start_x,start_y)] = 0
angles[(start_x,start_y)] = ini_theta
rpms_robot[(start_x,start_y)] = (0,0)

X_nodes = []
Y_nodes = []

dt = .5

RPM1 = 200
RPM2 = 100
RPM_Combin = [(0,RPM1),(RPM1,0),(RPM1,RPM1),(0,RPM2),(RPM2,0),(RPM2,RPM2),(RPM1,RPM2),(RPM2,RPM1)]

r_check = 35

while Q:
    
    parent_cost, posxy, theta, rpms_rob = getMin(cost_go,visited, Q, angles, rpms_robot)
    Q.remove(posxy)
    
    dest_final = (posxy[0], posxy[1])
    
    if ((posxy[0]-goal_x)**2+(posxy[1]-goal_y)**2) <= (r_check)**2:
        break
    
    else:
        nodes = []
        speeds_robot = []
        
        for rpm in RPM_Combin:
            
            if diff_constraint(rpm[0],rpm[1],posxy[0],posxy[1],dt,theta) != False:
                movements = diff_constraint(rpm[0], rpm[1], posxy[0], posxy[1], dt, theta)
                nodes.append([(movements[0],movements[1]),movements[3],movements[2]])
                speeds_robot.append(rpm)
                X_nodes.append(movements[0])
                Y_nodes.append(movements[1])
                
        for i in range(0,len(nodes)):
            
            if nodes[i][0] not in visited:
                rpms_robot[nodes[i][0]] = speeds_robot[i]
                
                next_x,next_y = nodes[i][0][0],nodes[i][0][1]
                visited[nodes[i][0]]= nodes[i][1]+ parent_cost
                angles[nodes[i][0]] = nodes[i][2]
                cost_go[nodes[i][0]] = heuristic(next_x,next_y,goal_x,goal_y) + parent_cost + nodes[i][1]
                Q.append(nodes[i][0])
                parent_sons[nodes[i][0]]=posxy
                
            else:
                if visited[nodes[i][0]]>(nodes[i][2]+parent_cost):
                    rpms_robot[nodes[i][0]] = speeds_robot[i]
                    
                    next_x,next_y = nodes[i][0][0],nodes[i][0][1]
                    angles[nodes[i][0]] = nodes[i][2]
                    visited[nodes[i][0]] = nodes[i][1] + parent_cost 
                    cost_go[nodes[i][0]] = nodes[i][1] + heuristic(next_x,next_y,goal_x,goal_y) + parent_cost 
                    parent_sons[nodes[i][0]]=posxy

    
"""Path w/ path graph"""    
path = getPath(dest_final,start)

for i in range(len(path)-1,0,-1):
    cv2.circle(empty, (int(path[i][0]),int(path[i][1])), 2, (0,0,0), -1)
#    empty[int(path[i][1])][int(path[i][0])]=[0,0,0]
    new_image=cv2.flip(empty,0)
    cv2.imshow("PATH",new_image)
    cv2.waitKey(1)
    video.write(new_image)

cap.release()    
video.release()
