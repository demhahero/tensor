import numpy as np
from numpy.random.mtrand import normal
import math
import DDPG_Class

STD_SPEED = 10;
MIN_V_SPEED, MAX_V_SPEED = 80*1000/(60*60), 120*1000/(60*60)
density_jam = 0.25;  # per meter
velocity_free = 38.8889;  # mps
#the distribution of speed
Mean_v=100*1000/(60*60); # m/s
Sigma_V=10*1.6*1000/(60*60);
Roh=2/1000;    # number of vehicles per meter (Density)
density = Roh;
Lamda=Roh*Mean_v;
MEMORY_CAPACITY = 100

T=100;
Dim_Area=570;
Rotor_speed=100;
Blade_dimension=570;
Air_density=1.225;
Drag_coefficient=0.4;
UAV_mass=5;
UAV_surface=0.25;
Highway_distance=250; # in Km
M=3;   #Number_of_Drones
R_c=1000;   # Coverage of the UAV
Vmax = 50; #mph

sigma=1;
N = int(T/sigma);

def get_truncated_normal():
    global STD_SPEED, MIN_V_SPEED, MAX_V_SPEED

    speed = STD_SPEED * normal(0.0, 1.0) + get_expected_velocity()
    while speed < 0 or speed > MAX_V_SPEED or speed < MIN_V_SPEED:
        speed = STD_SPEED * normal(0.0, 1.0) + get_expected_velocity()
    return speed

def get_expected_velocity():
    global density, density_jam, velocity_free
    return velocity_free * (1 - (density) / density_jam)


def rate(K_n,x1,x2):
    P= 0.1;
    h=10**-5
    B=1;
    sigma = 10**-12;
    distance = ((x1-x2)**2 + 10000)**(1/2);
    #return (B/K_n) * math.log((1+((P*h)/(sigma**2 * distance))),2);
    return math.log((1+((P*h)/(sigma**2 * distance))),2);

#Coverage_Cell=R_c/2/(cell_Size_New);




arr= np.random.poisson(Lamda, size=int(N));
print(arr)
K = sum(arr);

speed = [0] * K


#Get each vehicle arrival time
counter1 = 0;
counter2 = 0;
counter3 = 0;
counter4 = 0;

arrival_times = [None] * K;
for val in arr:
    counter2 = counter1 + val
    if(counter1 < counter2):
        for i in range(counter2 - counter1):
            arrival_times[counter3] = counter4
            counter3 = counter3 + 1

    counter4 = counter4 + 1
    counter1 = counter2



#Get each car location with time slots
Matrix = [[0 for x in range(K)] for y in range(N)]
for i in range(K):
    speed[i] = get_truncated_normal();
    passed = False;
    for n in range(N):
        if(arrival_times[i] > n or passed):
            Matrix[n][i] = math.inf;
        elif(Matrix[n-1][i] != math.inf and Matrix[n-1][i]+ speed[i] >= Highway_distance):
            passed = True;
            Matrix[n][i] = math.inf;
        elif(arrival_times[i] == n):
            Matrix[n][i] = 0
        else:
            Matrix[n][i] = Matrix[n-1][i] + speed[i];


f1=open('datafile.txt', 'w')
for n in range(N):
    for i in range(K):
        f1.write(str(Matrix[n][i])+" ")
    f1.write("\n")
f1.close()
exit();
a = 0 ;

C = [[0 for x in range(K)] for y in range(N)]   # Reward


a_dim = 1;
a_bound=[50]
s_dim=100
ddpg = DDPG_Class.DDPG(a_dim, s_dim, a_bound)

# for n in range(N):
#     for i in range(K):
#         r = 0;
#         number_of_vehicles = 0;
#         for i2 in range(K):
#             if(Matrix[n][i2]!=math.inf):
#                 number_of_vehicles = number_of_vehicles + 1;
#
#         if(Matrix[n][i2]!=math.inf):
#             a = ddpg.choose_action(s)
#             if(rate(number_of_vehicles, Matrix[n][i], a) >= 10):
#                 C[n][i] = 1;
#                 r = r + C[n][i]


MAX_EPISODES=400
MAX_EP_STEPS=200
var = 40  # control exploration


for i in range(MAX_EPISODES):
    ep_reward = 0
    current_uav_position = 0;
    s = [0] * s_dim
    print("\n\n\n\n ***********************Episode " + str(i) + "**************** \n\n\n\n")

    for j in range(MAX_EP_STEPS):
        r = 0;
        s = np.asarray(s, dtype=np.float32)
        env = [0] * s_dim;
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -50, 50)  # add randomness to action selection for exploration
        current_uav_position += a[0];
        for n in range(N):

            immediate_cars_counter=0;
            for i in range(K):
                number_of_vehicles = 0;

                #Just to calculate number of immediate vehicles
                for i2 in range(K):
                    if(Matrix[n][i2]!=math.inf):
                        number_of_vehicles = number_of_vehicles + 1;


                if(Matrix[n][i]!=math.inf):
                    #print(a[0])
                    env[i] = 1;
                    if(rate(number_of_vehicles, Matrix[n][i], current_uav_position) >= 12):
                        #env[immediate_cars_counter] = 1;
                        #r = r + env[immediate_cars_counter]

                        r = r + 1
                    else:
                        #env[n] = -1;
                        r = r - 1
                    immediate_cars_counter = immediate_cars_counter + 1;
                    if(immediate_cars_counter == s_dim):
                        break;
        #print(env)
        #print(Matrix[n][i2])
        #print(Matrix)
        #exit()
        s_ = env

            # print(s_)
            # print("\n")
            # print(str(a)+" "+str(current_uav_position))
            # print("\n")
        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
    print(ep_reward);