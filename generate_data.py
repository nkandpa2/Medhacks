import numpy as np
import matplotlib.pyplot as plt

TIME_STEPS = 1000
N_FEATURES = 2 
N_POP = 1500
corners = [(120,80,'g-'),(129,84,'y-'),(139,89,'m-'),(159,99,'r-')]

def get_bin(point, corners):
    level = 0
    for (sys_standard, dia_standard, _) in corners:
        (sys, dia) = point
        if sys < sys_standard and dia < dia_standard:
            return level
        level += 1

    return 4

def generate_data(start_means, end_means, start_std, end_std, pop_size, pop_type):
    if pop_type == 'untreated': 
        return generate_pop(start_means, end_means, start_std, end_std, pop_size)
    
    elif pop_type == 'normal':
        return generate_pop(start_means, start_means, start_std, start_std, pop_size)

    elif pop_type == 'treated':
        data = np.zeros((TIME_STEPS, pop_size, N_FEATURES))
        
        treatment_times = np.random.randint(1, TIME_STEPS, (pop_size))
        treatment_times.fill(500)

        for feat in range(N_FEATURES):
            s_mean = start_means[feat]
            e_mean = end_means[feat]

            s_std = start_std[feat]
            e_std = end_std[feat]

            starting_points = np.random.normal(s_mean, s_std, pop_size)
            ending_points = np.random.normal(e_mean, e_std, pop_size)

            index = 0
            for start,end,time in zip(starting_points, ending_points, treatment_times):
                noise = np.random.normal(0, 5.0/3, TIME_STEPS)
                trend_up = np.linspace(start, end, TIME_STEPS)[0:time]
                trend_down = np.linspace(trend_up[len(trend_up)-1], start, TIME_STEPS/2)[0:time]
                data[:,index,feat] = np.concatenate((trend_up, trend_down),axis=0) + noise
                index += 1
      
        return data
        

def generate_pop(start_means, end_means, start_std, end_std, pop_size):
    data = np.zeros((TIME_STEPS, pop_size, N_FEATURES))
    
    end_points = np.zeros((N_FEATURES, pop_size))
    start_points = np.zeros((N_FEATURES, pop_size))

    for feat in range(N_FEATURES):
        s_mean = start_means[feat]
        e_mean = end_means[feat]

        s_std = start_std[feat]
        e_std = end_std[feat]

        starting_points = np.random.normal(s_mean, s_std, pop_size)
        ending_points = np.random.normal(e_mean, e_std, pop_size)

        end_points[feat] = ending_points
        start_points[feat] = starting_points
        
        index = 0
        for start,end in zip(starting_points, ending_points):
            noise = np.random.normal(0, 5.0/3, TIME_STEPS)
            data[:,index,feat] = np.linspace(start, end, TIME_STEPS) + noise
            index += 1
      
    plt.plot(end_points[0], end_points[1], 'ro')
    plt.plot(start_points[0], start_points[1], 'bo')

    for c in corners:
        (sys, di, color) = c
        plt.plot([0,sys,sys],[di,di,0],color,linewidth=4.0)
    
    plt.axis([100,200,50,100])
    plt.show()
    return data

start_means = [122,67]
end_means = [152,74]
start_std = [8,6]
end_std = [12,9]

data_normal = generate_data(start_means, end_means, start_std, end_std, N_POP, 'normal')
"""
for d in range(N_POP):
    p1_feat1 = data_normal[:,d,1]
    plt.plot(p1_feat1)
plt.show()
"""
data_untreated = generate_data(start_means, end_means, start_std, end_std, N_POP, 'untreated')
"""
for d in range(N_POP):
    p1_feat1 = data_untreated[:,d,1]
    plt.plot(p1_feat1)
plt.show()
"""
data_treated = generate_data(start_means, end_means, start_std, end_std, N_POP, 'treated')

"""
for d in range(N_POP):
    p1_feat1 = data_treated[:,d,1]
    plt.plot(p1_feat1)
    plt.show()
"""

data_treated_mean = np.mean(data_treated, axis=1)
data_untreated_mean = np.mean(data_untreated, axis=1)
data_normal_mean = np.mean(data_normal, axis=1)

data_treated_std = np.std(data_treated, axis=1)
data_untreated_std = np.std(data_untreated, axis=1)
data_normal_std = np.std(data_normal, axis=1)

plt.plot(data_treated_mean)
plt.plot(data_treated_mean + data_treated_std)
plt.plot(data_treated_mean - data_treated_std)
plt.show()

plt.plot(data_untreated_mean)
plt.plot(data_untreated_mean + data_untreated_std)
plt.plot(data_untreated_mean - data_untreated_std)
plt.show()

plt.plot(data_normal_mean)
plt.plot(data_normal_mean + data_normal_std)
plt.plot(data_normal_mean - data_normal_std)
plt.show()

risk = []
for point in data_treated_mean:
    risk.append(get_bin(point, corners))

plt.plot(risk)

risk = []
for point in data_untreated_mean:
    risk.append(get_bin(point, corners))

plt.plot(risk)

risk = []
for point in data_normal_mean:
    risk.append(get_bin(point, corners))

plt.plot(risk)

plt.show()

#np.save('data.npy', data)
