import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lnm

TIME_STEPS = 1000
N_FEATURES = 2 
N_POP = 1500
corners = [(120,80,'g-'),(129,84,'y-'),(139,89,'m-'),(159,99,'r-')]
corners_black = [(150, 85, 'g-'),(160,93,'y-'),(170,99,'m-'),(175,109,'r-')]

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

    plt.axis([100,200,50,100])
    plt.xlabel('Systolic', fontsize=16)
    plt.ylabel('Diastolic', fontsize=16)
    plt.subplots_adjust(bottom=0.25, left=0.25)
     
    plt.plot(end_points[0], end_points[1], 'ro')
    plt.plot(start_points[0], start_points[1], 'bo')
    

    

    for c in corners:
        (sys, di, color) = c
        plt.plot([0,sys,sys],[di,di,0],color,linewidth=4.0)
    
    plt.show()
    return data

start_means = [122,67]
end_means = [152,74]
start_std = [8,6]
end_std = [12,9]

start_sys_good = np.random.normal(start_means[0], start_std[0], N_POP)
start_dia_good = np.random.normal(start_means[1], start_std[1], N_POP)

start_sys_bad = np.random.normal(end_means[0], end_std[0], N_POP)
start_dia_bad = np.random.normal(end_means[1], end_std[1], N_POP)

plt.plot(start_sys_good, start_dia_good, 'bo')
plt.axis([100,200,50,115])
plt.xlabel('Systolic', fontsize=16)
plt.ylabel('Diastolic', fontsize=16)

for c in corners:
    (sys, di, color) = c
    plt.plot([0,sys,sys],[di,di,0],color,linewidth=4.0)
plt.show()

plt.axis([100,200,50,115])
plt.xlabel('Systolic', fontsize=16)
plt.ylabel('Diastolic', fontsize=16)
plt.plot(start_sys_bad, start_dia_bad, 'bo')
for c in corners_black:
    (sys, di, color) = c
    plt.plot([0,sys,sys],[di,di,0],color,linewidth=4.0)
plt.show()




data_normal = generate_data(start_means, end_means, start_std, end_std, N_POP, 'normal')
np.savetxt('data_treated_start.csv', data_normal[0,:,:], delimiter=',', header='systolic,diastolic', comments='')
np.savetxt('data_treated_end.csv', data_normal[-1,:,:], delimiter=',', header='systolic,diastolic', comments='')


data_untreated = generate_data(start_means, end_means, start_std, end_std, N_POP, 'untreated')
np.savetxt('data_untreated_start.csv', data_untreated[0,:,:], delimiter=',', header='systolic,diastolic', comments='')
np.savetxt('data_untreated_end.csv', data_untreated[-1,:,:], delimiter=',', header='systolic,diastolic', comments='')

data_treated = generate_data(start_means, end_means, start_std, end_std, N_POP, 'treated')

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

for d in range(N_POP):
    individual = data_untreated[:,d,:]
    sys = individual[:,0]
    dia = individual[:,1]
    h = lnm.HuberRegressor()
    point = [0,0]
    for i in range(2):
        prev_x = np.array(list(range(TIME_STEPS - 200, TIME_STEPS)))
        prev_y = np.array(individual[TIME_STEPS - 200:TIME_STEPS, i])
        h.fit(prev_x.reshape(-1,1), prev_y)
        next_x = np.array(list(range(TIME_STEPS, TIME_STEPS + 100)))
        next_y = h.predict(next_x.reshape(-1,1))
        plt.plot(individual[:,i], 'b-')
        plt.plot(next_x, next_y, 'r-')
        point[i] = next_y[-1]
        if i == 0:
            prev_file = 'previous_systolic.csv'
            next_file = 'next_systolic.csv'
        else:
            prev_file = 'previous_diastolic.csv'
            next_file = 'next_diastolic.csv'

        x = np.array(list(range(0, TIME_STEPS)))
        x = np.expand_dims(x, axis=1)
        ind = np.expand_dims(individual[:,i], axis=1)
        data = np.hstack((x, ind))
        print(np.shape(data))
        np.savetxt(prev_file, data, delimiter=',', header='time,y', comments='')
        next_x = np.expand_dims(next_x, axis=1)
        next_y = np.expand_dims(next_y, axis=1)
        data = np.hstack((next_x, next_y))
        print(np.shape(data))
        np.savetxt(next_file, data, delimiter=',', header='time,y', comments='')
        
    p = (point[0], point[1])
    print('Bin: {0}'.format(get_bin(p, corners)))
    plt.xlabel('Time')
    plt.ylabel('Blood Pressure')
    plt.show()

adhere = np.linspace(0,1,20)
slope1 = -(0.03/0.75)*adhere+0.03 + np.random.normal(0,0.001,20)
plt.plot(adhere,slope1)

plt.plot([0,1],[0,0], 'k-')

adhere = np.linspace(0,1,20)
slope2 = -(0.03/1.3)*adhere+0.03 + np.random.normal(0,0.001,20)
plt.plot(adhere,slope2)
plt.axis([0,1,-0.05,0.05])
plt.xlabel('Adherence Probability')
plt.ylabel('Blood Pressure Slope')
plt.show()

#np.save('data.npy', data)
