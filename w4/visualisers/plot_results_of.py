import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

    
with open('plots/results.pkl', 'rb') as f:
    results = pkl.load(f) #const_type distance block_size search_radius elapsed_time msen pepn
    
with open('plots/results2.pkl', 'rb') as f:
    results2 = pkl.load(f) #const_type distance block_size search_radius elapsed_time msen pepn
    
results = results+results2
    
filt = [x for x in results if x[0:2] == ['forward', 'MSE'] and x[2] != 128]

block_sizes = [8,16,21,32,41,48,64,81]
search_radii = [8,16,21,32,41,48]
mat_msen = np.zeros((len(block_sizes),len(search_radii)))

for i in range(len(block_sizes)):
    for j in range(len(search_radii)):
        mat_msen[i,j] = [x[5] for x in filt if x[2:4] == [block_sizes[i], search_radii[j]]][0]
        # print([x[5] for x in filt if x[2:4] == [block_sizes[i], search_radii[j]]][0])
        
plt.figure(1)
plt.imshow(mat_msen, cmap='RdYlGn_r')
plt.yticks(range(len(block_sizes)),labels=block_sizes)
plt.xticks(range(len(search_radii)),labels=search_radii)
plt.ylabel('Block size')
plt.xlabel('Search radius')
plt.title('MSEN')
plt.colorbar()
plt.clim(1.5,4.5)
plt.savefig('Results MSEN')

mat_time = np.zeros((len(block_sizes),len(search_radii)))

for i in range(len(block_sizes)):
    for j in range(len(search_radii)):
        mat_time[i,j] = [x[4] for x in filt if x[2:4] == [block_sizes[i], search_radii[j]]][0]
        # print([x[5] for x in filt if x[2:4] == [block_sizes[i], search_radii[j]]][0])
        
plt.figure(2)
plt.imshow(mat_time, cmap='RdYlGn_r')
plt.yticks(range(len(block_sizes)),labels=block_sizes)
plt.xticks(range(len(search_radii)),labels=search_radii)
plt.ylabel('Block size')
plt.xlabel('Search radius')
plt.title('Time')
plt.colorbar()
plt.clim(0,800)
plt.savefig('Results time')

mat_pepn = np.zeros((len(block_sizes),len(search_radii)))

for i in range(len(block_sizes)):
    for j in range(len(search_radii)):
        mat_pepn[i,j] = [x[6] for x in filt if x[2:4] == [block_sizes[i], search_radii[j]]][0]
        # print([x[5] for x in filt if x[2:4] == [block_sizes[i], search_radii[j]]][0])
        
plt.figure(3)
plt.imshow(mat_pepn, cmap='RdYlGn_r')
plt.yticks(range(len(block_sizes)),labels=block_sizes)
plt.xticks(range(len(search_radii)),labels=search_radii)
plt.ylabel('Block size')
plt.xlabel('Search radius')
plt.title('PEPN')
plt.colorbar()
plt.savefig('Results PEPN')
# plt.clim(0,800)

plt.figure(4)
plt.plot(mat_pepn[:,0])
plt.xticks(range(len(block_sizes)),labels=block_sizes)
plt.xlabel('Block size')
plt.ylabel('PEPN')
plt.savefig('Results PEPN plot')
