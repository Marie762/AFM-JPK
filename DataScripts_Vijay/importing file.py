import os
import afmformats
import matplotlib.pyplot as plt

def force():
    allfilesinfolder = os.listdir(r'Data') 
    must_end_in = '.jpk-force'
    jpk_force_files = [os.path.join('Data', file) for file in allfilesinfolder if file.endswith(must_end_in)]

    # create empty list to store all the data extracted from each jpk-force file
    jpk_force_data_list = []
    
    # for loop to extract and append all the separate jpk-force data to the list jpk_force_data_list (length equal to the number of files in folder 'Data')
    for file in jpk_force_files:
        data_extract = afmformats.load_data(file)
        jpk_force_data_list.append(data_extract)

    # scale conversion constants
    ysc = 1e9  # nN
    dsc = 1e6  # microns

    # create three empty lists to store the height (d), force (F), and time (t) values of each jpk-force file  
    d = []
    F = []
    t = []

    # add all the height, force, and time data to separate lists, with the element corresponding to the jpk_force_data_list
    for j in range(len(jpk_force_files)):
        # create three empty lists to locally store the [approach, intermediate, retract] data
        d_local = []
        F_local = []
        t_local = []
        
        d_local.append(jpk_force_data_list[j][0].appr["height (measured)"] * dsc)
        F_local.append(jpk_force_data_list[j][0].appr["force"] * ysc)
        t_local.append(jpk_force_data_list[j][0].appr["time"])
        
        if jpk_force_data_list[j][0].modality == 'creep compliance':
            d_local.append(jpk_force_data_list[j][0].intr["height (measured)"] * dsc)
            F_local.append(jpk_force_data_list[j][0].intr["force"] * ysc)
            t_local.append(jpk_force_data_list[j][0].intr["time"])
            
        d_local.append(jpk_force_data_list[j][0].retr["height (measured)"] * dsc)
        F_local.append(jpk_force_data_list[j][0].retr["force"] * ysc)
        t_local.append(jpk_force_data_list[j][0].retr["time"])
    
        d.append(d_local)
        F.append(F_local)
        t.append(t_local)
    
    return d, F, t

# Call the force function and get the data
d, F, t = force()

# Plot all graphs
for i in range(len(d)):
    plt.figure(figsize=(10, 6))
    for j in range(len(d[i])):
        plt.plot(d[i][j], F[i][j], label=f'Segment {j + 1}')
    plt.xlabel('Height (microns)')
    plt.ylabel('Force (nN)')
    plt.title(f'Force vs. Height for File {i + 1}')
    plt.legend()
    plt.show()
