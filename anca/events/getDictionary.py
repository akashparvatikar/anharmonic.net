def getDictionary(topfile, traj_list, dim, Na):
    
    my_dict = {}
    last = 0
    a = []
    k = 1
    a.append(0)
    for i in range(0, size(traj_list)):
        frames = (getCoordinates(topfile, traj_list[i]))
        a.append(size(frames)/(dim*Na))
        a[k] = a[k] + a[k-1]
        lst = list(range(last, a[k]))
        my_dict[traj_list[i]] = lst   
        last = a[k]
        k = k + 1  
    return my_dict
