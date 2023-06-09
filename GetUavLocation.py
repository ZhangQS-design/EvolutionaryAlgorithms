import os
from Configuration import uavs_location

uav_speed = 4.5

def getLocation():

    path = os.getcwd() + "//data"

    with open(path + "//uav_location.txt", 'w') as f:
        f.writelines("uav_id,timestamp,x,y")
        f.writelines("\n")
        for each_uav in uavs_location:
            each_uav_id = uavs_location[each_uav][0]
            each_uav_x = uavs_location[each_uav][1]
            each_uav_y = uavs_location[each_uav][2]
            s = str(each_uav_id) + ",0," + str(each_uav_x) + "," + str(each_uav_y)
            f.writelines(s)
            f.writelines("\n")
            if each_uav_id <= 110:
                for t in range(1, 51):
                    each_uav_x = each_uav_x
                    each_uav_y = each_uav_y + uav_speed
                    s = str(each_uav_id) + "," + str(t) + "," + str(each_uav_x) + "," + str(each_uav_y)
                    f.writelines(s)
                    f.writelines("\n")
                for t in range(51, 151):
                    each_uav_x = each_uav_x + uav_speed
                    each_uav_y = each_uav_y
                    s = str(each_uav_id) + "," + str(t) + "," + str(each_uav_x) + "," + str(each_uav_y)
                    f.writelines(s)
                    f.writelines("\n")
                for t in range(151, 201):
                    each_uav_x = each_uav_x
                    each_uav_y = each_uav_y - uav_speed
                    s = str(each_uav_id) + "," + str(t) + "," + str(each_uav_x) + "," + str(each_uav_y)
                    f.writelines(s)
                    f.writelines("\n")
                for t in range(201, 301):
                    each_uav_x = each_uav_x - uav_speed
                    each_uav_y = each_uav_y
                    s = str(each_uav_id) + "," + str(t) + "," + str(each_uav_x) + "," + str(each_uav_y)
                    f.writelines(s)
                    f.writelines("\n")
            else:
                for t in range(1, 51):
                    each_uav_x = each_uav_x
                    each_uav_y = each_uav_y + uav_speed
                    s = str(each_uav_id) + "," + str(t) + "," + str(each_uav_x) + "," + str(each_uav_y)
                    f.writelines(s)
                    f.writelines("\n")
                for t in range(51, 151):
                    each_uav_x = each_uav_x + uav_speed
                    each_uav_y = each_uav_y
                    s = str(each_uav_id) + "," + str(t) + "," + str(each_uav_x) + "," + str(each_uav_y)
                    f.writelines(s)
                    f.writelines("\n")
                for t in range(151, 201):
                    each_uav_x = each_uav_x
                    each_uav_y = each_uav_y - uav_speed
                    s = str(each_uav_id) + "," + str(t) + "," + str(each_uav_x) + "," + str(each_uav_y)
                    f.writelines(s)
                    f.writelines("\n")
                for t in range(201, 301):
                    each_uav_x = each_uav_x - uav_speed
                    each_uav_y = each_uav_y
                    s = str(each_uav_id) + "," + str(t) + "," + str(each_uav_x) + "," + str(each_uav_y)
                    f.writelines(s)
                    f.writelines("\n")

if __name__ == '__main__':
    getLocation()

