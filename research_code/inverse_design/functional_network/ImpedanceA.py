import numpy as np
import math
p0 = 1.21*343
def get_Z(a):
    """
    通过吸声系数得到阻抗
    """
    temp1 = 1-math.sqrt(1-a)
    temp2 = 1+math.sqrt(1-a)
    temp = temp1/temp2
    return temp*p0

def get_parallel_z(z1,z2):
    """
    求并联阻抗
    """
    t = 1/z1 + 1/z2
    return 1/t

def get_a(z):
    """从阻抗得到吸声系数"""
    return 1-((p0-z)/(p0+z))**2

if __name__=="__main__":
    z1 = get_Z(0.9)

    z2 = get_Z(0.9)
    print(z1,z2,p0*3/17)
    z = get_parallel_z(z1,z2)
    print(z)
    print(get_a(z))
