"""General helper functions for notebooks."""

def get_all_EE(energies):
    ha = 27.211386245981
    all_ee = []
    for i in range(len(energies)-1):
        ee = (energies[i+1] - energies[0])*ha
        all_ee.append(round(ee, 5))
    return (all_ee)
