import skfuzzy as fuzz

import numpy as np
import matplotlib.pyplot as plt

hottemp_x = np.arange(50,101 , 1)
coldtemp_x = np.arange(0, 36, 1)
watertemp_x  = np.arange(0, 101, 1)

hottemp_low = fuzz.trapmf(hottemp_x, [50, 50, 60, 70])
hottemp_med = fuzz.trapmf(hottemp_x, [60, 70, 80, 90])
hottemp_high = fuzz.trapmf(hottemp_x, [80, 90, 100, 100])
coldtemp_low = fuzz.trapmf(coldtemp_x, [0, 0, 7, 14])
coldtemp_med = fuzz.trapmf(coldtemp_x, [7, 14, 21, 28])
coldtemp_high = fuzz.trapmf(coldtemp_x, [21, 28, 35, 35])
watertemp_low = fuzz.trimf(watertemp_x, [0, 0, 50])
watertemp_med = fuzz.trimf(watertemp_x, [0, 50, 100])
watertemp_high = fuzz.trimf(watertemp_x, [50, 100, 100])

fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(16, 4))

ax0.plot(hottemp_x, hottemp_low, 'b', linewidth=1.5, label='LOW')
ax0.plot(hottemp_x, hottemp_med, 'g', linewidth=1.5, label='MEDIUM')
ax0.plot(hottemp_x, hottemp_high, 'r', linewidth=1.5, label='HIGH')
ax0.set_title('HOT WATER TEMPERATURE')
ax0.legend()

ax1.plot(coldtemp_x, coldtemp_low, 'b', linewidth=1.5, label='LOW')
ax1.plot(coldtemp_x, coldtemp_med, 'g', linewidth=1.5, label='MEDIUM')
ax1.plot(coldtemp_x, coldtemp_high, 'r', linewidth=1.5, label='HIGH')
ax1.set_title('COLD WATER TEMPERATURE')
ax1.legend()

ax2.plot(watertemp_x, watertemp_low, 'b', linewidth=1.5, label='LOW')
ax2.plot(watertemp_x, watertemp_med, 'g', linewidth=1.5, label='MEDIUM')
ax2.plot(watertemp_x, watertemp_high, 'r', linewidth=1.5, label='HIGH')
ax2.set_title('WATER TEMPERATUR')
ax2.legend()

plt.show()


#Set input to fuzzy
hottemp_levlo = fuzz.interp_membership(hottemp_x, hottemp_low, 60)
hottemp_levmd = fuzz.interp_membership(hottemp_x, hottemp_med, 60)
hottemp_levhi = fuzz.interp_membership(hottemp_x, hottemp_high, 60)

coldtemp_levlo = fuzz.interp_membership(coldtemp_x, coldtemp_low, 9.5)
coldtemp_levmd = fuzz.interp_membership(coldtemp_x, coldtemp_med, 9.5)
coldtemp_levhi = fuzz.interp_membership(coldtemp_x, coldtemp_high, 9.5)

#Rule 
#low temp and medium temp
active_rule1 = np.fmax(hottemp_levlo, coldtemp_levlo)
watertemp_activation_low = np.fmin(active_rule1, watertemp_low) 
watertemp_activation_med = np.fmin(coldtemp_levmd, watertemp_med)

active_rule3 = np.fmax(hottemp_levhi, coldtemp_levhi)
watertemp_activation_hi = np.fmin(active_rule3, watertemp_high)
watertemp = np.zeros_like(watertemp_x)

#show rule
fig, ax0 = plt.subplots(figsize=(10, 5))

ax0.fill_between(watertemp_x, watertemp, watertemp_activation_low, facecolor='b', alpha=0.7)
ax0.plot(watertemp_x, watertemp_low, 'b', linewidth=0.7, linestyle='--', )
ax0.fill_between(watertemp_x, watertemp, watertemp_activation_med, facecolor='g', alpha=0.7)
ax0.plot(watertemp_x, watertemp_med, 'g', linewidth=0.7, linestyle='--')
ax0.fill_between(watertemp_x, watertemp, watertemp_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(watertemp_x, watertemp_high, 'r', linewidth=0.7, linestyle='--')
ax0.set_title('Membership Activity')

plt.show()

aggregated = np.fmax(watertemp_activation_low,
                    np.fmax(watertemp_activation_med, 
                            watertemp_activation_hi))

water = fuzz.defuzz(watertemp_x, aggregated, 'centroid')
watertemp_activation = fuzz.interp_membership(watertemp_x, aggregated, water) 
print('water temperature =',water)

fig, ax0 = plt.subplots(figsize=(10, 5))

ax0.plot(watertemp_x, watertemp_low, 'b', linewidth=0.7, linestyle='--', )
ax0.plot(watertemp_x, watertemp_med, 'g', linewidth=0.7, linestyle='--')
ax0.plot(watertemp_x, watertemp_high, 'r', linewidth=0.7, linestyle='--')
ax0.fill_between(watertemp_x, watertemp, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([water, water], [0, watertemp_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result')

plt.show()