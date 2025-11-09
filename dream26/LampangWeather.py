import matplotlib.pyplot as plt
import pandas as pd

# --- Data (approximate from the chart) ---
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
temperature = [29.0, 30.5, 32.5, 35.5, 35.0, 33.5, 32.5, 32.0, 31.5, 31.0, 29.5, 28.5]  # °C
rainfall = [10, 20, 40, 60, 130, 150, 180, 240, 200, 120, 40, 10]  # mm

df = pd.DataFrame({'Month': months, 'Temperature': temperature, 'Rainfall': rainfall})

# --- Plot setup ---
fig, ax1 = plt.subplots(figsize=(8,5))

# Primary axis (Temperature)
ax1.plot(df['Month'], df['Temperature'], color='orange', marker='o', linewidth=2, label='Temperature')
ax1.set_ylabel('Temperature (°C)', color='orange')
ax1.set_ylim(28, 37)
ax1.tick_params(axis='y', labelcolor='orange')

# Secondary axis (Rainfall)
ax2 = ax1.twinx()
ax2.bar(df['Month'], df['Rainfall'], color='royalblue', alpha=0.7, label='Rainfall')
ax2.set_ylabel('Rainfall (mm)', color='royalblue')
ax2.set_ylim(0, 260)
ax2.tick_params(axis='y', labelcolor='royalblue')

# Title and styling
plt.title('Lampang Thailand Average Monthly Rainfall and Temperature\nAverage Precipitation & Temperatures (approx. 1993–2018)', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
