import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, send_file

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/Current Vs Potential at 0.5ppb concentration')
def generate_current_vs_potential():
    mix_d = np.loadtxt(r'./As_Cr.txt')
    xA = mix_d[:, 0:2]
    pA = xA[:, 0]
    cA = xA[:, 1]
    xB = mix_d[:, 2:4]
    pB= xB[:, 0]
    cB = xB[:, 1]
    xC = mix_d[:, 4:6]
    pC = xC[:, 0]
    cC = xC[:, 1]
    xD = mix_d[:, 6:8]
    pD = xD[:, 0]
    cD = xD[:, 1]
    xE = mix_d[:, 8:10]
    pE = xE[:, 0]
    cE = xE[:, 1]

    # Creating the graph
    plt.figure(figsize=(8, 10))

    plt.subplots_adjust(hspace=1.5)  # Adjust vertical spacing between subplots

    plt.subplot(5, 1, 1)
    plt.plot(pA, cA)
    plt.ylabel('current')
    plt.xlabel('potential')
    plt.title('Current Vs Potential at 0.5ppb concentration')

    plt.subplot(5, 1, 2)
    plt.plot(pB, cB)
    plt.ylabel('current')
    plt.xlabel('potential')
    plt.title('Current Vs Potential at 1ppb concentration')

    plt.subplot(5, 1, 3)
    plt.plot(pC, cC)
    plt.ylabel('current')
    plt.xlabel('potential')
    plt.title('Current Vs Potential at 1.5ppb concentration')

    plt.subplot(5, 1, 4)
    plt.plot(pD, cD)
    plt.ylabel('current')
    plt.xlabel('potential')
    plt.title('Current Vs Potential at 2ppb concentration')

    plt.subplot(5, 1, 5)
    plt.plot(pE, cE)
    plt.ylabel('current')
    plt.xlabel('potential')
    plt.title('Current Vs Potential at 2.5ppb concentration')

    # Save the graph as an image
    image_path_1 = 'Current Vs Potential at 0.5ppb concentration.png'
    image_path_2 = 'Current Vs Potential at 1ppb concentration.png'
    image_path_3 = 'Current Vs Potential at 1.5ppb concentration.png'
    image_path_4 = 'Current Vs Potential at 2ppb concentration.png'
    image_path_5 = 'Current Vs Potential at 2.5ppb concentration.png'

    plt.savefig(image_path_1, format='png')
    plt.close()

    # Return a response
    return send_file(image_path_1, mimetype='image/png')

@app.route('/Window Technique')
def generate_window_technique():
    # Loading data and generating the graph
    AsIII_Arr = np.loadtxt(r'./AsIII_array.txt')
    AsV_Arr = np.loadtxt(r'./arsenicV_array.txt')
    CrIII_Arr = np.loadtxt(r'./CrIII_array_.5.txt')
    CrVI_Arr = np.loadtxt(r'./CrVI_array_.5.txt')

    # Creating the graph
    plt.figure(figsize=(8, 10))

    plt.subplots_adjust(hspace=1.5)  # Adjust vertical spacing between subplots

    # Subplot for AsIII concentration vs. potential
    plt.subplot(4, 1, 1)
    plt.plot(AsIII_Arr[:, 0], AsIII_Arr[:, 1])
    plt.xlabel('concentration')
    plt.ylabel('potential')
    plt.title('Potential range for ARSENIC III concentration')

    # Subplot for AsV concentration vs. potential
    plt.subplot(4, 1, 2)
    plt.plot(AsV_Arr[:, 0], AsV_Arr[:, 1])
    plt.xlabel('concentration')
    plt.ylabel('potential')
    plt.title('Potential range for ARSENIC V concentration')

    # Subplot for CrIII concentration vs. potential
    plt.subplot(4, 1, 3)
    plt.plot(CrIII_Arr[:, 0], CrIII_Arr[:, 1])
    plt.xlabel('concentration')
    plt.ylabel('potential')
    plt.title('Potential range for CHROMIUM III concentration')

    # Subplot for CrVI concentration vs. potential
    plt.subplot(4, 1, 4)
    plt.plot(CrVI_Arr[:, 0], CrVI_Arr[:, 1])
    plt.xlabel('concentration')
    plt.ylabel('potential')
    plt.title('Potential range for CHROMIUM VI concentration')

    # Save the graph as an image
    image_path_1 = 'Potential range for ARSENIC III concentration.png'
    image_path_2 = 'Potential range for ARSENIC V concentration.png'
    image_path_3 = 'Potential range for CHROMIUM III concentration.png'
    image_path_4 = 'Potential range for CHROMIUM VI concentration.png'

    image_path = 'window_technique.png'
    plt.savefig(image_path_1, format='png')
    plt.close()


    # image_path = 'window_technique.png'
    # plt.savefig(image_path, format='png')
    # plt.close()

    # Return a response
    return send_file(image_path, mimetype='image/png')

@app.route('/Concentrations of Heavy Metal Ions')
def generate_graph():
    
    # Loading data and generating the graph
    mix_d = np.loadtxt(r'./As_Cr.txt')
    xA = mix_d[:, 0:2]
    pA = xA[:, 0]
    cA = xA[:, 1]
    xB = mix_d[:, 2:4]
    pB= xB[:, 0]
    cB = xB[:, 1]
    xC = mix_d[:, 4:6]
    pC = xC[:, 0]
    cC = xC[:, 1]
    xD = mix_d[:, 6:8]
    pD = xD[:, 0]
    cD = xD[:, 1]
    xE = mix_d[:, 8:10]
    pE = xE[:, 0]
    cE = xE[:, 1]

# Loading data and generating the graph
    AsIII_Arr = np.loadtxt(r'./AsIII_array.txt')
    AsV_Arr = np.loadtxt(r'./arsenicV_array.txt')
    CrIII_Arr = np.loadtxt(r'./CrIII_array_.5.txt')
    CrVI_Arr = np.loadtxt(r'./CrVI_array_.5.txt')

# Finding indices of interest based on potential range
    ind1_AsIII = np.where((pC >= 0.30) & (pC <= 0.32))[0]
    ind1_AsV = np.where((pC >= 1.30) & (pC <= 1.33))[0]
    ind2_CrIII = np.where((pC >= 0.32) & (pC <= 0.38))[0]
    ind2_CrVI = np.where((pC >= 1.38) & (pC <= 1.41))[0]

# Extracting potential values for ARSENIC and CHROMIUM
    asIII = pC[ind1_AsIII]
    asV = pC[ind1_AsV]
    cr_III = pC[ind2_CrIII]
    cr_VI = pC[ind2_CrVI]

# Extracting current values for ARSENIC and CHROMIUM
    x1_asIII = cC[ind1_AsIII]
    x1_asV = cC[ind1_AsV]
    x1_cr_III = cC[ind2_CrIII]
    x1_cr_VI = cC[ind2_CrVI]

# Finding maximum current and corresponding potential for ARSENIC (asIII)
    max_x1_asIII = np.max(x1_asIII)
    max_x1_asIII_rows = np.argmax(x1_asIII)
    as_at_max_x1_asIII = asIII[max_x1_asIII_rows]

# Finding maximum current and corresponding potential for ARSENIC (asV)
    max_x1_asV = np.max(x1_asV)
    max_x1_asV_rows = np.argmax(x1_asV)
    as_at_max_x1_asV = asV[max_x1_asV_rows]

# Finding maximum current and corresponding potential for CHROMIUM (crIII)
    max_x1_crIII = np.max(x1_cr_III)
    max_x1_crIII_rows = np.argmax(x1_cr_III)
    cr_at_max_x1_crIII = cr_III[max_x1_crIII_rows]

# Finding maximum current and corresponding potential for CHROMIUM (crVI)
    max_x1_crVI = np.max(x1_cr_VI)
    max_x1_crVI_rows = np.argmax(x1_cr_VI)
    cr_at_max_x1_crVI = cr_VI[max_x1_crVI_rows]


    p_known_asIII = AsIII_Arr[:, 1]
    c_known_asIII = AsIII_Arr[:, 0]
    p_known_asV = AsV_Arr[:, 1]
    c_known_asV = AsV_Arr[:, 0]

# Unique Potential
    p_unique_asIII, idx_AsIII = np.unique(p_known_asIII, return_index=True)
    c_unique_asIII = c_known_asIII[idx_AsIII]

# Interpolating
    new_c_values_asIII = np.interp(as_at_max_x1_asIII, p_unique_asIII, c_unique_asIII, left=0, right=0)

    p_known_asV = AsV_Arr[:, 1]
    c_known_asV = AsV_Arr[:, 0]

# Finding unique potential values and corresponding concentration values for ARSENIC (asV)
    p_unique_asV, idx_AsV = np.unique(p_known_asV, return_index=True)
    c_unique_asV = c_known_asV[idx_AsV]
# Interpolate to find new concentration values at max potential for ARSENIC (asV)
    new_c_values_asV = np.interp(as_at_max_x1_asV, p_unique_asV, c_unique_asV, left=0, right=0)

# Defining known potential and concentration values for CHROMIUM (crIII)
    p_known_crIII = CrIII_Arr[:, 1]
    c_known_crIII = CrIII_Arr[:, 0]

# Find unique potential values and corresponding concentration values for CHROMIUM (crIII)
    p_unique_crIII, idx_crIII = np.unique(p_known_crIII, return_index=True)
    c_unique_crIII = c_known_crIII[idx_crIII]

# Interpolate to find new concentration values at max potential for CHROMIUM (crIII
    new_c_values_crIII = np.interp(cr_at_max_x1_crIII, p_unique_crIII, c_unique_crIII, left=0, right=0)

# Process CHROMIUM data (crVI)
# Define known potential and concentration values for CHROMIUM (crVI)
    p_known_crVI = CrVI_Arr[:, 1]
    c_known_crVI = CrVI_Arr[:, 0]

# Find unique potential values and corresponding concentration values for CHROMIUM (crVI)
    p_unique_crVI, idx_crVI = np.unique(p_known_crVI, return_index=True)
    c_unique_crVI = c_known_crVI[idx_crVI]

# Interpolate to find new concentration values at max potential for CHROMIUM (crVI)
    new_c_values_crVI = np.interp(cr_at_max_x1_crVI, p_unique_crVI, c_unique_crVI, left=0, right=0)

# Display mean & standard deviation of potential & concentration for ARSENIC and CHROMIUM
    plt.figure(8, figsize=(12, 10))  # Adjust the figure size as needed
    plt.subplots_adjust(hspace=1.5)  #for vertical space
    
# Subplot for AsV potential vs. concentration
    plt.subplot(4,2,1)
    plt.scatter(as_at_max_x1_asV, new_c_values_asV, c='red', marker='o', label='AsV')
    plt.xlabel('potential')
    plt.ylabel('concentration')
    plt.title('AsV')
    plt.grid(True)
    plt.legend()
    

# Subplot for AsV current vs. concentration
    plt.subplot(4,2,2)
    plt.scatter(max_x1_asV, new_c_values_asV, c='red', marker='o', label='AsV')
    plt.xlabel('CURRENT')
    plt.ylabel('concentration')
    plt.title('AsV')

# Subplot for AsIII potential vs. concentration
    plt.subplot(4,2,3)
    plt.scatter(as_at_max_x1_asIII, new_c_values_asIII, c='red', marker='o', label='AsIII')
    plt.xlabel('potential')
    plt.ylabel('concentration')
    plt.title('AsIII')
    plt.grid(True)
    plt.legend()

# Subplot for AsIII current vs. concentration
    plt.subplot(4,2,4)
    plt.scatter(max_x1_asIII, new_c_values_asIII, c='red', marker='o', label='AsIII')
    plt.xlabel('CURRENT')
    plt.ylabel('concentration')
    plt.title('AsIII')

# Subplot for CrIII potential vs. concentration
    plt.subplot(4,2,5)
    plt.scatter(cr_at_max_x1_crIII, new_c_values_crIII, c='red', marker='o', label='CrIII')
    plt.xlabel('potential')
    plt.ylabel('concentration')
    plt.title('CrIII')
    plt.grid(True)
    plt.legend()

# Subplot for CrIII current vs. concentration
    plt.subplot(4,2,6)
    plt.scatter(max_x1_crIII, new_c_values_crIII, c='red', marker='o', label='CrIII')
    plt.xlabel('CURRENT')
    plt.ylabel('concentration')
    plt.title('CrIII')

# Subplot for CrVI potential vs. concentration
    plt.subplot(4,2,7)
    plt.scatter(cr_at_max_x1_crVI, new_c_values_crVI, c='red', marker='o', label='CrVI')
    plt.xlabel('potential')
    plt.ylabel('concentration')
    plt.title('CrVI')
    plt.grid(True)
    plt.legend()

# Subplot for CrVI current vs. concentration
    plt.subplot(4,2,8)
    plt.scatter(max_x1_crVI, new_c_values_crVI, c='red', marker='o', label='CrVI')
    plt.xlabel('CURRENT')
    plt.ylabel('concentration')
    plt.title('CrVI')

    plt.tight_layout()

    plt.show()

# Save the graph as an image
    image_path = 'Arsenic III.png'
    image_path = 'Arsenic V.png'
    image_path = 'Chromium III.png'
    image_path = 'Chromium VI.png'

    plt.savefig(image_path, format='png')
    plt.close()

    # Return a response
    return send_file(image_path, mimetype='image/png')

@app.route('/Heavy Metal Ion and Concentration')
def generate_bargraph():

    # Loading data and generating the graph
    mix_d = np.loadtxt(r'./As_Cr.txt')
    xA = mix_d[:, 0:2]
    pA = xA[:, 0]
    cA = xA[:, 1]
    xB = mix_d[:, 2:4]
    pB= xB[:, 0]
    cB = xB[:, 1]
    xC = mix_d[:, 4:6]
    pC = xC[:, 0]
    cC = xC[:, 1]
    xD = mix_d[:, 6:8]
    pD = xD[:, 0]
    cD = xD[:, 1]
    xE = mix_d[:, 8:10]
    pE = xE[:, 0]
    cE = xE[:, 1]

# Loading data and generating the graph
    AsIII_Arr = np.loadtxt(r'./AsIII_array.txt')
    AsV_Arr = np.loadtxt(r'./arsenicV_array.txt')
    CrIII_Arr = np.loadtxt(r'./CrIII_array_.5.txt')
    CrVI_Arr = np.loadtxt(r'./CrVI_array_.5.txt')

# Finding indices of interest based on potential range
    ind1_AsIII = np.where((pC >= 0.30) & (pC <= 0.32))[0]
    ind1_AsV = np.where((pC >= 1.30) & (pC <= 1.33))[0]
    ind2_CrIII = np.where((pC >= 0.32) & (pC <= 0.38))[0]
    ind2_CrVI = np.where((pC >= 1.38) & (pC <= 1.41))[0]

# Extracting potential values for ARSENIC and CHROMIUM
    asIII = pC[ind1_AsIII]
    asV = pC[ind1_AsV]
    cr_III = pC[ind2_CrIII]
    cr_VI = pC[ind2_CrVI]

# Extracting current values for ARSENIC and CHROMIUM
    x1_asIII = cC[ind1_AsIII]
    x1_asV = cC[ind1_AsV]
    x1_cr_III = cC[ind2_CrIII]
    x1_cr_VI = cC[ind2_CrVI]

# Finding maximum current and corresponding potential for ARSENIC (asIII)
    max_x1_asIII = np.max(x1_asIII)
    max_x1_asIII_rows = np.argmax(x1_asIII)
    as_at_max_x1_asIII = asIII[max_x1_asIII_rows]

# Finding maximum current and corresponding potential for ARSENIC (asV)
    max_x1_asV = np.max(x1_asV)
    max_x1_asV_rows = np.argmax(x1_asV)
    as_at_max_x1_asV = asV[max_x1_asV_rows]

# Finding maximum current and corresponding potential for CHROMIUM (crIII)
    max_x1_crIII = np.max(x1_cr_III)
    max_x1_crIII_rows = np.argmax(x1_cr_III)
    cr_at_max_x1_crIII = cr_III[max_x1_crIII_rows]

# Finding maximum current and corresponding potential for CHROMIUM (crVI)
    max_x1_crVI = np.max(x1_cr_VI)
    max_x1_crVI_rows = np.argmax(x1_cr_VI)
    cr_at_max_x1_crVI = cr_VI[max_x1_crVI_rows]


    p_known_asIII = AsIII_Arr[:, 1]
    c_known_asIII = AsIII_Arr[:, 0]
    p_known_asV = AsV_Arr[:, 1]
    c_known_asV = AsV_Arr[:, 0]

# Unique Potential
    p_unique_asIII, idx_AsIII = np.unique(p_known_asIII, return_index=True)
    c_unique_asIII = c_known_asIII[idx_AsIII]

# Interpolating
    new_c_values_asIII = np.interp(as_at_max_x1_asIII, p_unique_asIII, c_unique_asIII, left=0, right=0)

    p_known_asV = AsV_Arr[:, 1]
    c_known_asV = AsV_Arr[:, 0]

# Finding unique potential values and corresponding concentration values for ARSENIC (asV)
    p_unique_asV, idx_AsV = np.unique(p_known_asV, return_index=True)
    c_unique_asV = c_known_asV[idx_AsV]
# Interpolate to find new concentration values at max potential for ARSENIC (asV)
    new_c_values_asV = np.interp(as_at_max_x1_asV, p_unique_asV, c_unique_asV, left=0, right=0)

# Defining known potential and concentration values for CHROMIUM (crIII)
    p_known_crIII = CrIII_Arr[:, 1]
    c_known_crIII = CrIII_Arr[:, 0]

# Find unique potential values and corresponding concentration values for CHROMIUM (crIII)
    p_unique_crIII, idx_crIII = np.unique(p_known_crIII, return_index=True)
    c_unique_crIII = c_known_crIII[idx_crIII]

# Interpolate to find new concentration values at max potential for CHROMIUM (crIII
    new_c_values_crIII = np.interp(cr_at_max_x1_crIII, p_unique_crIII, c_unique_crIII, left=0, right=0)

# Process CHROMIUM data (crVI)
# Define known potential and concentration values for CHROMIUM (crVI)
    p_known_crVI = CrVI_Arr[:, 1]
    c_known_crVI = CrVI_Arr[:, 0]

# Find unique potential values and corresponding concentration values for CHROMIUM (crVI)
    p_unique_crVI, idx_crVI = np.unique(p_known_crVI, return_index=True)
    c_unique_crVI = c_known_crVI[idx_crVI]

# Interpolate to find new concentration values at max potential for CHROMIUM (crVI)
    new_c_values_crVI = np.interp(cr_at_max_x1_crVI, p_unique_crVI, c_unique_crVI, left=0, right=0)

    # plt.tight_layout()
    fig = plt.figure(figsize=(8, 6))

# Displaying the Concentrations of Heavy Metal Ions
    print("Heavy Metal Ion and Concentration:")
    ions = []
    concentrations = []

    if new_c_values_asIII:
        ions.append("AsIII")
        concentrations.append(new_c_values_asIII)

    if new_c_values_asV:
        ions.append("AsV")
        concentrations.append(new_c_values_asV)

    if new_c_values_crIII:
        ions.append("CrIII")
        concentrations.append(new_c_values_crIII)

    if new_c_values_crVI:
        ions.append("CrVI")
        concentrations.append(new_c_values_crVI)

    # Generate colors dynamically based on the number of ions
    colors = plt.cm.viridis(np.linspace(0, 1, len(ions)))

    # Plotting the concentrations
    bars = plt.bar(ions, concentrations, color=colors)

    # Add the values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

    plt.xlabel('Heavy Metal Ions')
    plt.ylabel('Concentration')
    plt.title('Heavy Metal Ion and Concentration')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(axis='y')  # Show grid lines for y-axis

    # Add a legend
    plt.legend(bars, ions)

    plt.tight_layout()

    # Save the graph as an image
    image_path_1 = 'Heavy Metal Ion and Concentration.png'
    
    plt.savefig(image_path_1, format='png')
    plt.show()
    plt.close()


    # Return the graph image as a response
    return send_file(image_path_1, mimetype='image/png')

app.run(debug=True, use_reloader=False)

# if __name__ == '__main__':
#         app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
# Use 0.0.0.0 as the host to allow external access
    app.run(debug=False, host='0.0.0.0')


# , port=8080, debug=True, use_reloader=False
