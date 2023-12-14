How this code is implemented
-----------------------------

1. First disperison relationships are extracted from literature and each branch is saved as CSV files.

2. Following files read and curve fit for the branches for different nanotubes.
    1. disperison_CNT_10_10.py
    2. disperison_CNT_10_0.py
    3. disperison_BNNT_10_10.py

3. Then we fit for A_1 and s for CNT (10,10) and BNNT (10,10)
    > For this step we use the following scripts.
        1. thermal_transport_A_s_fits.py
        2. thermal_fit_BNNT.py

    > We import the following files from the previous step to get the dispersion relations.
        1. disperison_CNT_10_10.py
        2. disperison_BNNT_10_10.py

    > In this step, experimental data in the following csv files are used to generate the fits.
        1. highTk.csv --> CNTs
        2. BNNT_kT.csv --> BNNTs

    > Then we obtain A_1 and s for CNTs and BNNTs from the two scripts.

4. Simultaneously we can obtain mean free path variations from the following scripts. Same experimental data are used.
    1. lambda_BNNT.py
    2. lambda_CNT.py

5. Then one can back calculate thermal conductivity of CNT (10,10) and BNNT (10,10) from the following scripts.
    1. thermal_prop_CNT_10_10.py
    2. thermal_prop_BNNT_10_10.py

6. Then one can predict the thermal conductivity for CNT (10,0) from the following script.
    1. thermal_prop_CNT_10_0.py

7. Post processing is done in the script "Combine_Plots.py"
