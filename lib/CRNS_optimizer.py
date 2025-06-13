"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.


Authors:
Roland Baatz roland.baatz @ zalf.de

Maintainers and contact:
Currently maintained by the authors.

This file requires output of the CRNS_*.ipynb

Copyright (C) Leibniz Centre for Agricultural Landscape Research (ZALF)
"""


import time
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Moving Block Bootstrap function
def moving_block_bootstrap(data, block_length):
    #Calculate lengths of the blocks for sampling.
    n = len(data)
    num_blocks = n // block_length
    #Create blocks of equal length.
    blocks = [data[i:i + block_length] for i in range(n - block_length + 1)]
    
    #From these, Sample random blocks and extend...
    bootstrap_sample = []
    for _ in range(num_blocks):
        block = blocks[np.random.randint(len(blocks))]
        bootstrap_sample.extend(block)
    
    bootstrap_sample = np.array(bootstrap_sample[:n])  # Ensure same length as original data
    #print(bootstrap_sample.shape)
    
    return bootstrap_sample

# Bootstrapping function
def bootstrap_fit(initial_params, mod_values, pressure_values, incoming_values, abs_h_values, param_bounds, block_length, n_iterations=100):
    bootstrap_params = []
    n_samples = len(mod_values)
    
    for _ in range(n_iterations):
        # Resample data using MBB
        mod_sample = moving_block_bootstrap(mod_values, block_length)
        pressure_sample = moving_block_bootstrap(pressure_values, block_length)
        incoming_sample = moving_block_bootstrap(incoming_values, block_length)
        abs_h_sample = moving_block_bootstrap(abs_h_values, block_length)
        
        # Fit the model to the resampled data
        res = minimize(objective, initial_params, args=(mod_sample, pressure_sample, incoming_sample, abs_h_sample), bounds=param_bounds)
        
        if res.success:
            bootstrap_params.append(res.x)
    
    bootstrap_params = np.array(bootstrap_params)
    uncertainties = np.std(bootstrap_params, axis=0)
    return uncertainties

def bootstrap_fit_rmse_u(initial_params, data_args, param_bounds, block_length, n_iterations=100):
    bootstrap_params = []
    # Concatenate data_args columns into a single array
    combined_data = np.column_stack(data_args)
    
    #print(combined_data.shape)
    block_length_fraction = block_length
    #print(param_bounds)
    #print(block_length)
    for _ in range(n_iterations):
        # Resample data using MBB
        bootstrap_sample = moving_block_bootstrap_subsample(combined_data, block_length_fraction)
        pressure_valid_boot, pressure_prev_valid_boot, abs_valid_boot, abs_prev_valid_boot, i_valid_boot, i_prev_valid_boot, mod_prev_valid_boot, mod_valid_boot = np.hsplit(bootstrap_sample, bootstrap_sample.shape[1])

        
        # Fit the model to the resampled data
        #res = minimize(objective, initial_params, args=(mod_sample, pressure_sample, incoming_sample, abs_h_sample), bounds=param_bounds)
        res = opt.minimize(objective_consecutive_rmse, initial_params, args=(pressure_valid_boot.flatten(),
            pressure_prev_valid_boot.flatten(), abs_valid_boot.flatten(), abs_prev_valid_boot.flatten(), 
            i_valid_boot.flatten(), i_prev_valid_boot.flatten(), mod_prev_valid_boot.flatten(), mod_valid_boot.flatten()),
            bounds=param_bounds
            )
        
        if res.success:
            bootstrap_params.append(res.x)
    
    bootstrap_params = np.array(bootstrap_params)
    uncertainties = np.std(bootstrap_params, axis=0)
    return uncertainties
# Function to perform Nelder-Mead optimization
def optimize_with_nelder_mead(initial_params, data_args):
    result = opt.minimize(
        objective_consecutive_rmse, 
        initial_params, 
        args=data_args, 
        method='Nelder-Mead', 
        options={'disp': False, 'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100000}
    )
    return result.x  # Return optimized parameters

# Moving Block Bootstrap function for a subsample
def moving_block_bootstrap_subsample(data, block_length_fraction):
    n = len(data)  # Length of the original data
    #print("SHAPE:")
    #print(data.shape)
    #print("Length=",n)
    block_length = block_length_fraction  # Calculate block length as a fraction of the original length
    #print(block_length_fraction)
    if block_length == 0:
        raise ValueError("Block length is too small compared to the data length.")
    #print(n)
    #print("New length=",    block_length)
    # Choose a random starting index such that the block fits in the data
    start_index = np.random.randint(0, n - block_length + 1)
    
    # Return the consecutive block of data starting from start_index
    data_sample = data[start_index:start_index + block_length]
    
    #print(data_sample.shape)
    return np.array(data_sample)
    

# Function to estimate uncertainties via bootstrap
def bootstrap_uncertainty(num_bootstrap, block_length_fraction, initial_params, data_args):
    all_bootstrap_params = []
    
    # Concatenate data_args columns into a single array
    combined_data = np.column_stack(data_args)

    for _ in range(num_bootstrap):
        # Apply the moving block bootstrap subsample to the entire dataset
        bootstrap_sample = moving_block_bootstrap_subsample(combined_data, block_length_fraction)
        
        # Split the bootstrap sample back into individual variables
        pressure_valid_boot, pressure_prev_valid_boot, abs_valid_boot, abs_prev_valid_boot, i_valid_boot, i_prev_valid_boot, mod_prev_valid_boot, mod_valid_boot = np.hsplit(bootstrap_sample, bootstrap_sample.shape[1])

        # Optimize on the bootstrap sample
        bootstrap_params = optimize_with_nelder_mead(
            initial_params, 
            (pressure_valid_boot.flatten(), pressure_prev_valid_boot.flatten(), abs_valid_boot.flatten(), abs_prev_valid_boot.flatten(), 
             i_valid_boot.flatten(), i_prev_valid_boot.flatten(), mod_prev_valid_boot.flatten(), mod_valid_boot.flatten())
        )

        # Store the optimized parameters
        all_bootstrap_params.append(bootstrap_params)

    # Convert list to a numpy array for easier statistical analysis
    all_bootstrap_params = np.array(all_bootstrap_params)
    
    # Calculate uncertainties (standard deviations) across bootstrap samples
    param_uncertainties = np.std(all_bootstrap_params, axis=0)
    
    return param_uncertainties


# Calculate uncertainties using the Jacobian
def jacobian(params, mod, pressure, incoming, abs_h):
    beta, omega, psi = params
    n = len(mod)
    J = np.zeros((n - 1, 3))

    for i in range(1, n):
        P_diff = pressure[i] - pressure[i - 1]
        H_diff = abs_h[i] - abs_h[i - 1]
        I_diff = incoming[i] - incoming[i - 1]

        cor_p = corr_exp_pressure(pressure[i], pressure[i - 1], beta)
        cor_h = corr_lin_humidity(abs_h[i], abs_h[i - 1], omega)
        cor_i = corr_lin_incoming_correction(incoming[i], incoming[i - 1], psi)

        J[i - 1, 0] = -2 * (mod[i] - mod[i - 1] * cor_p * cor_h * cor_i) * mod[i - 1] * cor_p * P_diff * np.exp(beta * P_diff)
        J[i - 1, 1] = -2 * (mod[i] - mod[i - 1] * cor_p * cor_h * cor_i) * mod[i - 1] * cor_p * cor_i * H_diff
        J[i - 1, 2] = -2 * (mod[i] - mod[i - 1] * cor_p * cor_h * cor_i) * mod[i - 1] * cor_p * cor_h * I_diff

    return J

def corr_lin_humidity(abs_h, abs_humid_ref, omega):
    """ 
    Calculate correction factor for humidity according to  Rosolem et al., 2013
    Args:
        abs_h (float, range: 0, 35): absolute humdity value
        abs_humidity_ref (float): absolute humidity reference value
        my_parameter (float): fitting parameter for linear scaling
    Returns:
        cor_h (float, range: 0.8, 1.2): humidity correction factor
    """
    cor_h = 1 + omega * (abs_h - abs_humid_ref)
    return cor_h

def corr_exp_humidity(abs_h, abs_humid_ref, omega):
    cor_h =  np.exp(omega * (abs_h - abs_humid_ref))
    return cor_h

def corr_lin_incoming_correction(Inc, Inc_ref, psi):
    """ 
    Calculate correction factor for incoming cosmic ray intensity according to Desilets & Zreda, 2003
    Args:
        Inc (float, range: 800, 1200, hPa): air pressure
        Inc_ref (float: 800, 1200, hPa): reference air pressure
        my_parameter (float):  fitting parameter for linear scaling
        
    Returns:
        cor_i (float, range: 0.8, 1.2): humidity correction factor
    """
    #cor_i = 1 + psi * (Inc - Inc_ref)
    cor_i = 1 + psi * (Inc / Inc_ref - 1)
    return cor_i

def corr_exp_incoming_correction(Inc, Inc_ref, psi):
    cor_i = np.exp(psi * (Inc - Inc_ref))
    return cor_i

def corr_exp_pressure(P, p_ref, beta=0.0076):
    """ 
    Calculate correction factor for humidity according to Desilets & Zreda, 2003
    Args:
        P (float, range: 800, 1200, hPa): air pressure
        p_ref (float: 800, 1200, hPa): reference air pressure
        beta (float):  fitting parameter for exponential scaling
        
    Returns:
        cor_p (float, range: 0.8, 1.2): humidity correction factor
    """
    cor_p = np.exp(beta * (P - p_ref))
    return cor_p

def Calculate_swc_from_Npih(Npih,No,bd):
    SWC = ((0.0808/((Npih/No)-0.372)-0.115)*bd)
    #SWC = (((bd*0.0808)/((Npih/No)'-0.372)) - 0.115*bd)'
    return SWC

def Absolute_conv(Rh,T):
    converted_absolute_humidity = (6.112 * np.exp((17.67*T)/(T+243.5)) * Rh * 2.1674)/(273.15 + T)
    return converted_absolute_humidity



# Objective function to minimize
def objective(params, mod, pressure, incoming, abs_h):
    beta, omega, psi = params
    cor_p = corr_exp_pressure(pressure[1:], pressure[:-1], beta)
    cor_h = corr_lin_humidity(abs_h[1:], abs_h[:-1], omega)
    cor_i = corr_lin_incoming_correction(incoming[1:], incoming[:-1], psi)
    mod_estimated = mod[:-1] * cor_p * cor_h * cor_i
    return np.sum((mod[1:] - mod_estimated)**2)


def objective_consecutive(params, pressure_valid,pressure_prev_valid,abs_valid,abs_prev_valid,i_valid,i_prev_valid,mod_prev_valid,mod_valid):
    #print(daily_df.columns)
    beta, omega, psi = params

    cor_p = corr_exp_pressure(pressure_valid, pressure_prev_valid, beta)
    cor_h = corr_lin_humidity(abs_valid, abs_prev_valid, omega)
    cor_i = corr_lin_incoming_correction(i_valid,i_prev_valid, psi)
    
    # Identify valid indices where the difference is exactly 1 day
    mod_estimated = mod_prev_valid * cor_p * cor_h * cor_i
    
    return np.sum((mod_valid - mod_estimated)**2)

def objective_consecutive_mae(params, data):
    beta, omega, psi = params
    pressure_valid, pressure_prev_valid, abs_valid, abs_prev_valid, i_valid, i_prev_valid, mod_prev_valid, mod_valid = data

    cor_p = corr_exp_pressure(pressure_valid, pressure_prev_valid, beta)
    cor_h = corr_lin_humidity(abs_valid, abs_prev_valid, omega)
    cor_i = corr_lin_incoming_correction(i_valid,i_prev_valid, psi)
    
    # Identify valid indices where the difference is exactly 1 day
    mod_estimated = mod_prev_valid * cor_p * cor_h * cor_i
    return np.mean(abs(mod_valid - mod_estimated))

def objective_consecutive_rmse(params, pressure_valid,pressure_prev_valid,abs_valid,abs_prev_valid,i_valid,i_prev_valid,mod_prev_valid,mod_valid):
    #print(daily_df.columns)
    beta, omega, psi = params

    cor_p = corr_exp_pressure(pressure_valid, pressure_prev_valid, beta)
    cor_h = corr_lin_humidity(abs_valid, abs_prev_valid, omega)
    cor_i = corr_lin_incoming_correction(i_valid,i_prev_valid, psi)
    
    # Identify valid indices where the difference is exactly 1 day
    mod_estimated = mod_prev_valid * cor_p * cor_h * cor_i
    return np.sqrt(np.sum((mod_valid - mod_estimated)**2))

# Objective function to minimize
def objective_rmse(params, mod, pressure, incoming, abs_h):
    beta, omega, psi = params
    cor_p = corr_exp_pressure(pressure[1:], pressure[:-1], beta)
    cor_h = corr_lin_humidity(abs_h[1:], abs_h[:-1], omega)
    cor_i = corr_lin_incoming_correction(incoming[1:], incoming[:-1], psi)
    mod_estimated = mod[:-1] * cor_p * cor_h * cor_i
    return np.sqrt(np.sum((mod[1:] - mod_estimated)**2))

# Objective function to minimize
def objective_mae(params, mod, pressure, incoming, abs_h):
    beta, omega, psi = params
    cor_p = corr_exp_pressure(pressure[1:], pressure[:-1], beta)
    cor_h = corr_lin_humidity(abs_h[1:], abs_h[:-1], omega)
    cor_i = corr_lin_incoming_correction(incoming[1:], incoming[:-1], psi)
    mod_estimated = mod[:-1] * cor_p * cor_h * cor_i
    return np.mean(abs(mod[1:] - mod_estimated))

# Composite objective function
def composite_objective(params, mod, pressure, incoming, abs_h):
    mae = objective_mae(params, mod, pressure, incoming, abs_h)
    rmse = objective_rmse(params, mod, pressure, incoming, abs_h)
    alpha = 0.5
    beta = 0.5
    return alpha * mae + beta * rmse
        
        
def parameter_estimator_a(pressure,neutrons,incoming,humidity,fcttype="exp",plot_flag=1, site_name="default",error_metric="mse"):
    """
    Calculates beta, omgea and phi to correct neutrons for 
    air pressure, incoming cosmic ray intensity, and humidity
    
    Args:
        fcttype (str, values: 'exp' or 'lin'): type of function for correction #not used yet
        pressure (float, range: 500 to 1200, hPa): air pressure time series in hPa
        neutrons (float, range: 200 to 40000, counts): neutron or cosmic ray intensity measured
        incoming (float, range: 0.5 to 1.5, unitless): relative incoming cosmic ray intensity from nmdb.eu after pressure correction
        humidity (float, range: 0 to 35, g/m3): air humidity
        error_metric (str, values: mse rmse or mae): error metric 
        
    Returns:
        parameters (float): beta, omega, phi, betastd, omegastd, phistd values for correction
    
    """
    print("ERROR METRIC:")
    print(error_metric)
    mydf = pd.concat([pressure, neutrons, incoming, humidity], axis=1)
    mydf.columns = ['Pressure', 'MOD', 'Incoming', 'Abs_h']

    #print("Tail of mydf dataframe in parameter_estimator method:")
    #print(mydf.tail(3))
    """
    #check for ranges and remove outliers
    """

    if plot_flag>=1: print("Len before removing values out of range:", len(mydf))
    #print(mydf.tail(48))
    mydf.loc[mydf['MOD'] < 50, 'MOD'] = np.nan
    mydf.loc[mydf['MOD'] > 6000, 'MOD'] = np.nan
    mydf.loc[mydf['Pressure'] < 400, 'Pressure'] = np.nan
    mydf.loc[mydf['Pressure'] > 1200, 'Pressure'] = np.nan
    mydf.loc[mydf['Incoming'] < 0, 'Incoming'] = np.nan
    mydf.loc[mydf['Incoming'] > 10000, 'Incoming'] = np.nan
    mydf.loc[mydf['Abs_h'] < 0, 'Abs_h'] = np.nan
    mydf.loc[mydf['Abs_h'] > 40, 'Abs_h'] = np.nan
    
    mydf=mydf.dropna(subset=['MOD'])
    mydf=mydf.dropna(subset=['Pressure'])
    mydf=mydf.dropna(subset=['Incoming'])
    mydf=mydf.dropna(subset=['Abs_h'])

    #print(mydf.tail(48))
    if plot_flag>=1: print("Len after removing values out of range:", len(mydf))

    """
    #check for nearest neighboor and remove those with neutron count difference larger 10% to nn+/-1
    """
    mydf['MOD_diff_minus_1'] = abs(mydf.MOD.pct_change(periods=-1)) # difference in neutron count
    mydf['MOD_diff_plus_1'] = abs(mydf.MOD.pct_change(periods=1)) # difference in neutron count
    mydf = mydf[(mydf['MOD_diff_minus_1']  <= 0.10)&(mydf['MOD_diff_plus_1']  <= 0.10)]


    mydf['MOD_diff_minus_1'] = abs(mydf.Pressure.pct_change(periods=-1)) # difference in air pressure
    mydf['MOD_diff_plus_1'] = abs(mydf.Pressure.pct_change(periods=1)) # difference in air pressure
    mydf = mydf[(mydf['MOD_diff_minus_1']  <= 0.10)&(mydf['MOD_diff_plus_1']  <= 0.10)]
    if plot_flag>=1: print("Len after removing values MOD_diff_plus_1 and MOD_diff_minus_1:", len(mydf))

    mydf['MOD_diff_minus_1'] = abs(mydf.Abs_h.pct_change(periods=-1)) # difference in air h
    mydf['MOD_diff_plus_1'] = abs(mydf.Abs_h.pct_change(periods=1)) # difference in air h
    mydf = mydf[(mydf['MOD_diff_minus_1']  <= 0.10)&(mydf['MOD_diff_plus_1']  <= 0.10)]
    
    if plot_flag>=1: print("Len after removing values MOD_diff_plus_1 and MOD_diff_minus_1:", len(mydf))
    #print(mydf.tail(10))
    #print(mydf.head(3))

    """
    #plot time series as dots and add rolling means as lines
    """
    # Calculate the running mean (24 hours)
    # Plotting
    if plot_flag>=1:
        running_mean = mydf.rolling(window=24).mean()
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
        # Define the columns to plot
        columns_to_plot = ['MOD', 'Pressure', 'Incoming', 'Abs_h']
        for ax, column in zip(axes, columns_to_plot):
            ax.plot(mydf.index, mydf[column], 'o', color='grey', markersize=3, label=f'{column} values')
            ax.plot(running_mean.index, running_mean[column], color='blue', label=f'{column} 24h running mean')
            ax.set_title(f'{column} over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(column)
            ax.legend()
        # Adjust layout
        plt.tight_layout()
        plt.show()
        plt.close()
        
    #first, accumulate (sum) to daily values
    #second, calculate daily differences with diff(-1)
    #third, minimize the difference by estimateing the parameters beta, omega and psi 
    # so that the difference of (estimated to observed) vectors (MOD(estimated)-MOD(t=0))) is minimized 
    #    using the MOD(t=-1) value times the factors of
    # corr_exp_pressure(Pressure(t=0),Pressure(t=-1), beta)
    # corr_lin_humidity(Abs_H(t=0),Abs_H(t=-1), omega=0.0054)
    # corr_lin_incoming_correction(Incoming(t=0),Incoming(t=-1), psi)

    # Step 1: Accumulate to daily values
    ##Resample sum for neutorn flux and rest to mean
    
    #print(mydf.tail(48))    
    #daily_df = {}  # Dictionary to store resampled DataFrames
    #
    #for col in mydf.columns:
    #    if mydf[col].dtype == 'float64':  # Check if the column contains numeric data
    #        daily_df[col] = mydf[col].resample('D').mean()

    daily_df = mydf.resample('D').mean()
    #daily_df = daily_df.reset_index()

    if plot_flag>=1: print("daily_df Min Max and SD and NA.ct are       :", np.nanmin(daily_df), np.nanmax(daily_df), np.nanstd(daily_df),np.count_nonzero(np.isnan(daily_df)))
    
    daily_df=daily_df.dropna(subset=['MOD'])
    daily_df=daily_df.dropna(subset=['Pressure'])
    daily_df=daily_df.dropna(subset=['Incoming'])
    daily_df=daily_df.dropna(subset=['Abs_h'])
    
    #print(daily_df.tail(10))
    if plot_flag>=1: print("daily_df Min Max and SD and NA.ct NA.drop   :", np.nanmin(daily_df), np.nanmax(daily_df), np.nanstd(daily_df),np.count_nonzero(np.isnan(daily_df)))
    # Step 2: Calculate daily differences
    # Prepare data for optimization
    mod_values = daily_df['MOD'].values
    pressure_values = daily_df['Pressure'].values
    incoming_values = daily_df['Incoming'].values
    abs_h_values = daily_df['Abs_h'].values
    if plot_flag>=1:
        running_mean = mydf.rolling(window=24).mean()
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
        # Define the columns to plot
        columns_to_plot = ['MOD', 'Pressure', 'Incoming', 'Abs_h']
        for ax, column in zip(axes, columns_to_plot):
            ax.plot(mydf.index, mydf[column], 'o', color='grey', markersize=3, label=f'{column} values')
            ax.plot(running_mean.index, running_mean[column], color='blue', label=f'{column} 24h running mean',marker='',linestyle="-")
            ax.plot(daily_df.index, daily_df[column], color='black', label=f'{column} Select',linestyle='',marker='.')
            ax.set_title(f'{column} over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(column)
            ax.legend()
        # Adjust layout
        plt.tight_layout()
        # Save the plot as a PNG file
        plt.savefig(f"./plots/{site_name}_plot.png", bbox_inches='tight')

        # Close the figure to free up memory
        plt.close()



    # Initial guess for the parameters
    initial_params = [-0.0076, -0.0054, 0.5]
    

    # Perform the optimization
    #result = opt.minimize(objective, initial_params, args=(mod_values, pressure_values, incoming_values, abs_h_values))
    
    # Define bounds for each parameter (example values)
    #beta omega psi
    #param_bounds = [(-0.033, 0.0333), (-0.8/35, 0.8/35), (-1.9, 1.9)]  # Adjust as needed
    param_bounds = [(-0.033, 0.0333), (-1.1, 1.1), (-1.9, 1.9)]  # Adjust as needed
    if plot_flag>=1:
        print("param_bounds are:")
        print(param_bounds)
        print("\nPressure:")
        cor_p = corr_exp_pressure(pressure_values[1:], pressure_values[:-1], 0.0076)
        #print(cor_p)
        print("daily_df Min Max and SD and NA.ct are       :", np.nanmin(cor_p), np.nanmax(cor_p), np.nanstd(cor_p),np.count_nonzero(np.isnan(cor_p)))
        #print(pressure_values)
        print("\nHumidity")
        cor_h=corr_lin_humidity(abs_h_values[1:], abs_h_values[:-1], 0.0054)
        #print(cor_h)
        print("daily_df Min Max and SD and NA.ct are       :", np.nanmin(cor_h), np.nanmax(cor_h), np.nanstd(cor_h),np.count_nonzero(np.isnan(cor_h)))
        #print(abs_h_values) 
        print("\nINC")
        cor_i= corr_lin_incoming_correction(incoming_values[1:], incoming_values[:-1], 1)
        #print(cor_i)
        print("daily_df Min Max and SD and NA.ct are       :", np.nanmin(cor_i), np.nanmax(cor_i), np.nanstd(cor_i),np.count_nonzero(np.isnan(cor_i)))
        #print(incoming_values)
        print("INC samples\n")
        #print(corr_lin_incoming_correction(1,0.9, 1))
        #print(corr_lin_incoming_correction(0.9,1, 1))
        #print(corr_lin_incoming_correction(0.9,1, 0.5))
        print("Perform opt")
    print(error_metric.lower())
    # Perform the optimization
    if error_metric.lower() == "mse":
        result = opt.minimize(objective, initial_params, args=(mod_values, pressure_values, incoming_values, abs_h_values), bounds=param_bounds)
    elif error_metric.lower() == "rmse":
        result = opt.minimize(objective_rmse, initial_params, args=(mod_values, pressure_values, incoming_values, abs_h_values), bounds=param_bounds)
    elif error_metric.lower() == "mae":
        result = opt.minimize(objective_mae, initial_params, args=(mod_values, pressure_values, incoming_values, abs_h_values), bounds=param_bounds)
    else:
        print("variable error_metric badly set")
        quitprogramnow("variable error_metric badly set")

    beta_opt, omega_opt, psi_opt = result.x
    if plot_flag>=1: print(f"\nOptimized parameters:     beta(P)={beta_opt:.5f},     omega(h)={omega_opt:.5f},     psi(i)={psi_opt:.5f}")
    if plot_flag>=1: 
        print("Parameter bounds were:", param_bounds)
        print('\nBlock completed at time:                       '+ time.strftime("%H:%M:%S on %Y-%m-%d")) 
    
    jac = jacobian([beta_opt, omega_opt, psi_opt], mod_values, pressure_values, incoming_values, abs_h_values)
    hessian = jac.T @ jac
    cov_matrix = np.linalg.inv(hessian)
    uncertainties = np.sqrt(np.diag(cov_matrix))
    
    param_estimates = result.x  # Replace with actual optimized parameters
    # Create a DataFrame
    param_names = ['param1', 'param2', 'param3']
    df = pd.DataFrame({
        'Parameter': param_names,
        'Estimate': param_estimates,
        'Uncertainty': uncertainties
    })
    if plot_flag>=1: 
        print(df)
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['Parameter'], df['Estimate'], yerr=df['Uncertainty'], fmt='*', capsize=5, color='blue', ecolor='red', elinewidth=2, capthick=2)
        plt.ylabel('Parameter Estimate')
        plt.title('Parameter Estimates with Uncertainties')
        plt.grid(True)
        plt.show()
    
    # Print the 

    # Example data and initial parameters
    #mod_values = np.random.randn(100)
    #pressure_values = np.random.randn(100)
    #incoming_values = np.random.randn(100)
    #abs_h_values = np.random.randn(100)
    # Choose block length
    block_length = int(len(mod_values)/4) #block_size
    param_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5)]

    # Estimate uncertainties
    # Estimate uncertainties
    uncertainties = bootstrap_fit(initial_params, mod_values, pressure_values, incoming_values, abs_h_values, param_bounds, block_length)

    param_estimates = result.x  # Replace with actual optimized parameters
    # Create a DataFrame
    param_names = ['param1', 'param2', 'param3']
    df = pd.DataFrame({
        'Parameter': param_names,
        'Estimate': param_estimates,
        'Uncertainty': uncertainties
    })


    if plot_flag>=1: 
        print(df)
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['Parameter'], df['Estimate'], yerr=df['Uncertainty'], fmt='*', capsize=5, color='blue', ecolor='red', elinewidth=2, capthick=2)
        plt.ylabel('Parameter Estimate')
        plt.title('Parameter Estimates with Uncertainties')
        plt.grid(True)
        plt.show()
    
    # Print the uncertainties
    if plot_flag>=1: print(f"Uncertainties: beta={uncertainties[0]:.8f}, omega(h)={uncertainties[1]:.8f}, psi(i)={uncertainties[2]:.8f}")

    if plot_flag>=2:
        # Plotting the correction factors
        cor_p = corr_exp_pressure(pressure_values[1:], pressure_values[:-1], beta_opt)
        cor_h = corr_lin_humidity(abs_h_values[1:], abs_h_values[:-1], omega_opt)
        cor_i = corr_lin_incoming_correction(incoming_values[1:], incoming_values[:-1], psi_opt)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(cor_p, 'o', color='grey')
        plt.title('Pressure Correction Factor')

        plt.subplot(1, 3, 2)
        plt.plot(cor_h, 'o', color='grey')
        plt.title('Humidity Correction Factor')

        plt.subplot(1, 3, 3)
        plt.plot(cor_i, 'o', color='grey')
        plt.title('Incoming Radiation Correction Factor')

        plt.tight_layout()
        plt.show()

    if plot_flag>=1: print(f"Optimized parameters: beta(P)={beta_opt:.5f} +/-{uncertainties[0]:.6f}, omega(h)={omega_opt:.5f} +/-{uncertainties[1]:.6f}, psi(i)={psi_opt:.2f} +/-{uncertainties[2]:.3f}")
    
    parameters=[beta_opt, omega_opt, psi_opt,uncertainties[0],uncertainties[1],uncertainties[2],np.nanmean(pressure_values)]
    
    return parameters

def parameter_estimator(pressure,neutrons,incoming,humidity,fcttype="exp",plot_flag=1, site_name="default",error_metric="mse"):
    """
    Calculates beta, omgea and phi to correct neutrons for 
    air pressure, incoming cosmic ray intensity, and humidity
    
    Args:
        fcttype (str, values: 'exp' or 'lin'): type of function for correction #not used yet
        pressure (float, range: 500 to 1200, hPa): air pressure time series in hPa
        neutrons (float, range: 200 to 40000, counts): neutron or cosmic ray intensity measured
        incoming (float, range: 0.5 to 1.5, unitless): relative incoming cosmic ray intensity from nmdb.eu after pressure correction
        humidity (float, range: 0 to 35, g/m3): air humidity
        error_metric (str, values: mse rmse or mae): error metric 
        
    Returns:
        parameters (float): beta, omega, phi, betastd, omegastd, phistd values for correction
    
    """
    
    mydf = pd.concat([pressure, neutrons, incoming, humidity], axis=1)
    mydf.columns = ['Pressure', 'MOD', 'Incoming', 'Abs_h']

    #print("Tail of mydf dataframe in parameter_estimator method:")
    #print(mydf.tail(3))
    """
    #check for ranges and remove outliers
    """

    if plot_flag>=1: print("Len before removing values out of range:", len(mydf))
    #print(mydf.tail(48))
    mydf.loc[mydf['MOD'] < 50, 'MOD'] = np.nan
    mydf.loc[mydf['MOD'] > 6000, 'MOD'] = np.nan
    mydf.loc[mydf['Pressure'] < 400, 'Pressure'] = np.nan
    mydf.loc[mydf['Pressure'] > 1200, 'Pressure'] = np.nan
    mydf.loc[mydf['Incoming'] < 0.5, 'Incoming'] = np.nan
    mydf.loc[mydf['Incoming'] > 1.5, 'Incoming'] = np.nan
    mydf.loc[mydf['Abs_h'] < 0, 'Abs_h'] = np.nan
    mydf.loc[mydf['Abs_h'] > 35, 'Abs_h'] = np.nan
    
    mydf=mydf.dropna(subset=['MOD'])
    mydf=mydf.dropna(subset=['Pressure'])
    mydf=mydf.dropna(subset=['Incoming'])
    mydf=mydf.dropna(subset=['Abs_h'])

    #print(mydf.tail(48))
    if plot_flag>=1: print("Len after removing values out of range:", len(mydf))

    """
    #check for nearest neighboor and remove those with neutron count difference larger 10% to nn+/-1
    """
    mydf['MOD_diff_minus_1'] = abs(mydf.MOD.pct_change(periods=-1)) # difference in neutron count
    mydf['MOD_diff_plus_1'] = abs(mydf.MOD.pct_change(periods=1)) # difference in neutron count
    mydf = mydf[(mydf['MOD_diff_minus_1']  <= 0.10)&(mydf['MOD_diff_plus_1']  <= 0.10)]


    mydf['MOD_diff_minus_1'] = abs(mydf.Pressure.pct_change(periods=-1)) # difference in air pressure
    mydf['MOD_diff_plus_1'] = abs(mydf.Pressure.pct_change(periods=1)) # difference in air pressure
    mydf = mydf[(mydf['MOD_diff_minus_1']  <= 0.10)&(mydf['MOD_diff_plus_1']  <= 0.10)]
    if plot_flag>=1: print("Len after removing values MOD_diff_plus_1 and MOD_diff_minus_1:", len(mydf))

    mydf['MOD_diff_minus_1'] = abs(mydf.Abs_h.pct_change(periods=-1)) # difference in air h
    mydf['MOD_diff_plus_1'] = abs(mydf.Abs_h.pct_change(periods=1)) # difference in air h
    mydf = mydf[(mydf['MOD_diff_minus_1']  <= 0.10)&(mydf['MOD_diff_plus_1']  <= 0.10)]
    
    if plot_flag>=1: print("Len after removing values MOD_diff_plus_1 and MOD_diff_minus_1:", len(mydf))
    #print(mydf.tail(10))
    #print(mydf.head(3))

    """
    #plot time series as dots and add rolling means as lines
    """
    # Calculate the running mean (24 hours)
    # Plotting
    if plot_flag>=1:
        running_mean = mydf.rolling(window=24).mean()
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
        # Define the columns to plot
        columns_to_plot = ['MOD', 'Pressure', 'Incoming', 'Abs_h']
        for ax, column in zip(axes, columns_to_plot):
            ax.plot(mydf.index, mydf[column], 'o', color='grey', markersize=3, label=f'{column} values')
            ax.plot(running_mean.index, running_mean[column], color='blue', label=f'{column} 24h running mean')
            ax.set_title(f'{column} over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(column)
            ax.legend()
        # Adjust layout
        plt.tight_layout()
        plt.show()
        plt.close()
        
    #first, accumulate (sum) to daily values
    #second, calculate daily differences with diff(-1)
    #third, minimize the difference by estimateing the parameters beta, omega and psi 
    # so that the difference of (estimated to observed) vectors (MOD(estimated)-MOD(t=0))) is minimized 
    #    using the MOD(t=-1) value times the factors of
    # corr_exp_pressure(Pressure(t=0),Pressure(t=-1), beta)
    # corr_lin_humidity(Abs_H(t=0),Abs_H(t=-1), omega=0.0054)
    # corr_lin_incoming_correction(Incoming(t=0),Incoming(t=-1), psi)

    # Step 1: Accumulate to daily values
    ##Resample sum for neutorn flux and rest to mean
    
    #print(mydf.tail(48))    
    #daily_df = {}  # Dictionary to store resampled DataFrames
    #
    #for col in mydf.columns:
    #    if mydf[col].dtype == 'float64':  # Check if the column contains numeric data
    #        daily_df[col] = mydf[col].resample('D').mean()
    
    daily_df = mydf.resample('D').mean()

    if plot_flag>=1: print("daily_df Min Max and SD and NA.ct are       :", np.nanmin(daily_df), np.nanmax(daily_df), np.nanstd(daily_df),np.count_nonzero(np.isnan(daily_df)))
    
    daily_df=daily_df.dropna(subset=['MOD'])
    daily_df=daily_df.dropna(subset=['Pressure'])
    daily_df=daily_df.dropna(subset=['Incoming'])
    daily_df=daily_df.dropna(subset=['Abs_h'])

    daily_df = daily_df.reset_index()
    
    # Calculate the difference between consecutive days
    daily_df['Day_Diff'] = daily_df['Date'].diff().dt.days
    
    # Identify valid indices where the difference is exactly 1 day
    valid_indices = daily_df[daily_df['Day_Diff'] == 1].index
    with open("your_file.txt", "a") as file:
        file.write(daily_df.tail(20).to_string(index=False) + "\n")

    if plot_flag>=1: print("daily_df Min Max and SD and NA.ct NA.drop   :", np.nanmin(daily_df), np.nanmax(daily_df), np.nanstd(daily_df),np.count_nonzero(np.isnan(daily_df)))
    # Step 2: Calculate daily differences
    # Prepare data for optimization
    mod_values = daily_df['MOD']
    pressure_values = daily_df['Pressure']
    incoming_values = daily_df['Incoming']
    abs_h_values = daily_df['Abs_h']
    
    if plot_flag>=1:
        running_mean = mydf.rolling(window=24).mean()
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16))
        # Define the columns to plot
        columns_to_plot = ['MOD', 'Pressure', 'Incoming', 'Abs_h']
        for ax, column in zip(axes, columns_to_plot):
            ax.plot(mydf.index, mydf[column], 'o', color='grey', markersize=3, label=f'{column} values')
            ax.plot(running_mean.index, running_mean[column], color='blue', label=f'{column} 24h running mean',marker='',linestyle="-")
            ax.plot(daily_df.index, daily_df[column], color='black', label=f'{column} Select',linestyle='',marker='.')
            ax.set_title(f'{column} over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(column)
            ax.legend()
        # Adjust layout
        plt.tight_layout()
        # Save the plot as a PNG file
        plt.savefig(f"./plots/{site_name}_plot.png", bbox_inches='tight')

        # Close the figure to free up memory
        plt.close()



    # Initial guess for the parameters
    initial_params = [-0.0076, -0.0054, 0.5]
    

    # Perform the optimization
    #result = opt.minimize(objective, initial_params, args=(mod_values, pressure_values, incoming_values, abs_h_values))
    
    # Define bounds for each parameter (example values)
    #beta omega psi
    #param_bounds = [(-0.033, 0.0333), (-0.8/35, 0.8/35), (-1.9, 1.9)]  # Adjust as needed
    param_bounds = [(-0.033, 0.0333), (-1.1, 1.1), (-1.9, 1.9)]  # Adjust as needed
    if plot_flag>=1:
        print("param_bounds are:")
        print(param_bounds)
        print("\nPressure:")
        cor_p = corr_exp_pressure(pressure_values[1:], pressure_values[:-1], 0.0076)
        #print(cor_p)
        print("daily_df Min Max and SD and NA.ct are       :", np.nanmin(cor_p), np.nanmax(cor_p), np.nanstd(cor_p),np.count_nonzero(np.isnan(cor_p)))
        #print(pressure_values)
        print("\nHumidity")
        cor_h=corr_lin_humidity(abs_h_values[1:], abs_h_values[:-1], 0.0054)
        #print(cor_h)
        print("daily_df Min Max and SD and NA.ct are       :", np.nanmin(cor_h), np.nanmax(cor_h), np.nanstd(cor_h),np.count_nonzero(np.isnan(cor_h)))
        #print(abs_h_values) 
        print("\nINC")
        cor_i= corr_lin_incoming_correction(incoming_values[1:], incoming_values[:-1], 1)
        #print(cor_i)
        print("daily_df Min Max and SD and NA.ct are       :", np.nanmin(cor_i), np.nanmax(cor_i), np.nanstd(cor_i),np.count_nonzero(np.isnan(cor_i)))
        #print(incoming_values)
        print("INC samples\n")
        #print(corr_lin_incoming_correction(1,0.9, 1))
        #print(corr_lin_incoming_correction(0.9,1, 1))
        #print(corr_lin_incoming_correction(0.9,1, 0.5))
        print("Perform opt")
    valid_indices = daily_df[daily_df['Day_Diff'] == 1].index
    # Create arrays for pressure values on consecutive days
    pressure_valid = daily_df.loc[valid_indices, 'Pressure'].values
    pressure_prev_valid = daily_df.loc[valid_indices - 1, 'Pressure'].values

    # Create arrays for pressure values on consecutive days
    abs_valid = daily_df.loc[valid_indices, 'Abs_h'].values
    abs_prev_valid = daily_df.loc[valid_indices - 1, 'Abs_h'].values
    i_valid = daily_df.loc[valid_indices, 'Incoming'].values
    i_prev_valid = daily_df.loc[valid_indices - 1, 'Incoming'].values
    
    mod_valid = daily_df.loc[valid_indices, 'MOD'].values
    mod_prev_valid = daily_df.loc[valid_indices - 1, 'MOD'].values
    
    # Combine inputs into a single tuple
    #data = (pressure_valid, pressure_prev_valid, abs_valid, abs_prev_valid, i_valid, i_prev_valid, mod_prev_valid, mod_valid)
    data = np.array([pressure_valid, pressure_prev_valid, abs_valid, abs_prev_valid, i_valid, i_prev_valid, mod_prev_valid, mod_valid])
    
    
    # Perform the optimization
    if error_metric.lower() == "mse":
        #result = opt.minimize(objective, initial_params, args=(mod_values, pressure_values, incoming_values, abs_h_values), bounds=param_bounds)
        #result = opt.minimize(objective_consecutive, initial_params, args=(daily_df), bounds=param_bounds)
        result = opt.minimize(objective_consecutive, initial_params, args=(pressure_valid,pressure_prev_valid,abs_valid,abs_prev_valid,i_valid,i_prev_valid,mod_prev_valid,mod_valid), bounds=param_bounds)

    elif error_metric.lower() == "rmse" or error_metric.lower() == "rmse_u":
        #result = opt.minimize(objective_consecutive_rmse, initial_params, args=(daily_df), bounds=param_bounds)
        result = opt.minimize(objective_consecutive_rmse, initial_params, args=(pressure_valid,pressure_prev_valid,abs_valid,abs_prev_valid,i_valid,i_prev_valid,mod_prev_valid,mod_valid), bounds=param_bounds)
        #result = opt.minimize(objective_rmse, initial_params, args=(mod_values, pressure_values, incoming_values, abs_h_values), bounds=param_bounds)
    elif error_metric.lower() == "mae":
        result = opt.minimize(objective_consecutive_mae, initial_params, args=(data,), bounds=param_bounds)
        #result = opt.minimize(objective_consecutive_mae, initial_params, args=(pressure_valid,pressure_prev_valid,abs_valid,abs_prev_valid,i_valid,i_prev_valid,mod_prev_valid,mod_valid), bounds=param_bounds)
        
        #result = opt.minimize(objective_mae, initial_params, args=(mod_values, pressure_values, incoming_values, abs_h_values), bounds=param_bounds)
    elif error_metric.lower() == "nelder" or error_metric.lower() == "nelder_u":
        result = opt.minimize(objective_consecutive_rmse,  # Your objective function
            initial_params,              # Initial guess for the parameters
            args=(pressure_valid, pressure_prev_valid, abs_valid, abs_prev_valid, i_valid, i_prev_valid, mod_prev_valid, mod_valid),  # Arguments for your objective
            method='Nelder-Mead',        # Specify Nelder-Mead as the method
            options={'disp': False, 'maxiter': 100000, 'xatol': 1e-4, 'fatol': 1e-4}  # Display progress and allow more iterations
            )
        # Check if the optimization was successful
        
        if result.success:
            pass
            #print("Optimization converged successfully!")
        else:
            print("Optimization did not converge. Reason:", result.message)

        ## Display important results
        #print("Number of iterations:", result.nit)
        #print("Final objective function value (RMSE/MAE):", result.fun)
        #print("Optimized parameters:", result.x)
    elif error_metric.lower() == "nelder_mae":
        result = opt.minimize(objective_consecutive_mae,  # Your objective function
            initial_params,              # Initial guess for the parameters
            args=(data,),  # Arguments for your objective
            method='Nelder-Mead',        # Specify Nelder-Mead as the method
            options={'disp': False, 'maxiter': 100000, 'xatol': 1e-10, 'fatol': 1e-10}  # Display progress and allow more iterations
            )
        # Check if the optimization was successful
        
        if result.success:
            pass
            #print("Optimization converged successfully!")
        else:
            print("Optimization did not converge. Reason:", result.message)

        ## Display important results
        #print("Number of iterations:", result.nit)
        #print("Final objective function value (RMSE/MAE):", result.fun)
        #print("Optimized parameters:", result.x)
    else:
        print("variable error_metric badly set")
        quitprogramnow("variable error_metric badly set")
    
    beta_opt, omega_opt, psi_opt = result.x
    if plot_flag>=1: print(f"\nOptimized parameters:     beta(P)={beta_opt:.5f},     omega(h)={omega_opt:.5f},     psi(i)={psi_opt:.5f}")
    if plot_flag>=1: 
        print("Parameter bounds were:", param_bounds)
        print('\nBlock completed at time:                       '+ time.strftime("%H:%M:%S on %Y-%m-%d")) 
    
    jac = jacobian([beta_opt, omega_opt, psi_opt], mod_values, pressure_values, incoming_values, abs_h_values)
    hessian = jac.T @ jac
    cov_matrix = np.linalg.inv(hessian)
    uncertainties = np.sqrt(np.diag(cov_matrix))
    
    param_estimates = result.x  # Replace with actual optimized parameters
    # Create a DataFrame
    param_names = ['param1', 'param2', 'param3']
    df = pd.DataFrame({
        'Parameter': param_names,
        'Estimate': param_estimates,
        'Uncertainty': uncertainties
    })
    if plot_flag>=1: 
        print(df)
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['Parameter'], df['Estimate'], yerr=df['Uncertainty'], fmt='*', capsize=5, color='blue', ecolor='red', elinewidth=2, capthick=2)
        plt.ylabel('Parameter Estimate')
        plt.title('Parameter Estimates with Uncertainties')
        plt.grid(True)
        plt.show()
    
    # Print the 

    # Example data and initial parameters
    #mod_values = np.random.randn(100)
    #pressure_values = np.random.randn(100)
    #incoming_values = np.random.randn(100)
    #abs_h_values = np.random.randn(100)
    # Choose block length
    block_length = int(len(mod_values)/7) #block_size
    param_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5)]
    
    # Estimate uncertainties
    # Estimate uncertainties
    if error_metric.lower() == "nelder_u":
        # Run bootstrap to calculate uncertainties
        num_bootstrap = 100  # Number of bootstrap samples
        #block_length = int(len(mod_values)/4)  # Define the block length for moving block bootstrap
        #print(block_length)
        uncertainties = bootstrap_uncertainty(num_bootstrap, block_length, initial_params, data)
        
        #print(f"Parameter uncertainties: beta={uncertainties[0]:.5f}, omega={uncertainties[1]:.4f}, psi={uncertainties[2]:.3f}")
        if plot_flag>=1: print(f"Parameter uncertainties: 1={uncertainties[0]:.5f}, 2={uncertainties[1]:.4f}, 3={uncertainties[2]:.3f}")
        
    elif error_metric.lower() == "rmse_u":
        # Run bootstrap to calculate uncertainties
        #print(data.shape)
        #print(block_length)
        #print("Run uncertainties")
        uncertainties = bootstrap_fit_rmse_u(initial_params, data, param_bounds, block_length)
        
        #print(f"Parameter uncertainties: beta={uncertainties[0]:.5f}, omega={uncertainties[1]:.4f}, psi={uncertainties[2]:.3f}")
        if plot_flag>=1: print(f"Parameter uncertainties: 1={uncertainties[0]:.5f}, 2={uncertainties[1]:.4f}, 3={uncertainties[2]:.3f}")
        
    else:
        uncertainties = bootstrap_fit(initial_params, mod_values, pressure_values, incoming_values, abs_h_values, param_bounds, block_length)
        #print(len(pressure_values),end= ' ')

    param_estimates = result.x  # Replace with actual optimized parameters
    # Create a DataFrame
    param_names = ['param1', 'param2', 'param3']
    df = pd.DataFrame({
        'Parameter': param_names,
        'Estimate': param_estimates,
        'Uncertainty': uncertainties
    })


    if plot_flag>=1: 
        print(df)
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['Parameter'], df['Estimate'], yerr=df['Uncertainty'], fmt='*', capsize=5, color='blue', ecolor='red', elinewidth=2, capthick=2)
        plt.ylabel('Parameter Estimate')
        plt.title('Parameter Estimates with Uncertainties')
        plt.grid(True)
        plt.show()
    
    # Print the uncertainties
    if plot_flag>=1: print(f"Uncertainties: beta={uncertainties[0]:.8f}, omega(h)={uncertainties[1]:.8f}, psi(i)={uncertainties[2]:.8f}")

    if plot_flag>=2:
        # Plotting the correction factors
        cor_p = corr_exp_pressure(pressure_values[1:], pressure_values[:-1], beta_opt)
        cor_h = corr_lin_humidity(abs_h_values[1:], abs_h_values[:-1], omega_opt)
        cor_i = corr_lin_incoming_correction(incoming_values[1:], incoming_values[:-1], psi_opt)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(cor_p, 'o', color='grey')
        plt.title('Pressure Correction Factor')

        plt.subplot(1, 3, 2)
        plt.plot(cor_h, 'o', color='grey')
        plt.title('Humidity Correction Factor')

        plt.subplot(1, 3, 3)
        plt.plot(cor_i, 'o', color='grey')
        plt.title('Incoming Radiation Correction Factor')

        plt.tight_layout()
        plt.show()

    if plot_flag>=1: print(f"Optimized parameters: beta(P)={beta_opt:.5f} +/-{uncertainties[0]:.6f}, omega(h)={omega_opt:.5f} +/-{uncertainties[1]:.6f}, psi(i)={psi_opt:.2f} +/-{uncertainties[2]:.3f}")
    #print(f"Optimized parameters: beta(P)={beta_opt:.5f} +/-{uncertainties[0]:.6f}, omega(h)={omega_opt:.5f} +/-{uncertainties[1]:.6f}, psi(i)={psi_opt:.2f} +/-{uncertainties[2]:.3f}")
    parameters=[beta_opt, omega_opt, psi_opt,uncertainties[0],uncertainties[1],uncertainties[2],np.nanmean(pressure_values),daily_df[daily_df['Day_Diff'] == 1].shape[0],np.nanmean(mod_values)]

    return parameters
