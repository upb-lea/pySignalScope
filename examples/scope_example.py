"""Example file to demonstrate the scope functionality."""
import pysignalscope as pss

# Use this line to change the fonts into LaTeX font.
# pss.global_plot_settings_font_latex()

# ------------------------------------------
# Example 1: Read curves from tektronix scope csv file, plot the signals and perform FFT
# ------------------------------------------

# Read curves from scope csv file
[voltage_prim, voltage_sec, current_prim, current_sec] = pss.HandleScope.from_tektronix('scope_example_data_tek.csv')

# Add labels and units to channels
voltage_prim = pss.HandleScope.modify(voltage_prim, channel_label='voltage primary', channel_unit='V', )
voltage_sec = pss.HandleScope.modify(voltage_sec, channel_label='voltage secondary', channel_unit='V')
current_prim = pss.HandleScope.modify(current_prim, channel_label='current primary', channel_unit='A')
current_sec = pss.HandleScope.modify(current_sec, channel_label='Secondary current', channel_unit='A')

# Show gain and DC-offset
current_sec = pss.HandleScope.modify(current_sec, channel_data_factor=1.3, channel_data_offset=10)

# Plot channels
fig1 = pss.HandleScope.plot_channels([voltage_prim, voltage_sec], [current_prim, current_sec], timebase='us')
pss.HandleScope.save(fig1, 'test')

# short channels to a single period, perform FFT for current waveforms
current_prim = pss.HandleScope.short_to_period(current_prim, f0=200000)
current_prim = pss.HandleScope.modify(current_prim, channel_time_shift=5e-6)
# pss.HandleScope.fft(current_prim)

# ------------------------------------------
# Example 2: Read curves from LeCroy csv-Files and GeckoCirucits. Compare these signals.
# ------------------------------------------

meas_v_ob, meas_il_ib, meas_il_ob = pss.HandleScope.from_lecroy('scope_example_data_lecroy_1.csv',
                                                                'scope_example_data_lecroy_2.csv',
                                                                'scope_example_data_lecroy_3.csv')

meas_v_ob = pss.HandleScope.modify(meas_v_ob, channel_label='vl_ob_meas', channel_unit='V')
meas_il_ib = pss.HandleScope.modify(meas_il_ib, channel_label='il_ib_meas', channel_unit='A', channel_color='b')
meas_il_ob = pss.HandleScope.modify(meas_il_ob, channel_label='il_ob_meas', channel_unit='A', channel_color='m', channel_data_offset=1.5)

# read dataset from gecko simulation
gecko_data = pss.HandleScope.from_geckocircuits('scope_example_data_gecko', f0=200000)

gecko_il_ib = gecko_data[0]
gecko_il_ib = pss.HandleScope.modify(gecko_il_ib, channel_label='il_ib_gecko', channel_unit='A', channel_color='r', channel_linestyle="--")
gecko_il_ob = gecko_data[-1]
gecko_il_ob = pss.HandleScope.modify(gecko_il_ob, channel_data_factor=-1, channel_label='il_ob_gecko', channel_unit='A', channel_color='g')

# compare both waveforms
# pss.HandleScope.compare_channels(meas_il_ib, gecko_il_ib, meas_il_ob, gecko_il_ob, shift=[-67.53e-6, 0, -67.53e-6, 0], timebase='us')

# Shift data
meas_il_ob = pss.HandleScope.modify(meas_il_ob, channel_time_shift=-67.53e-6)
# Calculate sum
sum_il_ob = pss.HandleScope.add(meas_il_ob, meas_il_ob)

# Calculate difference
diff_il_ob = pss.HandleScope.subtract(sum_il_ob, meas_il_ob)

# Copy values to new variable
abs_diff_il_ob = pss.HandleScope.copy(diff_il_ob)
# Calculate absolute values
pss.HandleScope.abs(abs_diff_il_ob)
# Copy values to new variable
sqr_diff_il_ob = pss.HandleScope.copy(diff_il_ob)
# Calculate square
pss.HandleScope.square(sqr_diff_il_ob)
# Calculate values:
# Root means square
rms_diff_il_ob = pss.HandleScope.rms(diff_il_ob)
# Average value
mean_diff_il_ob = pss.HandleScope.mean(diff_il_ob)
# Average of absolute values
absmean_diff_il_ob = pss.HandleScope.absmean(diff_il_ob)
# Print calculated values
print(f"Rootmeansquare={rms_diff_il_ob}\nAverage value={mean_diff_il_ob}\nAbsolute average value={absmean_diff_il_ob}\n")
# Plot results
pss.HandleScope.plot_channels([diff_il_ob])
# Plot results
pss.HandleScope.plot_channels([diff_il_ob, abs_diff_il_ob, sqr_diff_il_ob])
