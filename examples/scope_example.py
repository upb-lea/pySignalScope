"""Example file to demonstrate the scope functionality."""
import pysignalscope as pss

pss.global_plot_settings_font_latex()

# ------------------------------------------
# Example 1: Read curves from tektronix scope csv file, plot the signals and perform FFT
# ------------------------------------------

# Read curves from scope csv file
[voltage_prim, voltage_sec, current_prim, current_sec] = pss.Scope.from_tektronix('scope_example_data_tek.csv')

# Add labels and units to channels
voltage_prim.modify(channel_label='voltage primary', channel_unit='V', )
voltage_sec.modify(channel_label='voltage secondary', channel_unit='V')
current_prim.modify(channel_label='current primary', channel_unit='A')
current_sec.modify(channel_label='current secondary', channel_unit='A')

# Show gain and DC-offset
current_sec.modify(channel_data_factor=1.3, channel_data_offset=10)

# Plot channels
fig1 = pss.Scope.plot_channels([voltage_prim, voltage_sec], [current_prim, current_sec], timebase='us')
pss.Scope.save(fig1, 'test')

# short channels to a single period, perform FFT for current waveforms
current_prim.short_to_period(f0=200000)
current_prim.modify(channel_time_shift=5e-6)
current_prim.fft()

# ------------------------------------------
# Example 2: Read curves from LeCroy csv-Files and GeckoCirucits. Compare these signals.
# ------------------------------------------

meas_v_ob, meas_il_ib, meas_il_ob = pss.Scope.from_lecroy('scope_example_data_lecroy_1.csv',
                                                          'scope_example_data_lecroy_2.csv',
                                                          'scope_example_data_lecroy_3.csv')

meas_v_ob.modify(channel_label='vl_ob_meas', channel_unit='V')
meas_il_ib.modify(channel_label='il_ib_meas', channel_unit='A', channel_color='b')
meas_il_ob.modify(channel_label='il_ob_meas', channel_unit='A', channel_color='m', channel_data_offset=1.5)

# read dataset from gecko simulation
gecko_data = pss.Scope.from_geckocircuits('scope_example_data_gecko', f0=200000)

gecko_il_ib = gecko_data[0]
gecko_il_ib.modify(channel_label='il_ib_gecko', channel_unit='A', channel_color='r', channel_linestyle="--")
gecko_il_ob = gecko_data[-1]
gecko_il_ob.modify(channel_data_factor=-1, channel_label='il_ob_gecko', channel_unit='A', channel_color='g')

# compare both waveforms
pss.Scope.compare_channels(meas_il_ib, gecko_il_ib, meas_il_ob, gecko_il_ob, shift=[-67.53e-6, 0, -67.53e-6, 0], timebase='us')
