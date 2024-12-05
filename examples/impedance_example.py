"""Example how to generate, modify and plot impedance objects."""
import pysignalscope as pss

# import data as a channel object
example_data_0mm5 = pss.Impedance.from_waynekerr('impedance_example_data_0mm5.csv', '0.5mm air gap')
example_data_1mm5 = pss.Impedance.from_waynekerr('impedance_example_data1mm5.csv', '1.5mm air gap')
example_data_rlc = pss.Impedance.from_rlc('l', 1000, 500e-6, 10e-12)

# modify the data, e.g. change the color.
# For color changes, you can use standard colors or colors from the leapythontoolbox, see this example
example_data_0mm5 = pss.Impedance.modify(example_data_0mm5, channel_color="red", channel_unit="random unit")
example_data_1mm5 = pss.Impedance.modify(example_data_1mm5, channel_color=pss.gnome_colors["blue"])
example_data_rlc = pss.Impedance.modify(example_data_rlc, channel_color=pss.gnome_colors["green"], channel_label="from rlc")

# # recalculate rlc data from a impedance curve
recalculated_r, recalculated_l, recalculated_c = (pss.Impedance.calc_rlc(example_data_rlc, 'l', f_calc_l=10e3, f_calc_c=10e7, plot_figure=True))
print(f"{recalculated_r=}")
print(f"{recalculated_l=}")
print(f"{recalculated_c=}")

# plot multiple channel data
pss.Impedance.plot_impedance([example_data_0mm5, example_data_1mm5, example_data_rlc])

# save and load an impedance object
pss.Impedance.save(example_data_rlc, "example")
loaded_impedance_object = pss.Impedance.load("example.pkl")
