"""Class supports shifting channels within displayed plot."""
from enum import Enum
from typing import List, Tuple
import os
# 3rd party libraries
from matplotlib import pyplot as plt
# own libraries
from pysignalscope.logconfig import setup_logging
from pysignalscope.scope_dataclass import Channel
# python libraries
import logging
# Interactive shift plot
import matplotlib
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
from matplotlib.widgets import Slider

# in case of GitHubs CI, use Agg instead of TkAgg (normal usage)
if "IS_TEST" in os.environ:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

# - Logging setup ---------------------------------------------------------------------------------
setup_logging()

# Modul name für static methods
class_modulename = "channelshift"

# - Class definition ------------------------------------------------------------------------------

class ScopeChShift:
    """Class to shift channels interactively."""

    # - private members ------------------------------------------------------------------------------

    # Index of selected channel
    chn_index = 0
    last_val = 0
    shift_dir = 1

    # Shiftfigure
    shiftfig = None
    # Zoom axe
    zoom_ax = None
    # Reference channels to plot
    channelplotlist = list()
    # List of shift of channels
    shiftlist = list()

    # Widget container variables
    shift_text_box = None
    shsl_reset_button = None
    chn_sel_button = None
    dir_shift_button = None
    shift_slider = None
    selplotlabel = None

    # Shift step variables
    shiftstep_x = None
    shiftstep_y = None
    # Limits of shift steps
    shiftstep_x: List[float] = [0, 0]
    min_shiftstep_x = None
    max_shiftstep_x = None
    min_shiftstep_y = None
    max_shiftstep_y = None

    # Maximal values
    display_min_y = None
    display_max_x = None
    display_min_y = None
    display_max_y = None

    # Minimal zoom values
    zoom_delta_y = None
    zoom_delta_x = None

    # Shift zoom variables
    zoom_start_x = None
    zoom_start_y = None
    # Pixelbox for the graphic
    shiftfigbox = None
    # Zoomrect
    zoom_rect = None

    # Zoom state variabe

    class Zoom_State(Enum):
        """Enumeration to control the zoom state."""

        NoZoom = 0
        ZoomSelect = 1
        ZoomConfirm = 2

    zoom_state = Zoom_State.NoZoom

    @staticmethod
    def channel_shift(channels: List['Channel'], shiftstep_x: float, shiftstep_y: float, \
                      displayrange_x: Tuple[float, float], displayrange_y: Tuple[float, float]):
        """
        Interactive plot of channel datasets.

        Examples:
        >>> import channelshift as chshift
        >>> scopechannels=generate_scope_object([[0, 5e-4, 10e-4, 15e-4, 20e-4],[20,150,130,140,2]],[[10e-4, 12e-4, 21-4, 35e-4, 40e-4],[-20,130,-30,100,40]])
        >>> chshift.init_shiftstep_limits((1e-6,5e-5),(1,20))
        >>> chshift.ScopeChShift.plot_shiftchannels(scopechannels, 1e-5,10,[0,40e-4],[-100,200])
        Plots the channels ch1 and ch2. You can zoom into by selecting the zoom area with help of
        left mouse button. By moving the mouse while pressing the button  the area is marked by a red rectangle.
        If you release the left mouse button the area is marked. By moving the mouse within the area an perform
        a button press, you confirm and you zoom in the area. If you perform the left mouse button click outside
        of the marked area, you reject the selection. You reject the selection always by clicking the right mouse button independent you zoom out.
        button. If no area is selected or wait for confirmation, the click on the right mouse button leads to zooming out.
        There is a zoom limit in both directions. In this case, the rectangle shows the possible area (becomes larger),
        after you have release the left mouse button.

        Y-axis labels are set according to the channel_unit, presented in the last curve for the subplot.
        For own axis labeling, use as channel_unit for the last channel your label, e.g. r"$i_T$ in A".
        Note, that the r before the string gives the command to accept LaTeX formulas, like $$.
        The parameters has to fullfill conditions:
        Minimal shift step in x-direction is the minimal difference of 2 points of all provided channels

        :param channels: list of datasets
        :type channels: list[Channel]
        :param shiftstep_x: shift step in x-direction
        :type shiftstep_x: float
        :param shiftstep_y: shift step in y-direction
        :type shiftstep_y: float
        :param displayrange_x: Display range limits in x-direction (min_x, max_x)
        :type displayrange_x: tuple of float
        :param displayrange_y: Display range limits in y-direction (min_y, max_y)
        :type displayrange_y: tuple of float

        :return: List of x and y-shifts per channel
        :rtype: list[float]
        """
        # Create plot and clear lists
        ScopeChShift.shiftfig, ScopeChShift.zoom_ax = plt.subplots()
        ScopeChShift.channelplotlist = list()
        ScopeChShift.shiftlist = list()
        # Actual display range (x and y)
        act_displayrange_x = [0.0, 0.0]
        act_displayrange_y = [0.0, 0.0]

        # Read channel data for the plot
        for channel in channels:
            cur_channelplot, = ScopeChShift.zoom_ax.plot(channel.time, channel.data, label=channel.label, color=channel.color)
            ScopeChShift.channelplotlist.append(cur_channelplot)
            ScopeChShift.shiftlist.append([0, 0])

        # Init minimum and maximum values of the display window if required
        ScopeChShift.display_min_x, ScopeChShift.display_max_x = ScopeChShift.zoom_ax.get_xlim()

        # Check, if displayrange x need to initialize
        if displayrange_x[0] == displayrange_x[1]:
            act_displayrange_x[0] = ScopeChShift.display_min_x
            act_displayrange_x[1] = ScopeChShift.display_max_x
        else:
            # Overtake values as display range
            act_displayrange_x[0] = displayrange_x[0]
            act_displayrange_x[1] = displayrange_x[1]
            # Overtake values as new display range limits, if the range becomes higher
            if act_displayrange_x[0] < ScopeChShift.display_min_x:
                ScopeChShift.display_min_x = act_displayrange_x[0]
            if act_displayrange_x[1] > ScopeChShift.display_max_x:
                ScopeChShift.display_max_x = act_displayrange_x[1]

        # Set display range in x-direction
        ScopeChShift.zoom_ax.set_xlim(act_displayrange_x[0], act_displayrange_x[1])

        # Init minimum and maximum values of the display window if required
        ScopeChShift.display_min_y, ScopeChShift.display_max_y = ScopeChShift.zoom_ax.get_ylim()

        # Check, if displayrange y need to initialize
        if displayrange_y[0] == displayrange_y[1]:
            act_displayrange_y[0] = ScopeChShift.display_min_y
            act_displayrange_y[1] = ScopeChShift.display_max_y
        else:
            # Overtake values as display range
            act_displayrange_y[0] = displayrange_y[0]
            act_displayrange_y[1] = displayrange_y[1]
            # Overtake values as new display range limits, if the range becomes higher
            if act_displayrange_y[0] < ScopeChShift.display_min_y:
                ScopeChShift.display_min_y = act_displayrange_y[0]
            if act_displayrange_y[1] > ScopeChShift.display_max_y:
                ScopeChShift.display_max_y = act_displayrange_y[1]

        # Set display range in y-direction
        ScopeChShift.zoom_ax.set_ylim(act_displayrange_y[0], act_displayrange_y[1])

        # Define minimum zoom window as min_shiftstep_[xy]*5
        ScopeChShift.zoom_delta_y = ScopeChShift.min_shiftstep_y * 5
        ScopeChShift.zoom_delta_x = ScopeChShift.min_shiftstep_x * 5

        # Reset  channel index
        ScopeChShift.chn_index = 0
        # Set shift direction 0=x, 1=y
        ScopeChShift.shift_dir = 1
        # Set slider value
        ScopeChShift.last_val = 0
        # Hint for the user
        ScopeChShift.zoom_ax.set_title("Move the slider to move the selected channel curve")
        plt.subplots_adjust(bottom=0.45)

        # Overtake shift step parameter
        ScopeChShift.shiftstep_x = shiftstep_x
        ScopeChShift.shiftstep_y = shiftstep_y

        # -- Create the widgets

        # Button for selection of shift direction (x or y)
        button_ax = plt.axes([0.1, 0.15, 0.3, 0.075])  # Position and size of button
        ScopeChShift.dir_shift_button = Button(button_ax, 'Shift y')
        ScopeChShift.dir_shift_button.on_clicked(ScopeChShift.__toggle_xy_plot)  # Link to callback

        # Button for selection of graph
        button_ax = plt.axes([0.1, 0.25, 0.3, 0.075])  # Position and size of button
        ScopeChShift.chn_sel_button = Button(button_ax, 'Sel. Plot')
        ScopeChShift.chn_sel_button.on_clicked(ScopeChShift.__next_channel)  # links button to function

        # Button for reset slider position
        button_ax = plt.axes([0.85, 0.1, 0.14, 0.1])  # Position and size of button
        ScopeChShift.shsl_reset_button = Button(button_ax, 'Re-calibrate\nslider')
        ScopeChShift.shsl_reset_button.on_clicked(ScopeChShift.__reset_slider)  # links button to function

        # Textbox for step size
        text_box_ax = plt.axes([0.6, 0.15, 0.2, 0.05])
        ScopeChShift.shift_text_box = TextBox(text_box_ax, 'Shiftstep:', initial=str(ScopeChShift.shiftstep_y))
        ScopeChShift.shift_text_box.on_submit(ScopeChShift.__submit)

        # Selected dataset (Show label)
        labeltext = ScopeChShift.channelplotlist[0].get_label()
        # Check, if label text is not set
        if labeltext is None:
            labeltext = "●"
        else:
            labeltext = "● " + labeltext
        # Set labeltext in figure
        ScopeChShift.selplotlabel = ScopeChShift.shiftfig.text(0.6, 0.3, labeltext, ha='left', va='top', fontsize=12)
        ScopeChShift.selplotlabel.set_color(ScopeChShift.channelplotlist[ScopeChShift.chn_index].get_color())

        # Register eventhandler
        ScopeChShift.shiftfig.canvas.mpl_connect('button_press_event', ScopeChShift.__on_press)
        ScopeChShift.shiftfig.canvas.mpl_connect('motion_notify_event', ScopeChShift.__on_motion)
        ScopeChShift.shiftfig.canvas.mpl_connect('button_release_event', ScopeChShift.__on_release)

        # Slider for shift method
        slider_ax = plt.axes([0.2, 0.1, 0.55, 0.03], facecolor="lightgoldenrodyellow")  # Position and size of sliders
        ScopeChShift.shift_slider = Slider(slider_ax, 'Shift', -10.0, 10.0, valinit=0)  # Slider values
        # Link slider with callback
        ScopeChShift.shift_slider.on_changed(ScopeChShift.__shiftchannel)
        # Get the figure box object for check purpose
        ScopeChShift.shiftfigbox = ScopeChShift.zoom_ax.get_window_extent()

        # Log flow control
        logging.debug(f"{class_modulename} :\nDisplay range x:{ScopeChShift.display_min_x},{ScopeChShift.display_max_x} "
                      f"Display range y:{ScopeChShift.display_min_y},{ScopeChShift.display_max_y}\n"
                      f"Shift step x min,act,max:{ScopeChShift.min_shiftstep_x}, {shiftstep_x},{ScopeChShift.max_shiftstep_x}\n"
                      f"Shift step y min,act,max:{ScopeChShift.min_shiftstep_y}, {shiftstep_y},{ScopeChShift.max_shiftstep_y}\n"
                      f"Zoom area x,y:{ScopeChShift.zoom_delta_x}, {ScopeChShift.zoom_delta_y}")

        # Shows the figure
        plt.show()

        # Return the list of channel shifts
        return ScopeChShift.shiftlist

    @staticmethod
    def init_shiftstep_limits(shiftsteprange_x: Tuple[float, float], shiftsteprange_y: Tuple[float, float]):
        """
        Initialize the minimal and maximal shift step.

        Examples:
        >>> import channelshift as chshift
        >>> chshift.init_shiftstep_limits((1e-6,5e-5),(1,20))
        Set the limit of the shift step

        :param shiftsteprange_x: Default shift step, in calsLimits of shift steps in x-direction (min_x, max_x)
        :type shiftsteprange_x: tuple of float    :param shiftsteprange_x: Limits of shift steps in x-direction (min_x, max_x)
        :type shiftsteprange_x: tuple of float
        :param shiftsteprange_y: Limits of shift steps in y-direction (min_y, max_y)
        :type shiftsteprange_y: tuple of float
        """
        # Overtake shiftsteprange in x-direction
        ScopeChShift.min_shiftstep_x = shiftsteprange_x[0]
        ScopeChShift.max_shiftstep_x = shiftsteprange_x[1]

        # Overtake shiftsteprange in y-direction
        ScopeChShift.min_shiftstep_y = shiftsteprange_y[0]
        ScopeChShift.max_shiftstep_y = shiftsteprange_y[1]

    ##############################################################################
    # Callback methods for interactive shifting of plots
    ##############################################################################

    @staticmethod
    def __next_channel(event: any):
        """
        Select the next channel.

        This callback method is private (only for internal usage)
        Called by matplotlib-Event and assigned to a button.

        :param event: Container with some information: x and y position in pixel, x and y position in data scale,...
        :type event: any
        """
        # Increment the index of the selected channel
        ScopeChShift.chn_index = ScopeChShift.chn_index + 1
        # Check on overflow
        if ScopeChShift.chn_index >= len(ScopeChShift.channelplotlist):
            ScopeChShift.chn_index = 0
        # Display the label
        labeltext = ScopeChShift.channelplotlist[ScopeChShift.chn_index].get_label()
        # Check, if label text is not set
        if labeltext is None:
            labeltext = "●"
        else:
            labeltext = "● " + labeltext

        ScopeChShift.selplotlabel.set_text(labeltext)
        ScopeChShift.selplotlabel.set_color(ScopeChShift.channelplotlist[ScopeChShift.chn_index].get_color())

        ScopeChShift.shiftfig.canvas.draw_idle()  # update view
        ScopeChShift.last_val = 0
        ScopeChShift.shift_slider.set_val(0)

        # Log flow control
        logging.debug(f"{class_modulename} :Channel number {ScopeChShift.chn_index} is selected.")

    @staticmethod
    def __toggle_xy_plot(event):
        """
        Toggle the shift direction.

        This callback method is private (only for internal usage)
        Called by matplotlib-Event and assigned to a button.

        :param event: Container with some information: x and y position in pixel, x and y position in data scale,...
        :type event: any
        """
        # Check shift direction
        if ScopeChShift.shift_dir == 1:
            # Toggle to x-direction
            ScopeChShift.shift_dir = 0
            # Set button text
            ScopeChShift.dir_shift_button.label.set_text("Shift x")
            ScopeChShift.shift_text_box.set_val(str(ScopeChShift.shiftstep_x))
        else:
            # Toggle to y-direction
            ScopeChShift.shift_dir = 1
            # Set button text
            ScopeChShift.dir_shift_button.label.set_text("Shift y")
            ScopeChShift.shift_text_box.set_val(str(ScopeChShift.shiftstep_y))
        # Reset the values
        ScopeChShift.last_val = 0
        ScopeChShift.shift_slider.set_val(0)
        # Update the plot
        ScopeChShift.shiftfig.canvas.draw_idle()

        # Log flow control
        logging.debug(f"{class_modulename} :Shift direction is toggled to {ScopeChShift.shift_dir} (0=x, 1=y).")

    @staticmethod
    def __submit(text: str):
        """
        Change the shift-step size.

        This callback method is private (only for internal usage)
        Called by matplotlib-Event and assigned to an update of text.

        :param text: Updated text
        :type text: string
        """
        # Check if text is a number
        try:
            fvalue = float(text)
        except ValueError:
            fvalue = 0
        # Check shift direction
        if ScopeChShift.shift_dir == 1:
            ScopeChShift.shiftstep_y = fvalue
            if ScopeChShift.shiftstep_y > ScopeChShift.max_shiftstep_y:
                # Shift step x too high
                logging.info(
                    f"{class_modulename} :Shift step y {ScopeChShift.shiftstep_y}"
                    f" too high and set to {ScopeChShift.max_shiftstep_y}")
                # Correct the value
                ScopeChShift.shiftstep_y = ScopeChShift.max_shiftstep_y
                # Log flow control
                logging.info(f"{class_modulename} :Shift step y too small.{ScopeChShift.shiftstep_y}")
            elif ScopeChShift.shiftstep_y < ScopeChShift.min_shiftstep_y:
                # Shift step x too small
                logging.info(
                    f"{class_modulename} :Shift step y {ScopeChShift.shiftstep_y}"
                    f" too small and set to {ScopeChShift.min_shiftstep_y}")
                # Correct the value
                ScopeChShift.shiftstep_y = ScopeChShift.min_shiftstep_y

            ScopeChShift.shift_text_box.set_val(f"{ScopeChShift.shiftstep_y:.7g}")
            # Log flow control
            logging.debug(f"{class_modulename} :Shift step y = {ScopeChShift.shiftstep_y}")

        else:
            ScopeChShift.shiftstep_x = fvalue
            if ScopeChShift.shiftstep_x > ScopeChShift.max_shiftstep_x:
                # Shift step x too high
                logging.info(
                    f"{class_modulename} :Shift step x {ScopeChShift.shiftstep_y} too high and set to {ScopeChShift.max_shiftstep_x}")
                # Correct the value
                ScopeChShift.shiftstep_x = ScopeChShift.max_shiftstep_x
            elif ScopeChShift.shiftstep_x < ScopeChShift.min_shiftstep_x:
                # Shift step x too small
                logging.info(
                    f"{class_modulename} :Shift step x {ScopeChShift.shiftstep_x} too small and set to {ScopeChShift.min_shiftstep_x}")
                # Correct the value
                ScopeChShift.shiftstep_x = ScopeChShift.min_shiftstep_x

            # Update the format if necessary
            ScopeChShift.shift_text_box.set_val(f"{ScopeChShift.shiftstep_x:.7g}")

        # Log flow control
        logging.debug(
            f"{class_modulename}: Shift step x = {ScopeChShift.shiftstep_x},Shift step y = {ScopeChShift.shiftstep_y}")

        # Reset the values
        ScopeChShift.last_val = 0
        ScopeChShift.shift_slider.set_val(0)
        # Update the plot
        ScopeChShift.shiftfig.canvas.draw_idle()

        # Log flow control
        logging.debug(f"{class_modulename} :Text is overtaken {ScopeChShift.shift_dir} (0=x, 1=y).")

    @staticmethod
    def __shiftchannel(val: float):
        """
        Shift the channel by movement of the slider.

        This callback method is private (only for internal usage)
        Called by matplotlib-Event and assigned to the slider

        :param val:  slider position
        :type val: float
        """
        # Check shift direction
        if ScopeChShift.shift_dir == 1:
            delta = val - ScopeChShift.last_val
            dshift = (delta * ScopeChShift.shiftstep_y)
            ScopeChShift.shiftlist[ScopeChShift.chn_index][ScopeChShift.shift_dir] = (
                ScopeChShift.shiftlist[ScopeChShift.chn_index][ScopeChShift.shift_dir] + dshift
            )
            # Shift values in y-direction
            new_y = [value + dshift for value in ScopeChShift.channelplotlist[ScopeChShift.chn_index].get_ydata()]
            # Update dataset
            ScopeChShift.channelplotlist[ScopeChShift.chn_index].set_ydata(new_y)
        else:
            delta = val - ScopeChShift.last_val
            dshift = (delta * ScopeChShift.shiftstep_x)
            shift = ScopeChShift.shiftlist[ScopeChShift.chn_index][ScopeChShift.shift_dir] + dshift
            # Shift in x-direction
            new_x = [value + dshift for value in ScopeChShift.channelplotlist[ScopeChShift.chn_index].get_xdata()]
            # Update dataset
            ScopeChShift.channelplotlist[ScopeChShift.chn_index].set_xdata(new_x)

            # Overtake the shift
            ScopeChShift.shiftlist[ScopeChShift.chn_index][ScopeChShift.shift_dir] = shift
            # Update the plot
            ScopeChShift.shiftfig.canvas.draw_idle()

        # Store  slider value
        ScopeChShift.last_val = val

    @staticmethod
    def __reset_slider(event: any):
        """
        Recalibrate the slider position to zero without movement of the channel.

        This method is private (only for internal usage)
        Called by matplotlib-Event and assigned to a button

        :param event: Container with some information: x and y position in pixel, x and y position in data scale,...
        :type event: any
        """
        ScopeChShift.last_val = 0
        ScopeChShift.shift_slider.set_val(0)
        # Update the plot
        ScopeChShift.shiftfig.canvas.draw_idle()

        # Log flow control
        logging.debug(f"{class_modulename} :Slider reset was called.")

    # -- Zoomcallbacks ------------------------------------------------------------------------------------

    @staticmethod
    def __on_press(event: any):
        """
        Notification event if a mouse button is pressed.

        This callback method is private (only for internal usage)
        Called by matplotlib-Event on button press

        :param event: Container with some information: x and y position in pixel, x and y position in data scale,...
        :type event: any
        """
        # Check, if the button press was within the area
        if event.inaxes and ScopeChShift.shiftfigbox.contains(event.x, event.y):
            # Check, if the left button was pressed
            if event.button == 1:
                if ScopeChShift.zoom_state == ScopeChShift.Zoom_State.NoZoom:
                    # Overtake values
                    ScopeChShift.zoom_start_x, ScopeChShift.zoom_start_y = event.xdata, event.ydata
                    # Create rectangle
                    ScopeChShift.zoom_rect = patches.Rectangle((ScopeChShift.zoom_start_x, ScopeChShift.zoom_start_y), \
                                                               0, 0, linewidth=1, edgecolor='red', facecolor='none')
                    # Set Zoomstate
                    ScopeChShift.zoom_state = ScopeChShift.Zoom_State.ZoomSelect
                    # Draw rectangle
                    ScopeChShift.zoom_ax.add_patch(ScopeChShift.zoom_rect)
                    # Log flow control
                    logging.debug(f"Start zoom at x: {ScopeChShift.zoom_start_x} y:{ScopeChShift.zoom_start_y}.")
                    # Update plot
                    plt.draw()
                else:
                    # Log flow control
                    logging.debug(f"Button no. {event.button} is pressed when zoom state is {ScopeChShift.zoom_state}.")

            else:
                # Log flow control
                logging.debug(f"Button no. {event.button} is pressed inside the plot.")

        else:
            # Log flow control
            logging.debug("Button press outside of the plot.")

    @staticmethod
    def __on_motion(event: any):
        """
        Provide the current mouse position in case of mouse movement.

        This callback method is private (only for internal usage)
        Called by matplotlib-Event on button press

        :param event: Container with some information: x and y position in pixel, x and y position in data scale,...
        :type event: any
        """
        # Check Zoomstate
        if ScopeChShift.zoom_state == ScopeChShift.Zoom_State.ZoomSelect:
            # Check, if movement is within area and button still pressed
            if event.button == 1:
                if ScopeChShift.zoom_rect is not None and event.inaxes and ScopeChShift.shiftfigbox.contains(event.x, event.y):
                    # Overtake zoom end values
                    width = event.xdata - ScopeChShift.zoom_start_x
                    height = event.ydata - ScopeChShift.zoom_start_y
                    ScopeChShift.zoom_rect.set_width(width)
                    ScopeChShift.zoom_rect.set_height(height)
                    ScopeChShift.zoom_rect.set_xy((ScopeChShift.zoom_start_x, ScopeChShift.zoom_start_y))
                    # Update plot
                    plt.draw()
            else:
                # Remove rectangle and reset zoom state
                ScopeChShift.zoom_rect.remove()
                ScopeChShift.zoom_rect = None
                # Change Zoomstate to NoZoom
                ScopeChShift.zoom_state = ScopeChShift.Zoom_State.NoZoom
                # Update plot
                plt.draw()
                # Log flow control
                logging.debug(f"Right mouse button is pressed while zoom selection at: y={event.xdata} y={event.ydata}.")

    @staticmethod
    def __on_release(event: any):
        """
        Notify the mouse button release.

        This callback method is private (only for internal usage)
        Called by matplotlib-Event on button press

        :param event: Container with some information: x and y position in pixel, x and y position in data scale,...
        :type event: any
        """
        # Check, if the left button was released
        if event.button == 1:
            # Check Zoomstate
            if ScopeChShift.zoom_state == ScopeChShift.Zoom_State.ZoomSelect:
                # Check, if the button was pressed within the area
                if ScopeChShift.zoom_rect is not None and event.inaxes and ScopeChShift.shiftfigbox.contains(event.x, event.y):
                    # Define the area for zoom
                    ScopeChShift.zoom_end_x, ScopeChShift.zoom_end_y = event.xdata, event.ydata
                    # Sort data
                    if ScopeChShift.zoom_start_x > event.xdata:
                        ScopeChShift.zoom_end_x = ScopeChShift.zoom_start_x
                        ScopeChShift.zoom_start_x = event.xdata
                    if ScopeChShift.zoom_start_y > event.ydata:
                        ScopeChShift.zoom_end_y = ScopeChShift.zoom_start_y
                        ScopeChShift.zoom_start_y = event.ydata
                    # Check minimum zoom limits of x-axe
                    cur_zoom_delta = ScopeChShift.zoom_end_x - ScopeChShift.zoom_start_x
                    # Read  range
                    cur_start_x, cur_end_x = ScopeChShift.zoom_ax.get_xlim()
                    # Check if the minimum zoom is reached
                    if cur_end_x - cur_start_x <= ScopeChShift.zoom_delta_x:
                        # Overtake rect
                        ScopeChShift.zoom_end_x = cur_end_x
                        ScopeChShift.zoom_start_x = cur_start_x
                        # Log flow control
                        logging.debug("minimum zoom in x-direction is reached. Zooming in x-direction is denied.")
                    elif cur_zoom_delta < ScopeChShift.zoom_delta_x:
                        logging.debug(f"Selected range in x-direction {ScopeChShift.zoom_start_x} to {event.xdata} is too small.")
                        ScopeChShift.zoom_end_x = ScopeChShift.zoom_end_x + (ScopeChShift.zoom_delta_x - cur_zoom_delta) / 2
                        # Check, if maximum limit is exceed
                        if ScopeChShift.zoom_end_x > ScopeChShift.display_max_x:
                            ScopeChShift.zoom_end_x = ScopeChShift.display_max_x
                            ScopeChShift.zoom_start_x = ScopeChShift.zoom_end_x - ScopeChShift.zoom_delta_x
                        else:  # Calculate minimum window value
                            ScopeChShift.zoom_start_x = ScopeChShift.zoom_start_x - (ScopeChShift.zoom_delta_x - cur_zoom_delta) / 2
                        # Check, if minimum limit is exceed
                        if ScopeChShift.zoom_start_x < ScopeChShift.display_min_x:
                            ScopeChShift.zoom_start_x = ScopeChShift.display_mix_x
                            ScopeChShift.zoom_end_x = ScopeChShift.zoom_end_x + ScopeChShift.zoom_delta_x
                        # Log flow control
                        logging.debug(f"Range in x-direction is corrected to {ScopeChShift.zoom_start_x} to {ScopeChShift.zoom_end_x}.")
                    # Check minimum zoom limits of y-axe
                    cur_zoom_delta = ScopeChShift.zoom_end_y - ScopeChShift.zoom_start_y
                    # Read  range
                    cur_start_y, cur_end_y = ScopeChShift.zoom_ax.get_ylim()
                    # Check if the minimum zoom is reached
                    if cur_end_y - cur_start_y <= ScopeChShift.zoom_delta_y:
                        # Overtake rect
                        ScopeChShift.zoom_end_y = cur_end_y
                        ScopeChShift.zoom_start_y = cur_start_y
                        # Log flow control
                        logging.debug("minimum zoom in y-direction is reached. Zooming in y-direction is denied.")
                    elif cur_zoom_delta < ScopeChShift.zoom_delta_y:
                        logging.debug(f"Selected range in y-direction {ScopeChShift.zoom_start_y} to {event.ydata} is too small.")
                        ScopeChShift.zoom_end_y = ScopeChShift.zoom_end_y + (ScopeChShift.zoom_delta_y - cur_zoom_delta) / 2
                        # Check, if maximum limit is exceed
                        if ScopeChShift.zoom_end_y > ScopeChShift.display_max_y:
                            ScopeChShift.zoom_end_y = ScopeChShift.display_max_y
                            ScopeChShift.zoom_start_y = ScopeChShift.zoom_end_y - ScopeChShift.zoom_delta_y
                        else:  # Calculate minimum window value
                            ScopeChShift.zoom_start_y = (
                                ScopeChShift.zoom_start_y - (ScopeChShift.zoom_delta_y - cur_zoom_delta) / 2)
                        # Check, if minimum limit is exceed
                        if ScopeChShift.zoom_start_y < ScopeChShift.display_min_y:
                            ScopeChShift.zoom_start_y = ScopeChShift.display_mix_y
                            ScopeChShift.zoom_end_y = ScopeChShift.zoom_end_y + ScopeChShift.zoom_delta_y
                            # Log flow control
                            logging.debug(f"Range in y-direction is corrected to {ScopeChShift.zoom_start_y} to {ScopeChShift.zoom_end_y}.")
                    width = ScopeChShift.zoom_end_x - ScopeChShift.zoom_start_x
                    height = ScopeChShift.zoom_end_y - ScopeChShift.zoom_start_y
                    ScopeChShift.zoom_rect.set_width(width)
                    ScopeChShift.zoom_rect.set_height(height)
                    ScopeChShift.zoom_rect.set_xy((ScopeChShift.zoom_start_x, ScopeChShift.zoom_start_y))
                    # Update zoom state
                    ScopeChShift.zoom_state = ScopeChShift.Zoom_State.ZoomConfirm
                    # Update plot
                    plt.draw()
                elif ScopeChShift.zoom_rect is not None:
                    # Remove rectangle and reset zoom state
                    ScopeChShift.zoom_rect.remove()
                    ScopeChShift.zoom_rect = None
                    ScopeChShift.zoom_state = ScopeChShift.Zoom_State.NoZoom
                    # Update plot
                    plt.draw()
                    # Log flow control
                    logging.debug("Rectangle removed and reset zoom state to 'NoZoom'.")
            # Check if Zoomstate is ZoomConfirm
            elif ScopeChShift.zoom_state == ScopeChShift.Zoom_State.ZoomConfirm:
                # Check, if the button was pressed within the area
                if ScopeChShift.zoom_rect is not None and event.inaxes and ScopeChShift.shiftfigbox.contains(event.x, event.y):
                    if (event.xdata > ScopeChShift.zoom_start_x) and (event.ydata > ScopeChShift.zoom_start_y) and \
                       (event.xdata < ScopeChShift.zoom_end_x) and (event.ydata < ScopeChShift.zoom_end_y):
                        # Zoom into area
                        ScopeChShift.zoom_ax.set_xlim(min(ScopeChShift.zoom_start_x, ScopeChShift.zoom_end_x), \
                                                      max(ScopeChShift.zoom_start_x, ScopeChShift.zoom_end_x))
                        ScopeChShift.zoom_ax.set_ylim(min(ScopeChShift.zoom_start_y, ScopeChShift.zoom_end_y), \
                                                      max(ScopeChShift.zoom_start_y, ScopeChShift.zoom_end_y))
                # Remove rectangle and reset zoom state
                ScopeChShift.zoom_rect.remove()
                ScopeChShift.zoom_rect = None
                ScopeChShift.zoom_state = ScopeChShift.Zoom_State.NoZoom
                # Update plot
                plt.draw()
                # Log flow control
                logging.debug(f"Confirmed with left mouse button release at x,y: {event.xdata},{event.ydata}.")
        # Check if left button was released and Zoomstate was NoZoom and it was within area
        elif (event.button == 3 and event.inaxes and ScopeChShift.shiftfigbox.contains(event.x, event.y) and \
              ScopeChShift.zoom_state is not ScopeChShift.Zoom_State.ZoomSelect):
            # Check, if zoom state is conform
            if ScopeChShift.zoom_state == ScopeChShift.Zoom_State.ZoomConfirm:
                # Remove rectangle and reset zoom state
                ScopeChShift.zoom_rect.remove()
                ScopeChShift.zoom_rect = None
                ScopeChShift.zoom_state = ScopeChShift.Zoom_State.NoZoom
                # Log flow control
                logging.debug(f"Zoom range is not confirmed with right mouse button at: y={event.xdata} y={event.ydata}.")
            else:
                # Zoom out: Get  zoom
                cur_y_start, cur_y_end = ScopeChShift.zoom_ax.get_ylim()
                cur_x_start, cur_x_end = ScopeChShift.zoom_ax.get_xlim()
                # Calculate factor 2 for y dimension
                delta_half = (cur_y_end - cur_y_start) / 2
                cur_y_start = cur_y_start - delta_half
                cur_y_end = cur_y_end + delta_half
                # Check maximal limits
                if cur_y_start < ScopeChShift.display_min_y:
                    cur_y_start = ScopeChShift.display_min_y
                if cur_y_end > ScopeChShift.display_max_y:
                    cur_y_end = ScopeChShift.display_max_y
                # Calculate factor 2 for x dimension
                delta_half = (cur_x_end - cur_x_start) / 2
                cur_x_start = cur_x_start - delta_half
                cur_x_end = cur_x_end + delta_half
                # Check maximal limits
                if cur_x_start < ScopeChShift.display_min_x:
                    cur_x_start = ScopeChShift.display_min_x
                if cur_x_end > ScopeChShift.display_max_x:
                    cur_x_end = ScopeChShift.display_max_x
                # Zoom out
                ScopeChShift.zoom_ax.set_xlim(cur_x_start, cur_x_end)
                ScopeChShift.zoom_ax.set_ylim(cur_y_start, cur_y_end)
                # Log flow control
                logging.debug(
                    f"Zoom out to x-range: {cur_x_start} to {cur_x_end} and y-range:{cur_y_start} to {cur_y_end}.")
            # Update plot
            plt.draw()
            logging.debug(f"Confirmed with right mouse button release at x,y: {event.xdata},{event.ydata}.")
