"""Classes and methods to process scope data (from real scopes or from simulation tools) like in a real scope.

This module set the logging level and provides the function setup_logging to configure
 the logging for each module in a centralized way. It is important to add following line to each
 module so that the module can participate to the centralized logging approach.
 -> from logconfig import setup_logging
 -> import logging
 -> setup_logging()
"""
###################################################################################################
# date         Author  Description
# -----------  ------  ----------------------------------------------------------------------------
# 04.11.2024    ASA    Initial version
###################################################################################################
# - Imports ---------------------------------------------------------------------------------------

import logging
import logging.config

# - Code ------------------------------------------------------------------------------------------

# Set up the logging level for the module
def setup_logging():
    """Set up the logging capability.

    This function call has to be added to each file just after the import block.
    With this file you can modify the logging level.
    The basic logging level corresponds to the logging level, you want to set.
    You can adjust (suppress logging) of signal files by setting a higher level for this particular file.
    """
    # Set logging level for all imported modules
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')

    # Set logging level for my modules
    # Logger for scope
    Lm_logger = logging.getLogger('scope')
    Lm_logger.setLevel(logging.DEBUG)
    # Logger for color
    ts_logger = logging.getLogger('color')
    ts_logger.setLevel(logging.DEBUG)
    # Logger for generalplotsettings
    tf_logger = logging.getLogger('generalplotsettings')
    tf_logger.setLevel(logging.DEBUG)
    # Logger for function
    Lm_logger = logging.getLogger('function')
    Lm_logger.setLevel(logging.DEBUG)

    ###############################################################################################
    # Do not change the lines below. These suppress logging information of external modules
    # 'matplotlib' and 'PIL', which are imported by pySignalScope
    ###############################################################################################

    mp_logger = logging.getLogger('matplotlib')
    mp_logger.setLevel(logging.WARNING)

    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)
