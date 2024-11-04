###################################################################################################
# This module set the logging level and provides the function setup_logging to configure
# the logging for each module in a centralized way. It is important to add following line to each
# module so that the module can participate to the centralized logging approach.
# -> from logconfig import setup_logging
# -> import logging
# -> setup_logging()
###################################################################################################
# date         Author  Description
# -----------  ------  ----------------------------------------------------------------------------
# 04.11.2024    ASA    Initial version
###################################################################################################


#-- Imports ---------------------------------------------------------------------------------------

import logging
import logging.config

#-- Code ------------------------------------------------------------------------------------------

# Setup the logging level for the module
def setup_logging(default_level=logging.INFO):
    # Set logging level for all imported modules
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')

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

    mp_logger = logging.getLogger('matplotlib')
    mp_logger.setLevel(logging.WARNING)

    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)

    # logging.basicConfig(
    #    level=default_level,
    #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )


