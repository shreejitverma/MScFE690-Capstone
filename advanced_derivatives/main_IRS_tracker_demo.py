import sys
import os
sys.path.insert(0, os.path.join((os.path.dirname(os.path.dirname(__file__))))) 
from fwd_swap_tracker import FwdIRSTrackers
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the Forward Interest Rate Swap (FwdIRSTrackers) tracker
irs_tracker = FwdIRSTrackers(currency='USD', tenor=10)

# Retrieve the tracker DataFrame
tracker_data = irs_tracker.get_tracker_data()

# Plot the tracker data
tracker_data.plot()
plt.show()
