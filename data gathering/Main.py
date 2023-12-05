from Collect_data import *
from Process_data import *


class MainClass:
    collected_data = Collect_data.get_data()
    caclulated_data = Process_data.main(collected_data)