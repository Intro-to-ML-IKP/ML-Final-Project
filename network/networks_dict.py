from network.network_constructor import NetworkConstructor

import numpy as np


class NetworksDict(list["NetworkConstructor"]):
    def __call__(self):
        return self._sort_results()
    
    def _list_to_dict(self):
        nnDict = {}
        nn_results = np.array(self).T
        maes = nn_results[0]
        for count, mae in enumerate(maes):
            nnDict[mae] = self[count][1:]
        return nnDict

    def _sort_results(self) -> dict:
        """
        Sort the results by the mae (smallest to largest)
        """
        nnDict = self._list_to_dict()
        sorted_keys = sorted(nnDict.keys())
        sorted_results = {key: nnDict[key] for key in sorted_keys}
        return sorted_results
    
# This is how you get all the sorted results   
# sorted_result = NetworksDict()