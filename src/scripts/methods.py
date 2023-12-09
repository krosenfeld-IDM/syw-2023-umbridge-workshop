import json
from pydantic import BaseModel
import pandas as pd
from typing import Any

__all__ = ["DataBrick"]

class DefaultColors(BaseModel):
    analytic: list = 3*[0.7]
    R0: str = "k"
    emod: str = "C0"

class DataBrick(BaseModel):
    """
    Class to turn EMOD results into a pandas dataframe
    """

    data_brick: dict
    param_dict: dict
    data: dict = {}
    varables: list = []

    def __init__(self, tpath):

        # import simulation results
        with open(tpath / 'data_brick.json') as fid01:
            data_brick = json.load(fid01)
        with open(tpath / 'param_dict.json') as fid01:
            param_dict = json.load(fid01)

        super().__init__(data_brick=data_brick, param_dict=param_dict)

        # create brick
        self._parse_brick()

    def _parse_brick(self):
        # go through brick

        # initialize empty dict with key
        self.data = dict(run_number=[])
        for k,v in self.data_brick.items():
            self.data["run_number"].append(int(k))
            for kv,vv in v.items():
                if kv not in self.data:
                    self.data[kv] = []
                self.data[kv].append(vv)

        # merge with sim variables
        self.data = pd.merge(pd.DataFrame(self.data), pd.DataFrame(self.param_dict["EXP_VARIABLE"]), on="run_number").to_dict()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)