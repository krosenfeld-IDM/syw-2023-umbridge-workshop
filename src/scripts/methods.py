import json
from pydantic import BaseModel
import pandas as pd
from typing import Any

__all__ = ["DataBrick"]

class DefaultColors(BaseModel):
    analytic: list = 3*[0.7]
    R0: str = "k"
    emod: str = "C0"

class CalibBrick(BaseModel):
    param_calib: dict
    param_dict: dict
    data_brick: dict
    data: dict = {}

    def __init__(self, tpath=None, **data):
            
        if tpath is not None:
            # import simulation results
            with open(tpath / 'param_dict.json') as fid01:
                param_calib = json.load(fid01)
            with open(tpath / 'param_dict_iters.json') as fid01:
                param_dict = json.load(fid01)
            with open(tpath / 'data_brick_iters.json') as fid01:
                data_brick = json.load(fid01)

            super().__init__(param_calib=param_calib, param_dict=param_dict, data_brick=data_brick)
        else:
            super().__init__(**data)

        self._parse_brick()

    def _parse_brick(self):
        # number of calibration iterations
        num_iter = self.param_calib["NUM_ITER"]
        
        data = pd.DataFrame()
        for it in range(num_iter):
            # format of iteration number stored in the dictionaries
            # https://github.com/InstituteforDiseaseModeling/EMOD-Generic-Scripts/blob/main/model_demographics01/experiment_demog_UK01_calib/make06_pool_brick.py
            fmt = "{:02d}"
            # construct data brick
            db = DataBrick(data_brick=self.data_brick[fmt.format(it)], param_dict=self.param_dict[fmt.format(it)])            
            db_df = db.to_df()
            # add iteration 
            db_df["iteration"] = it
            # concatenate
            data = pd.concat([data, db_df], ignore_index=True)

        # reset index
        data = data.reset_index(drop=True)

        self.data = data.to_dict()

    def to_df(self) -> pd.DataFrame:
        # the data is a dict of dicts, so we 
        return pd.DataFrame(self.data)


class DataBrick(BaseModel):
    """
    Class to turn EMOD results into a pandas dataframe
    """

    data_brick: dict
    param_dict: dict
    data: dict = {}
    varables: list = []

    def __init__(self, tpath=None, **data):

        if tpath is not None:
            # import simulation results
            with open(tpath / 'data_brick.json') as fid01:
                data_brick = json.load(fid01)
            with open(tpath / 'param_dict.json') as fid01:
                param_dict = json.load(fid01)

            super().__init__(data_brick=data_brick, param_dict=param_dict)
        else:
            super().__init__(**data)

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