# Agent based disease transmission model

## Overview

In this benchmark we run [EMOD](https://docs.idmod.org/projects/emod-generic/en/latest/index.html), an agent based disease transmission model, looking at how $R_0$  and the correlation between individuals acquisition and transmission correlation can affect the ultimate attack fraction. There is 
no waning immunity.

## Authors
- [Katherine Rosenfeld](mailto:katherine.rosenfeld@gatesfoundation.org)
- [Kurt Frey](mailto:kurt.frey@gatesfoundation.org)

## Usage:

1. To build locally:
```
docker build -t workshop_model .
docker run -it -p 4243:4243 workshop_model
```

2. Then run pymc sampler:
```
python pymc-client.py  http://localhost:4243
```

## Properties

Model | Description
---|---
forward | Forward model
posterior | Posterior density

### posterior
Mapping | Dimensions | Description
---|---|---
input | [3] | [ $R_0$, variance of $R_0$, correlation between acquisition and transmission ]
output | [1] | Attack fraction

### posterior
Mapping | Dimensions | Description
---|---|---
input | [3] | [ $R_0$, variance of $R_0$, correlation between acquisition and transmission]
output | [1] | Log posterior density

Feature | Supported
---|---
Evaluate | True
Gradient | False
ApplyJacobian | False
ApplyHessian | False

Config | Type | Default | Description
---|---|---|---
None | | |

## Mount directories
Mount directory | Purpose
---|---
None |

## Description

The benchmark is fitting to an attack fraction of 40\% with a standard deviation of 10\%.