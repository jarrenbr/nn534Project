# Cpts 534 Project
GAN on [CASAS](https://duckduckgo.com).

## Setup
Python version: 3.9

Module install for mac:
```bash
pip3 install -r requirement.txt
```

## Task List
- [x] Proposal (Jocelyn, Jarren, Max) 
  - [x] Citation update in Latex Proposal
- [ ] Initialize project (Max, Jarren)
  - [x] Python dependencies (Jarren)
  - [x] Versioning (Max, Jarren)
  - [x] Guideline standards (Jarren, Max)
  - [ ] Create file directories (Jarren, Max)
- [ ] Preprocessing
  - [ ] File-saved preprocessing (Jarren)
  - [ ] Run-time preprocessing
    - [ ] Tensorflow Window Slider
  - [ ] Labeling (label to ordinal tracking)
- [ ] Core
  - [ ] Base GAN (Jocelyn)
    - [ ] GAN class
    - [ ] Discriminator Keras model
    - [ ] Generator Keras model
    - [ ] Plot losses
  - [ ] Stateful GAN (Jarren)
    - [ ] Port over WGAN
    - [ ] Add state
  - [ ] High-level GAN APIs (Jarren, Jocelyn)
- [ ] Post-processing (i.e. unnormalizing synthetic data)
- [ ] Analyses 
  - [ ] TSTR (Train on Synthetic, Test on Real) (Max)
    - [ ] Port over ML code
    - [ ] TRTR until synthetic available
  - [ ] Granger Causality
  - [ ] Supplement with qualitative analysis (Seaborn)
- [ ] Write Document
  - [ ] How traditional GAN works (Jocelyn)
  - [ ] Stateful GAN (Jarren)
  - [ ] TSTR (Max)
- [ ] Presentation


### Maybe
- [ ] Core
  - [ ] TCN State
- [ ] Analyses
  - [ ] Activity/Sensor joint probabilities given time step
    - [ ] Heatmap
    - [ ] Table with high-level stats
  - [ ] Feature distributions
    - [ ] Histogram
  - [ ] Violin plot of time differentials
  - [ ] Distribution analysis

## Standards

### Indentation
Indentation: tab

###Use asserts
If an assumption is made while coding, use an assert.

### Typing
Type hint and check as needed for clarity
```python
#too much
def is_div_3(var:int, verbose:bool=True) -> bool:
    if verbose:
        print("Type of var:", type(var))
    return not var % 3

#correct
def is_div_3(dividend, verbose=True):
  if verbose:
      print("Type of var:", type(dividend))
  return not dividend % 3

#correct
def preprocess(df:pd.DataFrame, scaler) -> np.ndarray:
    assert isinstance(df, pd.DataFrame)
    return df.to_numpy() * scaler
```

### Strings
Anything except string casting + concatting
```python
#bad
output = "lorem " + str(1) + " senpai"

#good
output = "lorem %d senpai" %(1,)
output = "Pi: {pi:.5f}".format(pi=3.1415965358979)
```

### Naming convention
| Type | Case |
| --- |------------------------------|
| Functions/classes | snake_case |
| Private vars | _camelCase |
| Global vars | UPPER_SNAKE_CASE |
| Private global vars | _UPPER_SNAKE_CASE |

