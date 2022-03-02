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
- [x] Initialize project (Max, Jarren)
  - [x] Python dependencies (Jarren)
  - [x] Versioning (Max, Jarren)
  - [x] Guideline standards (Jarren, Max)
  - [x] Create file directories (Jarren, Max)
- [x] Preprocessing (Jarren)
  - [x] File-saved preprocessing (Jarren)
  - [x] Run-time preprocessing (Jarren)
    - [x] Time differentials (Jarren)
    - [x] Normalization (Jarren)
    - [x] Tensorflow Window Slider (Jarren)
      - [x] Flexibility for GAN or classifier feeds (Jarren)
  - [x] Labeling (label to ordinal tracking) (Jarren)
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
    - [x] Create simple CNN classifier (Jarren) 
    - [ ] Create better classifiers
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


###Testing
Developing unit (single function or idea) or integrated (aggregation) tests would be great.

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
output = "lorem %d senpai" % (1,)
output = "Pi: {pi:.5f}".format(pi=3.1415965358979)
```

### Naming convention
| Type                           | Case                        |
|--------------------------------|-----------------------------|
| Functions/classes              | snake_case                  |
| Scoped vars                    | camelCase                   |
| Global vars                    | UPPER_SNAKE_CASE            |
| Private vars/functions/classes | _(corresponding convention) |
| Keras model names | Title_Snake_Case |

##How Do I???
The windows generator class is in utils.common.py.

See networks.classifiers.binaryCasasClassifier.py for an example of getting the 
data and training a model.