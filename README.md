# CAE
CAE DCAE SDCAE implements

# Requirement 
Keras >= 2.0.7 Tensorflow >= 1.2 (backend)

# Data
UC Irvine abalone dataset
The data is in '/data/rawdata'

# Preprocess
```bash
python preprocess/preprocess_data.py
```
## Training use CAE
```bash
python AE/CAE.py
```
## Training use DCAE
```bash
python AE/DCAE.py
```
## Training use SDCAE
```bash
python AE/SDCAE.py
```
