# Decision Engine Algorithms 
Algorithms for chaos experiment selection. 
 
Used as part of bachelor's thesis.

## Document Structure
The directory contains the following relevant files:

- __figures__ directory: The experiment results 
- __algorithms.py__: Algorithm implementations, using strategy pattern 
- __architecture.py__: Architecture class
- __architecture_model.json__: The architecture model used for evaluation
- __chaosmock.py__: Mocking functions originally used for running chaos experiments through ChaosToolkit
- __engine.py__: Experiment setup and execution
- __experiment.json__: Example experiment for ChaosToolkit
- __journal.json__: Example output of ChaosToolkit 
- __kolgsmir.py__, __kolgsmir_hystrix.py__, __kolgsmir_service.py__: Kolmogorov-Smirnov tests on data 
- __separate_service_data.py__: Separation of service data for retrieving of mocking results

The __experiment__ directory is empty. It should contain the raw data used for mocking purposes. The data is available at [zeonodo](https://zenodo.org/record/3265806#.XRtMgy_8LOQ).'

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3265806.svg)](https://doi.org/10.5281/zenodo.3265806)
