# Event-Triggered State Estimation under Injection Attacks in Cyber-Physical Systems

## Description
This project implements and extends an **event-triggered MMSE Kalman filter** for **state estimation in cyber-physical systems (CPS)** under **injection attacks** and **communication constraints**. The work involves MATLAB simulation, algorithm analysis, and new ideas for improving performance and robustness, including clustering and neural network-based adaptive filtering.

## Features
- Event-triggered MMSE Kalman filter implementation in MATLAB  
- Simulation of cyber-physical systems under injection attacks  
- Reduced communication load via event-triggered transmission  
- Stability and performance analysis of the estimation algorithm  
- Novel ideas for research, including:
  - Adaptive event-triggering strategy  
  - K-means clustering for sensor attack detection  
  - Neural network-assisted adaptive Kalman filtering
  - 
## Proposed Enhancements
### 1. Adaptive Event Trigger
The triggering threshold dynamically adapts upon detecting an attack, preventing predictable transmission times.

### 2. Sensor Clustering (K-Means)
Sensor data are clustered and ranked based on attack likelihood.  
Clean clusters contribute more to data fusion, improving robustness against compromised sensors.

### 3. Neural Network–Assisted Kalman Filter
A neural network weights sensor innovations between 0 and 1 based on attack probability, adaptively adjusting the Kalman gain to maintain estimation accuracy.

## Results
- Accurate state estimation maintained despite attacks and packet losses  
- Reduced communication overhead via event-triggered policy  
- K-means clustering successfully identified attacked sensors  
- Neural network Kalman filter improved resilience against sensor compromise  

## Tools and Environment
- **MATLAB R2023a**
- **Neural Network Toolbox**
- **Signal Processing Toolbox**

## References
[1] Le Liu, Xudong Zhao, Bohui Wang, Yuanqing Wu & Wei Xing,  
*Event-triggered state estimation for cyber-physical systems with partially observed injection attacks.*

## License
This project is licensed under the MIT License.
