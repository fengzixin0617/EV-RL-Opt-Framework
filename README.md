# EV-RL-Opt-Framework
An Integrated Simulation-Optimisation Framework for Future Public Electric Vehicle Charger Deployment

Despite the rapid expansion of public electric vehicle (EV) charging infrastructure, charging networks remain far less mature than conventional refuelling systems. Existing studies on public charger optimisation typically overlook EV drivers’ post-deployment behaviour change. This paper proposes an integrated, bi-directional simulation–optimisation framework to optimise future public charger locations and evaluate drivers’ behavioural responses to newly deployed chargers. We first develop an agent-based reinforcement learning (RL) model to simulate driving and charging behaviours of EV drivers. Building on this simulation, we design a spatial optimisation model to allocate new public chargers. The RL model is then re-implemented to evaluate how EV drivers adjust their behaviours in response to infrastructure provision changes. Applied to a case study area of Great Britain, the results show that the network-wide State of Charge (SOC) increases in early stages of network expansion, but declines as the network continues to expand. The decline reflects a perceived ‘safer charging environment’, where drivers become more confident to tolerate lower SOC levels before recharging.  Consequently, part of the charging opportunity gains from infrastructure expansion is absorbed by behavioural adaptation rather than translating into higher SOC levels. Notably, SOC levels between 20% -30% is a critical range in which EV drivers most actively adjust their charging decisions in response to a denser charging network. Overall, the findings highlight the need for policymakers to account for the dynamic feedback between network expansion and adaptive driver behaviour to ensure effective infrastructure planning. 

<img width="272" height="330" alt="image" src="https://github.com/user-attachments/assets/e7ce5c64-0076-4c54-a768-3b36aa894dc8" />

The framework consists of an improved version of Flow Refuelling Location Model (FRLM) and an Reinforcement Learning (RL) model. 

The FRLM is now publicly available at PySAL package (https://pysal.org/libpysal/stable/). 
More details of the RL model can be found at https://journals.sagepub.com/doi/full/10.1177/23998083261455937 


