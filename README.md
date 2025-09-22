## UAVEdgeOpt


Hardware Testbed Validation (Two Jetson Nano Boards)
Objective. We validate the feasibility of our decentralized federated learning (DFL) pipeline on edge-class hardware using two NVIDIA Jetson Nano boards connected over Wi-Fi. While the large-scale edge-selection behavior is evaluated in simulation (25–100 UAVs), the 2-node testbed demonstrates on-device RSSI-aware weighting, consensus averaging, fragility triggers, and real communication cost.
A. Setup and Mapping to the Algorithm
Hardware: Two Jetson Nano boards (4 GB), microSD, on-board Wi-Fi (or USB Wi-Fi), same LAN.
Software: Ubuntu (JetPack), Python 3.8+, PyTorch (Jetson build), NumPy, gRPC or Python sockets.
Datasets/Models: MNIST (MLP), Fashion-MNIST (LeNet-style CNN), CIFAR-10 (ResNet-18). Each Nano holds a local partition (IID/Non-IID via Dirichlet α=0.1).
Graph realization (N=2): The physical network degenerates to one feasible link (A↔B). Thus, edge selection is trivial; however, RSSI-based edge weight, triggering, gossip averaging, and communication measurement remain meaningful.
Interpretation in Paper: In our hardware testbed with two Jetson Nano boards, the physical graph reduces to a single feasible link, so the edge-selection step is trivial. Nevertheless, we validate the core pipeline on embedded devices:
RSSI-based weight assignment using the real wireless link. (TO DO)
Decentralized consensus averaging of local model updates. (Completed )
Fragility detection via spectral triggers (λ₂/Dirichlet gap). (IN PROGRESS)
Measurement of real communication cost and latency. (IN PROGRESS)
Note : Large-scale edge selection (25–100 UAVs) is evaluated in simulation, while the Jetson deployment confirms feasibility and systems performance on UAV-class hardware.
 
 
 



B. On-Device Protocol (Per Round)
Each board performs:
Local training: run SGD for a few local epochs.
RSSI sampling → link weight: measure Wi-Fi RSSI (e.g., iw dev wlan0 link), convert to weight γ.
Fragility trigger: compute λ₂ = 2 * w_{12}. If below τ_gap → mark fragile.
Gossip consensus averaging: exchange weights with peers, then average using γ.
Metrics logging: test accuracy, bytes sent/received, per-round latency, power if available.
C. Measurement Plan and Reported Metrics
Rounds @90%: Number of communication rounds until accuracy ≥ 90%.
Comm. (MB): Bytes exchanged until convergence or fixed T.
Final Acc. @T: Accuracy at round T (e.g., T=200).
Latency (ms): Send+receive+average per round.
System stats (optional): CPU/GPU utilization, power, thermals.
D. Practical Details (Jetson Nano)
Power & clocks: use max performance mode (nvpmodel, jetson_clocks).
Wi-Fi RSSI sampling: iw dev wlan0 link.
Serialization: PyTorch state_dict serialized, log bytes length for Comm.(MB).
Compression (optional): int8 quantization or sparsification.
F. Threats to Validity & Limitations
Two-node degeneracy: edge selection trivial, so only feasibility validated.
Wireless variability: RSSI fluctuates; mitigated via repeated trials and averages.
Model scaling: large models may stress Nano memory; use small batch sizes and fixed power mode.

