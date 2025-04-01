<div align="center">
  <h1>CitySeeker: A VLM Benchmark for Implicit Requests in Embodied Urban Navigation</h1>
<img src="assets/4.png" width="93%"/>  
  
</div>



## ✨ Highlights  
- **CitySeeker pioneers the first language-guided embodied urban navigation** with implicit requests in multi-city settings, incorporating real-world visual diversity, long-horizon planning, and unstructured instructions.  
- **VLM-based cognitive mapping framework** that translates implicit requests into multi-step plans through iterative observation-reasoning cycles.  
- **Extensive exploratory experiments** identifying key bottlenecks in VLMs’ spatial reasoning and providing actionable insights for spatial intelligence.  
<p align="center">
<img src="assets/1.png" width="50%"/>
</p>


## 📊 Dataset Overview  

**CitySeeker** is the first benchmark for **implicit request-based navigation** in dynamic urban environments, covering **8 city areas** with **41,128 nodes** and **6,440 instructions**.  

<p align="center">
<img src="assets/3.png" width="95%"/>
</p>

| **Dataset**   | **Instruction Type** | **#Instructions** | **Environment**       | **Source**               | **#City** | **#Nodes** | **Avg.Length** | **Avg.Token** |  
|---------------|----------------------|-------------------|-----------------------|--------------------------|-----------|------------|----------------|---------------|  
| Talk the Walk | explicit             | 786               | GridWorld             | 3D Rendering             | 1         | 100        | 6.8            | 34.5          |  
| Room-to-Room  | explicit             | 21,567            | Indoor                | Panoramas                | 1         | 10,800     | 6.0            | 29.0          |  
| Touchdown     | explicit             | 9,326             | Outdoor               | Street View              | 1         | 29,641     | 35.2           | 89.6          |  
| **CitySeeker**| **implicit**         | **6,440**         | **Outdoor+dynamic**   | **Street View + Map**    | **8**     | **41,128** | **18.3**       | **11.11**     |  

## 🗂️ Instruction Categories

| **Category**                     | **Description**                                      | **Example**                                                                 |
|----------------------------------|------------------------------------------------------|-----------------------------------------------------------------------------|
| **1. Basic POI Navigation**      | Request common urban facilities                      | "Please find the nearest hotel."                                           |
| **2. Brand-Specific Navigation** | Seek specific commercial brand locations             | "Please find the nearest Starbucks."                                       |
| **3. Transportation Hub Navigation** | Ask for public transit locations                 | "Please find the nearest bus station."                                     |
| **4. Latent POI Navigation**     | Indirectly observable targets requiring reasoning    | "Please find the nearest restroom."                                        |
| **5. Implicit Need Navigation**  | Express abstract human needs through context         | "I'm thirsty and would like something to drink. Help me find a nearby place." |
| **6. Inclusive Infrastructure Navigation** | Prioritize accessible infrastructure        | "Please find the nearest bank with an accessible entrance."                |
| **7. Semantic Preference Navigation** | Use descriptive language for subjective criteria | "Please find the nearest romantic restaurant."                             |

---

## 🏆 Benchmark Results  

**Key Metrics**:  
- **TCE (Task Completion Error)**, **TCP (Task Completion Precision)**, **TCC (Task Completion Consistency)**  
- **SPD (Shortest Path Distance)**, **nDTW (Normalized Dynamic Time Warping)**  

| **Model**          | **TCE**  | **TCP**  | **TCC**  | **SPD**  | **nDTW** |  
|--------------------|----------|----------|----------|----------|----------|  
| **GPT-4o**         | 2.39%    | 18.30%   | 6.84%    | 125.40   | 136.97   |  
| **Gemini-1.5-pro** | 1.91%    | 15.43%   | 7.48%    | 157.14   | 241.86   |  
| **InternVL2.5-38B**| 2.23%    | 18.14%   | 7.16%    | 136.55   | 169.18   |  

*(Full results in paper)*  

---

## 🔍 Key Innovations  

### 🔄 **Backtracking Mechanisms**  
Three strategies to mitigate error accumulation in long trajectories:  
1. **Basic Backtracking (B1)**: In this basic backtracking strategy, the agent reverts to the last "trusted" node when its internal confidence falls below a predefined threshold.  
2. **Step-Reward Backtracking (B2)**: This mechanism evaluates progress toward the goal by replacing subjective confidence scores with objective topological distance as the backtracking criterion.  
3. **Human-Guided Backtracking (B3)**: This strategy extends basic backtracking B1 with corrective guidance, providing minimal external "hint" that suggests the best action to take next after backtracking.  
<p align="center">
<img src="assets/b1.png" width="95%"/>
</p>


### 🗺️ **Enriching Spatial Cognition**  
- **Topology Cognitive Graph (C1)**: In this approach, the VLM is provided with a topological graph of recently traversed segments, which explicitly defines the connectivity between various locations.
- **Relative Position Maps (C2)**: In contrast, this approach emphasizes the spatial orientation of locations without directly specifying connectivity. 
<p align="center">
<img src="assets/c1.png" width="95%"/>
</p>


### 🧠 **Memory-Based Retrieval**  
1. **Topology-based (R1)**
2. **Spatial-based (R2)**
3. **Historical Trajectory Lookup (R3)**

Each component supports different aspects of memory retrieval across multiple reasoning iterations, mitigating error propagation during navigation tasks.
<p align="center">
<img src="assets/r1.png" width="95%"/>
</p>


## 📥 Data & Usage  
### Human Evaluation
We welcome community participation in human testing! Evaluation code and interfaces are available in the `website/` folder.
<p align="center">
<img src="assets/website.png" width="100%"/>
</p>


### Dataset Access
- **Raw image & topological graph data**: [Baidu Netdisk](https://pan.baidu.com/s/1HS3HL-uSUdxb69rmAOi0bw?pwd=2vjj)
  
- **Benchmark trajectories**: See `example/` folder for real navigation paths

## 🌐 License  
This project is open-sourced under **MIT License**.  
*Data for research use only. Commercial use requires permission.*  
