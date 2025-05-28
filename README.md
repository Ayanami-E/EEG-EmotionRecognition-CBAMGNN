# EEG-EmotionRecognition-CBAMGNN
对于本人毕设的一点点小开源，github上相关领域的开源太少，虽然我这个数值非跨92也达不到sota，但是希望能给分配到这个领域的本科毕设人一点小小的魔改基础，别的不保证，论文里的数只要你安装好环境就能跑出来，不做半点假，建设中。目前未整理的代码就在未整理里，虽然有点抽象但是可以复现数值，开源版还在整理中。
📄 原始论文出处  
本项目为下述本科毕设论文的配套代码实现：
**Peizhou Huang (2025)**  
*Emotion recognition based on graph convolutional networks and EEG signals*  
Lappeenranta-Lahti University of Technology LUT  
永久链接：[https://urn.fi/URN:NBN:fi-fe2025050536018](https://urn.fi/URN:NBN:fi-fe2025050536018)

📌 备注  
因项目时间与精力限制，本文只实现并开源了 SEED 数据集中效果最好的 **差分熵（DE）特征** 相关部分，其他特征维度或多模态方向未再展开，属实略有“偷懒”😅，但在可复现性和基础框架完整性方面已尽力保证。

欢迎基于本框架进一步扩展，也欢迎提交 PR！

# EEG-EmotionRecognition-CBAMGNN

A small open-source release based on my undergraduate thesis.  
There are very few open-source EEG + GNN projects on GitHub, especially for emotion recognition. Although this model doesn't reach state-of-the-art and can’t cross the 92% mark in subject-independent settings, I hope it can still serve as a **modifiable foundation** for undergrad students assigned to this topic.  

I don’t guarantee it’s perfect — but I **do guarantee** that the reported results in the thesis are **fully reproducible** as long as the environment is correctly set up. No fake numbers here.  

The "unorganized" folder contains the original working code. It's a bit rough, but does reproduce the thesis results. A cleaner version is still under construction.

---

📄 **Thesis Reference**  
This repository is the companion code for the following undergraduate thesis:

**Peizhou Huang (2025)**  
*Emotion recognition based on graph convolutional networks and EEG signals*  
Lappeenranta-Lahti University of Technology (LUT), Finland  
Permanent link: [https://urn.fi/URN:NBN:fi-fe2025050536018](https://urn.fi/URN:NBN:fi-fe2025050536018)

---

📌 **Note**  
Due to time and workload constraints, I only implemented and open-sourced the **differential entropy (DE)** feature pipeline — the one that gave the best performance on the SEED dataset.  
Other modalities or feature types were not included — not gonna lie, I kind of took the lazy route there 😅.  

Still, I did my best to ensure **reproducibility** and provide a **clean foundation** for further extension.

Feel free to build upon this repo — and PRs are always welcome!

🚧 **Under Construction**
