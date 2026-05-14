# PG-C2F: Gaussian Prior-Guided Pseudo Mask Generation for Point-Supervised Infrared Small Target Detection



> **Official implementation of the paper:**  
> *Gaussian Prior-Guided Pseudo Mask Generation for Point-Supervised Infrared Small Target Detection*  
> **Accepted by IEEE Signal Processing Letters (SPL)**  

This repository provides the source code for our **PG-C2F** (Prior-Guided Coarse-to-Fine) framework, which generates high-quality pixel‑level pseudo masks from only single‑point annotations for infrared small target detection (IRSTD).

---

## 📖Abstract

While point supervision reduces the annotation burden for Infrared Small Target Detection (IRSTD), existing methods often overlook the intrinsic physics of thermal imaging. To address this, we propose a Gaussian Prior-Guided Coarse-to-Fine (PG-C2F) pseudo mask generation framework, designed to derive high-fidelity pixel-level masks from singlepoint labels. Specifically, an Adaptive Gaussian Bounding Box Generation (AGBG) module is proposed, which utilizes twodimensional Gaussian fitting to adaptively estimate target scale. Subsequently, employing the AGBG as a spatial constraint, a Locally Constrained Random Walker (LCRW) method is introduced to precisely delineate the target mask using local gradient information. Experiments on two benchmark datasets demonstrate that PG-C2F significantly outperforms state-of-theart weakly supervised methods. Remarkably, utilizing solely point labels, our approach achieves approximately 91.78% of the performance of fully supervised approaches.

