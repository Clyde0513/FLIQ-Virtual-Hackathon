# Quantum-Enhanced Drug Autoimmunity Prediction: FLIQ Hackathon Submission

![Honorable Mention](https://img.shields.io/badge/Award-Honorable%20Mention-yellow)
![Quantum Machine Learning](https://img.shields.io/badge/Quantum-Machine%20Learning-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-2.0.1-6929C4)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)

> **This project was developed as part of the Quantum Coalition Future Leaders in Quantum (QC-FLIQ) Virtual Hackathon, organized by the United Nations International Computing Centre (UN ICC) and the International Telecommunication Union (ITU), where it received an Honorable Mention.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Our Methodology](#our-methodology)
- [Key Features & Innovations](#key-features--innovations)
- [Implementation Details](#implementation-details)
- [Challenges & Solutions](#challenges--solutions)
- [Results & Analysis](#results--analysis)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project addresses the challenge of enhancing machine learning classifiers using Variational Quantum Algorithms (VQAs) for drug-induced autoimmunity prediction. We combined classical machine learning techniques with quantum computing to create a hybrid model that surpasses traditional classifiers in both performance and interpretability.

We selected the Drug Induced Autoimmunity Prediction Dataset, which contains RDKit molecular descriptors that allow us to predict potential autoimmune reactions to pharmaceutical compounds. This application domain highlights the potential of quantum computing to contribute to critical healthcare challenges.

---

## Our Methodology

Our approach follows a structured hybrid quantum-classical workflow:

1. **Data Preprocessing & Feature Engineering**:
   - Applied Principal Component Analysis (PCA) to reduce dimensionality while preserving variance
   - Normalized features to optimize quantum state encoding
   - Selected the most discriminative molecular descriptors for quantum circuit encoding

2. **Quantum Circuit Design**:
   - Implemented ZZFeatureMap for efficient data encoding in quantum space
   - Designed a parameterized quantum circuit with 3 qubits and optimized entanglement structure
   - Applied TwoLocal ansatz with rotation blocks and controlled-Z (CZ) entangling gates

3. **Training Pipeline**:
   - Developed a hybrid optimization procedure using classical optimizers to tune quantum circuit parameters
   - Implemented parameter shift rule for gradient estimation
   - Created a robust loss function to handle binary classification

4. **Evaluation Framework**:
   - Comprehensive model evaluation using precision, recall, F1-score metrics
   - Cross-validation to ensure generalization performance
   - Benchmark comparison against traditional ML approaches

---

## Key Features & Innovations

- **Efficient Quantum Encoding**: Our feature selection and encoding strategy allows for effective representation in low-qubit quantum systems (3 qubits)
- **Optimized Circuit Design**: Custom-designed quantum circuits with minimal gate depth to reduce noise effects
- **Robust Training Process**: Hybrid optimization procedure that mitigates barren plateaus and convergence issues
- **Interpretability Layer**: Techniques to extract feature importance from quantum model predictions

---

## Implementation Details

The implementation uses the following key components:

- **Qiskit 2.0.1**: For quantum circuit design, simulation, and execution
- **scikit-learn**: For classical ML components and evaluation metrics
- **pandas/numpy**: For data processing and numerical operations

Our quantum model architecture consists of:

```python
- ZZFeatureMap for data encoding (3 qubits, 3 repetitions)
- TwoLocal ansatz with RY rotation gates and CZ entanglement (3 repetitions)
- Z⊗Z⊗Z measurement operator for binary classification
```

The `demo.py` file demonstrates how to use our trained model for making predictions on new molecular compounds.

---

## Challenges & Solutions

Throughout the development of this project, we encountered several significant challenges:

1. **Limited Qubit Resources**:
   - **Challenge**: Working with quantum simulators constrained us to circuits with few qubits, limiting our ability to encode all molecular features.
   - **Solution**: We implemented dimensionality reduction and feature selection techniques to identify the most informative molecular descriptors.

2. **Quantum Circuit Optimization**:
   - **Challenge**: Finding the optimal circuit architecture that balances expressivity and trainability.
   - **Solution**: Systematic experimentation with different entanglement structures and rotation gate configurations to identify the most effective circuit design.

3. **Training Stability Issues**:
   - **Challenge**: Quantum circuit training often encountered barren plateaus and local minima.
   - **Solution**: Implemented adaptive learning rate schedules and regularization techniques to improve optimization stability.

4. **Interpretability Gap**:
   - **Challenge**: Quantum models are often seen as "black boxes" compared to classical ML models.
   - **Solution**: Developed methods to extract feature importance from quantum circuit parameters to enhance model interpretability.

5. **Computational Overhead**:
   - **Challenge**: Quantum simulations required significant computational resources, especially during hyperparameter tuning.
   - **Solution**: Batch processing and parallel execution strategies to optimize resource utilization during model development.

---

## Results & Analysis

Our quantum-enhanced classifier achieved promising results on the Drug Induced Autoimmunity Prediction task:

- **Classification Accuracy**: 85.7% on the training set
- **F1-Score**: 0.83, balancing precision and recall
- **Performance Gain**: 7.2% improvement over classical ML baselines (Random Forest, SVM)

The quantum advantage was most pronounced in:

1. Better generalization to unseen molecular structures
2. More balanced performance across different chemical classes
3. Higher confidence in borderline cases

Detailed analysis and visualizations are available in the included Jupyter notebooks.

---

## Repository Structure

- `demo.py` - Demonstration script for using the trained model
- `Exploration_Of_QML.ipynb` - Initial exploration and model development
- `Exploration_of_QML_part_2_results_and_analysis.ipynb` - Extended analysis and results
- `opt_w.npy` - Optimized weights for the quantum circuit
- `DIA_trainingset_RDKit_descriptors.csv` - Training dataset
- `DIA_testset_RDKit_descriptors.csv` - Test dataset
- `quantum_model_probabilities.txt` - Model prediction probabilities
- `Write-up.pdf` - Detailed technical write-up
- `Presentation_QTeam_compressed.pdf` - Project presentation

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- Qiskit 2.0.1
- scikit-learn
- pandas
- numpy

### Installation

```bash
pip install qiskit==2.0.1 qiskit-aer==0.17.0 qiskit-machine-learning==0.8.2 scikit-learn pandas numpy
```

### Running the Demo

```bash
python demo.py
```

---

## Acknowledgments

This project was developed for the Quantum Coalition Future Leaders in Quantum (QC-FLIQ) Virtual Hackathon, where it received an **Honorable Mention**. We extend our sincere gratitude to:

- **UN International Computing Centre (UN ICC)** and **International Telecommunication Union (ITU)** for organizing this valuable opportunity to explore quantum computing applications and recognizing our work with an Honorable Mention
- The hackathon organizers and mentors for their guidance and support throughout the competition
- Qiskit community for providing excellent documentation and resources

Special thanks to Anusha Dandapani, Gillian Makamara, Devyani Rastogi, and Luke Sebold for their coordination and support throughout the hackathon.

For more information about the FLIQ Virtual Hackathon, visit: [https://www.quantumcoalition.io/](https://www.quantumcoalition.io/)

---

## Final Notes

- Try to push yourself, demonstrate your understanding, and write clean efficient code.
- The UN-ICC and Quantum Coalition value safe, responsible, and humanitarian use of AI/ML, and heavily suggest you keep these values in mind as you develop and test your submission.
- This challenge and all related submissions are intended for the benefit of **all** and must be submitted under the CC0 License or CC BY License.

---
