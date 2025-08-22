## Project Overview and Scientific Purpose

TransStop is a scientific project focused on predicting the efficacy of translational readthrough at premature termination codons (PTCs) in the human genome. It leverages a transformer-based deep learning model trained on large-scale experimental data to evaluate how different drugs can promote readthrough, which is a promising therapeutic strategy for genetic diseases caused by PTCs. The project integrates genomic sequence context and drug-specific effects to provide accurate, pan-drug predictions, supporting precision medicine, clinical trial design, and the development of personalized therapies.

## Project Workflow and Execution

This project consists of a sequential pipeline. To ensure reproducibility and simplify execution, you can run the entire workflow with a single command.

### Prerequisites

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Dichopsis/TransStop.git
    cd TransStop
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate TransStop
    ```

### How to Run the Pipeline

After setting up the environment and placing the data in the correct directories, you can run the entire pipeline using the main Python script:

```bash
python main.py
```