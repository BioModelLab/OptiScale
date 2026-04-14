# OptiScale

OptiScale is a data-driven framework for selecting optimal spatial and temporal resolutions in biodiversity analyses.

## Requirements

- Python 3.14.3

### Required libraries

- pandas (3.0.1)
- numpy (2.4.2)
- matplotlib (3.10.8)
- scipy (1.17.0)
- pwlf (2.5.2)

Standard Python libraries (sys, pathlib, re, time) are also used.

---

## Installation

### Option 1 (Recommended): Conda environment

Make sure you have Anaconda or Miniconda installed.

Then run:

conda env create -f environment.yml
conda activate OptiScale

Note: If `conda` is not recognized, use Anaconda Prompt or ensure Conda is added to your system PATH.

---

### Option 2: pip installation

pip install -r requirements.txt

---

## Usage

Run from terminal:

python OptiScale.py <dataset.csv> xmin=<int> xmax=<int> ymin=<int> ymax=<int> n=<int> s=<below|above|none> TW=<x,y,...,z>

### Example

python OptiScale.py Data_Ebro_1951-2022.csv xmin=-2 xmax=0 ymin=40 ymax=43 n=10 s=below TW=1,4,8,12

---

## Input data

CSV file with the following columns:

- species (string)
- lon (float)
- lat (float)
- year (integer)

Each row represents an occurrence record.

---

## Outputs

The workflow generates multiple CSV and PNG files.

---

## Reproducibility

The full computational environment is provided in:

- environment.yml (recommended)
- requirements.txt

---

## Data availability

The datasets used in this study are available at Zenodo:

https://doi.org/10.5281/zenodo.19563492
