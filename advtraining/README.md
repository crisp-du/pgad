
# TCPS: Anomaly Detection Using Adversarially Trained Models

This repository contains a single script to evaluate time-series forecasting models under various perturbations. The script processes time-series data, applies a trained forecasting model, computes anomaly scores, and outputs the results with anomaly labels.

This code is supplementary material for the following paper:

**Srinidhi Madabhushi** and **Rinku Dewri**  
*Mitigating Over-Generalization in Anomalous Power Consumption Detection using Adversarial Training*  
**ACM Transactions on Cyber-Physical Systems**, April 2025.

---

## Requirements

Install dependencies:

```bash
pip install tensorflow==2.12 pandas==2.0.3 numpy==1.23.5
````

---

## Usage

```bash
python main.py <model_name> <attack_name> <dataset_name>
```

* `<model_name>`: One of `mlp`, `lstm`, or `cnn`
* `<attack_name>`: One of `original`, `fgsm`, `bim`, `pgd`, `random`
* `<dataset_name>`: Name of the CSV file (without `.csv`) located in the `dataset/` folder

### Example

```bash
python script.py cnn fgsm sample_consumption
```

---

## Output

The script generates a file named `<dataset_name>_modified.csv` containing:

* `pred`: Model's forecasted values
* `scores`: Anomaly scores
* `anomalies`: Boolean labels where `True` indicates a detected anomaly

---

## Notes

* Models expect a fixed input size of 60 time steps.
* Anomaly labels are generated using pre-defined thresholds specific to each model and are aligned by removing the first 60 samples.
* The original dataset must include a numeric `consumption` column.

---

##  Citation

If you use this code, please cite:

**Srinidhi Madabhushi** and **Rinku Dewri**
*Mitigating Over-Generalization in Anomalous Power Consumption Detection using Adversarial Training*
**ACM Transactions on Cyber-Physical Systems**, April 2025.
