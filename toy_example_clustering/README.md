# Auxiliary Metamorphic Relations (MRs) Example Project

This project demonstrates the concept of **Auxiliary Metamorphic Relations (MRs)** using the **MNIST dataset** and a clustering algorithm as a case study. 

## Concept: Auxiliary Metamorphic Relations (MRs)

**Auxiliary MRs** extend traditional Metamorphic Testing by focusing on **output transformations** rather than input transformations. The general idea is:
1. **Transform the system’s output** (instead of the input).
2. Use an **auxiliary function** to measure the impact of the transformation on the system’s behavior.
3. Validate the correctness of the system based on how the output changes when passed through the auxiliary function.

In this project, we use the **intra-cluster distance** as the auxiliary function to evaluate the quality of clustering model.

## Dataset: MNIST Dataset

The **MNIST dataset** is used in this project. MNIST contains 70,000 images of handwritten digits (0-9), each with a size of 28x28 pixels. This dataset is widely used for machine learning and deep learning tasks, making it ideal for testing the clustering algorithm’s performance on high-dimensional image data.

We perform K-Means clustering on the MNIST dataset and calculate the **intra-cluster distance** to measure clustering quality before and after applying output transformations.

## Project Structure

- `src/`: Contains the main Python implementation of the model, and for auxiliary MRs using the MNIST dataset.
- `README.md`: This file, providing an overview of the project and instructions.
- `pyproject.toml`: Poetry project configuration.

## Dependencies

All dependencies are managed using **Poetry**. To install the required dependencies:

```bash
poetry install
```

## Running the Project

1. Activate the virtual environment using Poetry:
   ```bash
   poetry shell
   ```

2. Run the script to demonstrate auxiliary MRs using clustering on the MNIST dataset:
   ```bash
   python src/auxiliary_mr_clustering.py
   ```

### Example Output

The script will calculate the initial intra-cluster distance, apply a transformation to the cluster centers (output), and then recalculate the distance using the auxiliary function.

Example output:

```
Initial Intra-cluster Distance: 33.4215
Shifted Intra-cluster Distance: 42.7311
Auxiliary MR validated: Cluster quality worsened after shifting cluster centers.
```

This shows how the auxiliary MR concept is applied by observing the effects of transforming the output and evaluating the system based on the auxiliary function.

## Troubleshooting: PowerShell Script Execution Policy Issue

If you encounter the following error while running `poetry shell` on Windows:

```plaintext
File C:~\Local\pypoetry\Cache\virtualenvs\toy-example-clustering-lLbmjC5c-py3.12\Scripts\activate.ps1 cannot be loaded because running scripts is disabled on this system. For more information, see about_Execution_Policies at https:/go.microsoft.com/fwlink/?LinkID=135170.
    + CategoryInfo          : SecurityError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : UnauthorizedAccess
```

Follow these steps to resolve it:

1. **Open PowerShell as Administrator**:
   - Search for "PowerShell" in the Start menu.
   - Right-click on "Windows PowerShell" and select "Run as administrator."

2. **Change the Execution Policy**:
   Run this command to allow script execution:
   ```powershell
   Set-ExecutionPolicy RemoteSigned
   ```

3. **Confirm the Change**: Type `Y` to confirm.

4. **Retry the Command**:
   After changing the policy, retry:
   ```bash
   poetry shell
   ```

To revert the policy, run:
```powershell
Set-ExecutionPolicy Restricted
```

## License

This project is licensed under the MIT License.

