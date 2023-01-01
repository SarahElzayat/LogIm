# Installation

This code runs in JupyterNotebook, so you must have anaconda installed first

[Anaconda download](https://www.anaconda.com/products/distribution)

```
conda install pip

```

```
pip install opencv-python
pip install truth-table-generator
pip install ipynb
pip install customtkinter
pip install image

```

# Run

Navigate to final_integration.py

Run all cells 

Create a new cell containing the code below

```
## new cell code##
image = cv2.imread("./image_path") #replace with your image path

solve_expression(image, is_table=bool) #for solving an expression, False for is_table
				       #for solving a truth table, True for is_table

```
