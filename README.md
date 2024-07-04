# LungCTseg

LungCTseg is a project aimed at segmenting lung CT images using various image processing techniques. This repository contains scripts for processing gradient images, computing the region of interest (ROI) gradient matrix, and repairing contours using convex hulls.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Features](#features)
4. [Contributing](#contributing)

## Introduction

The LungCTseg project aims to segment lung CT images through image processing techniques. This project includes scripts for processing gradient images, computing the ROI gradient matrix, and repairing contours using convex hulls.

## Getting Started

To install and run this project, please follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/LungCTseg.git

# Navigate to the project directory
cd LungCTseg


# Process All Gradient Images
python code/process_all_gradient_images.py

# Compute ROI Gradient Matrix
python code/ROI_gradient_matrix.py

# Repair Contours Using Convex Hull
python code/repair_contours_convex_hull.py
```

## Features
```
Process gradient images
Compute the region of interest (ROI) gradient matrix
Repair contours using convex hulls
```

## Contributing
Contributions are welcome! Please follow these steps to contribute:
```
# Fork this repository

# Create your feature branch
git checkout -b feature/your-feature

# Commit your changes
git commit -am 'Add some feature'

# Push to the branch
git push origin feature/your-feature

# Create a new Pull Request
```
