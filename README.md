# QCombo: Automated Commutator Calculation for Quantum Many-Body Operators

QCombo is a Python library for automated computation of commutators for normal-ordered many-body operators based on the generalized Wick theorem. Designed specifically for nuclear physics and condensed matter applications, it provides efficient and accurate tools for handling complex quantum many-body calculations.

## Table of Contents

- Overview
- Features
- Installation
- Quick Start
- Command Line Interface
- Python API Usage
- Citation
- License
- Support

## Overview

### Background and Motivation

In quantum mechanics, operator commutators form the foundation of theoretical calculations. In nuclear physics, *ab initio* methods (such as NCSM, CC, IMSRG, etc.) are essential tools for studying nuclear many-body systems. These methods start from nuclear forces constructed from chiral effective field theory and provide approximate but rigorous solutions to quantum many-body problems.

As the number of many-body terms increases, the number of terms in commutators grows exponentially, making manual calculations extremely tedious and error-prone. QCombo addresses this challenge by providing an automated solution for computing commutators of normal-ordered many-body operators.

## Features

- **Based on Generalized Wick Theorem**: Implements automated computation of commutators for normal-ordered many-body operators
- **Many-Body Coupling Support**: Handles commutators of arbitrary-order many-body operators
- **Automated Output Generation**:
  - Generates LaTeX files for commutator expressions
  - Produces input files for the AMC software package to obtain expressions in J-scheme
- **Flexible Configuration**: Allows specification of output to specific many-body ranks
- **Symbolic Computation**: Built on SymPy for precise algebraic manipulations
- **Command Line Interface**: Easy-to-use terminal commands for quick calculations
- **Interactive Mode**: Step-by-step guided interface for exploratory calculations

## Installation

### Install from PyPI (Recommended)

```
pip install qcombo
```

### Install from Source

```
git clone https://github.com/chenlh73/qcombo.git
cd qcombo
pip install -e .
```

### System Requirements

- Python >= 3.12
- SymPy >= 1.13.3

## Quick Start

### Command Line Interface (Recommended)

After installation, you can use QCombo directly from the command line:

```
# Compute the [2B, 2B] commutator with all possible contractions
qcombo 2 2

# Compute only 0-body and 1-body contractions
qcombo 2 2 -c 0,1

# Compute with range of contractions (0-2 body)
qcombo 1 3 -c 0-2

# Specify custom output filenames
qcombo 2 2 -c 0,1,2 --latex-output my_result.tex --amc-output my_result.txt

# Use specific indices (Python list syntax)
qcombo "[['a','b'], ['c','d']]" "[['e','f','g'], ['h','i','j']]" -c 0,1,2

# Start interactive mode
qcombo
# or explicitly:
qcombo -i

# Show help message
qcombo --help

# Show version information
qcombo --version
```

### Python API

You can also use QCombo within Python scripts:

```
import qcombo

# Compute the [2B, 2B] commutator coupled to all possible many-body ranks
qcombo.easyCombo(2, 2)

# Output only terms coupled to 0-body and 1-body ranks
qcombo.easyCombo(2, 2, [0, 1])
```

## Command Line Interface

### Basic Usage

QCombo provides a powerful command-line interface that allows you to perform calculations directly from your terminal:

```
qcombo LEFT RIGHT [OPTIONS]
```

Where:
- `LEFT`: Number of particles in left operator (integer) or index list (Python list string)
- `RIGHT`: Number of particles in right operator (integer) or index list (Python list string)

### Command Line Options

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--contraction` | `-c` | Contraction body numbers: single integer(0), range(0-2), list(0,1,2), or 'all' (calculate all) | `qcombo 2 2 -c 0,1,2` |
| `--latex-output` | `-lo` | LaTeX output filename | `qcombo 2 2 -lo result.tex` |
| `--amc-output` | `-ao` | AMC program input file output name | `qcombo 2 2 -ao result.txt` |
| `--version` | `-v` | Show version information | `qcombo --version` |
| `--quiet` | `-q` | Quiet mode, reduce output | `qcombo 2 2 -q` |
| `--interactive` | `-i` | Start interactive mode | `qcombo -i` |

### Interactive Mode

For exploratory calculations, QCombo offers an interactive mode that guides you through the process step-by-step:

```
qcombo
```

Or explicitly:
```
qcombo -i
```

In interactive mode, you will be prompted for:
1. Left operator (integer or index list)
2. Right operator (integer or index list)
3. Contraction body numbers
4. Output filenames (optional)

### Examples

**Basic calculations:**
```
# 2-body with 2-body operator commutator
qcombo 2 2

# 1-body with 3-body operator commutator
qcombo 1 3

# 2-body with 2-body, only 0-body and 1-body terms
qcombo 2 2 -c 0,1
```

**With custom indices:**
```
# Using specific index lists
qcombo "[['a','b'], ['c','d']]" "[['e','f'], ['g','h']]"

# Using range contractions
qcombo "[['i','j'], ['k','l']]" "[['m','n','o'], ['p','q','r']]" -c 0-2
```

**With output files:**
```
# Save results to specific files
qcombo 2 2 --latex-output commutator.tex --amc-output commutator.txt

# Using short options
qcombo 2 2 -c 0,1,2 -lo result.tex -ao result.txt
```

## Python API Usage

### Function Parameters

```
qcombo.easyCombo(left, right, contraction=None, latexOutput=None, amcOutput=None)
```

- `left`: Many-body rank of the first operator (integer) or index list
- `right`: Many-body rank of the second operator (integer) or index list
- `contraction`: Optional parameter specifying output ranks
  - Example: `[0, 1]` outputs only zero-body and one-body terms
  - `None` (default) outputs all possible many-body terms
- `latexOutput`: Custom LaTeX output filename
- `amcOutput`: Custom AMC input filename

### Basic Examples

```
import qcombo

# Compute [2B, 2B] commutator
result = qcombo.easyCombo(2, 2)

# Compute with specific contractions
result = qcombo.easyCombo(2, 2, contraction=[0, 1])

# Compute with custom output files
result = qcombo.easyCombo(2, 2, 
                          contraction=[0, 1, 2],
                          latexOutput="my_commutator.tex",
                          amcOutput="my_commutator.txt")

# Compute with specific indices
left_indices = [['a', 'b'], ['c', 'd']]
right_indices = [['e', 'f', 'g'], ['h', 'i', 'j']]
result = qcombo.easyCombo(left_indices, right_indices, contraction=[0, 1])
```

### Output Description

After execution, QCombo generates:

1. **Console Output**: Displays the computed commutator expressions in formatted text
2. **LaTeX Files**: Generated expressions in LaTeX format (ready for inclusion in papers)
3. **AMC Input Files**: Input files for the AMC software package to obtain expressions in J-scheme

The output filenames follow the pattern:
- LaTeX: `commutator_[left]B[right]B_to_[contractions]B.tex`
- AMC: `commutator_[left]B[right]B_to_[contractions]B_amcInput.txt`

For example: `commutator_2B2B_to_0_1_2B.tex`

### Output Format

The returned result is a dictionary where keys are strings in the format `'{filter_body}B_lambda{lambda_body}B'` and values are SymPy expressions:

```
{
    '0B_lambda1B': sympy_expression_1,
    '0B_lambda2B': sympy_expression_2,
    # ...
}
```

## Applications

### Nuclear Physics Applications

In nuclear *ab initio* methods, QCombo can be used for:

1. **IMSRG Flow Equations**: Computing commutators between generators and Hamiltonians
2. **Similarity Renormalization Group (SRG)**: Evaluating flow equations
3. **Configuration Interaction**: Deriving effective interactions

### Education and Research

QCombo is also valuable for quantum mechanics education and research:
- Verifying theoretical derivations
- Exploring contributions from higher-order many-body terms
- Automating tedious symbolic calculations
- Teaching quantum many-body theory concepts

## Citation

If you use QCombo in your research, please cite:

```
@software{qcombo2024,
  title = {QCombo: Automated Commutator Calculation for Quantum Many-Body Operators},
  author = {Chen, L.H. and Li, Y. and Hergert, Heiko and Yao, J.M.},
  year = {2024},
  url = {https://github.com/chenlh73/qcombo},
  note = {Based on generalized Wick theorem for automatic commutator calculation of many-body operators}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: Report bugs or request features on GitHub Issues: https://github.com/chenlh73/qcombo/issues
- **Documentation**: Visit the GitHub repository for more information: https://github.com/chenlh73/qcombo
- **Questions**: Contact the development team via email

---

*QCombo - Quantum Commutator of Many-Body operator *