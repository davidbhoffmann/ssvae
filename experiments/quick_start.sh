#!/bin/bash
# Quick start script for SSVAE experiments

echo "========================================="
echo "SSVAE Label Robustness Experiments"
echo "========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null
then
    echo "Error: Python not found. Please install Python 3.6+."
    exit 1
fi

# Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 6) else 1)"
if [ $? -ne 0 ]; then
    echo "Error: Python 3.6 or higher is required."
    exit 1
fi

echo "Step 1: Checking dependencies..."
echo ""

# List of required packages
REQUIRED_PACKAGES="torch torchvision numpy scipy scikit-learn matplotlib seaborn"

# Check if packages are installed
MISSING_PACKAGES=""
for package in $REQUIRED_PACKAGES; do
    python -c "import $package" 2>/dev/null
    if [ $? -ne 0 ]; then
        MISSING_PACKAGES="$MISSING_PACKAGES $package"
    fi
done

# Check probtorch separately
python -c "import probtorch" 2>/dev/null
if [ $? -ne 0 ]; then
    MISSING_PACKAGES="$MISSING_PACKAGES probtorch"
fi

if [ ! -z "$MISSING_PACKAGES" ]; then
    echo "Missing packages:$MISSING_PACKAGES"
    echo ""
    read -p "Install missing packages? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing packages..."
        pip install $MISSING_PACKAGES
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install packages."
            exit 1
        fi
    else
        echo "Please install missing packages manually:"
        echo "  pip install $MISSING_PACKAGES"
        exit 1
    fi
fi

echo "✓ All dependencies installed"
echo ""

echo "Step 2: Running quick test..."
echo ""

python quick_test.py

if [ $? -ne 0 ]; then
    echo "Error: Quick test failed."
    exit 1
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "You can now run experiments:"
echo ""
echo "1. Full experiment (both datasets, all conditions):"
echo "   python label_robustness_experiment.py"
echo ""
echo "2. Quick experiment (MNIST only, fewer epochs):"
echo "   python label_robustness_experiment.py --dataset MNIST --num_epochs 50"
echo ""
echo "3. Specific experiments:"
echo "   python label_robustness_experiment.py --sparsity_only"
echo "   python label_robustness_experiment.py --noise_only"
echo ""
echo "4. Analyze results:"
echo "   jupyter notebook analyze_results.ipynb"
echo ""
