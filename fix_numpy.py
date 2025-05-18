import subprocess
import sys

def fix_numpy_compatibility():
    print("Fixing NumPy compatibility issue...")
    print("Your current Python environment will be modified.")
    print("This will downgrade NumPy to a version compatible with PyTorch.")
    
    try:
        # Uninstall NumPy 2.x
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
        
        # Install NumPy 1.24.3 (stable version compatible with PyTorch)
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3"])
        
        print("\nNumPy successfully downgraded to 1.24.3")
        print("Please restart your Python interpreter and try running your script again.")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("\nManual fix:")
        print("1. Open a command prompt")
        print("2. Activate your conda environment: conda activate SwinCoMER")
        print("3. Run: pip uninstall -y numpy")
        print("4. Run: pip install numpy==1.24.3")

if __name__ == "__main__":
    fix_numpy_compatibility() 