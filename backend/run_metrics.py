import subprocess, sys

if __name__ == "__main__":
    code = subprocess.call([sys.executable, "plot_roc.py"])
    if code != 0:
        sys.exit(code)
