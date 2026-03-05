
import sys
import os

# Add current directory to path so we can import app
sys.path.append(os.getcwd())

try:
    from app import EnhancedCoimbatoreRoadNetwork
    print("Successfully imported EnhancedCoimbatoreRoadNetwork")
    network = EnhancedCoimbatoreRoadNetwork()
    print("Successfully instantiated EnhancedCoimbatoreRoadNetwork")
except ImportError:
    print("Could not import app.EnhancedCoimbatoreRoadNetwork. app.py might have other dependencies.")
except AttributeError as e:
    print(f"Caught expected AttributeError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {e}")
