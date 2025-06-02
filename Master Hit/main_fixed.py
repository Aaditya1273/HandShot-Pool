import os
import sys
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import cv2
        import mediapipe
        import numpy
        import pyautogui
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def install_dependencies():
    """Install required dependencies using pip."""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please install them manually.")
        return False

def start_direct_controller():
    """Start the direct knife controller."""
    from direct_knife_controller import main
    main()

def print_banner():
    """Print the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███╗   ███╗ █████╗ ███████╗████████╗███████╗██████╗        ║
    ║   ████╗ ████║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗       ║
    ║   ██╔████╔██║███████║███████╗   ██║   █████╗  ██████╔╝       ║
    ║   ██║╚██╔╝██║██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗       ║
    ║   ██║ ╚═╝ ██║██║  ██║███████║   ██║   ███████╗██║  ██║       ║
    ║   ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝       ║
    ║                                                               ║
    ║   ██╗  ██╗██╗████████╗                                        ║
    ║   ██║  ██║██║╚══██╔══╝                                        ║
    ║   ███████║██║   ██║                                           ║
    ║   ██╔══██║██║   ██║                                           ║
    ║   ██║  ██║██║   ██║                                           ║
    ║   ╚═╝  ╚═╝╚═╝   ╚═╝                                           ║
    ║                                                               ║
    ║   Knife Throw Controller for Master Hit                       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main function to start the application."""
    print_banner()
    
    # Check if dependencies are installed
    if not check_dependencies():
        print("Some dependencies are missing.")
        install = input("Do you want to install them now? (y/n): ").lower()
        if install == 'y':
            if not install_dependencies():
                return
        else:
            print("Please install the required dependencies and try again.")
            return
    
    print("\nMaster Hit - Knife Throw Controller")
    print("=" * 50)
    print("Starting knife throw controller...")
    print("This will open a webcam window for hand tracking.")
    print("Controls:")
    print("- Open hand: Move cursor")
    print("- Quick close -> open: Throw knife")
    print("Press 'q' to quit, 'd' to toggle debug info")
    
    # Start the direct controller
    start_direct_controller()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
