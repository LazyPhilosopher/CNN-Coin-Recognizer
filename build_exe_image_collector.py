import subprocess
import os

def main():
    # Create the necessary directories if they don't exist.
    os.makedirs('build/dist', exist_ok=True)
    os.makedirs('build/build', exist_ok=True)
    os.makedirs('build/spec', exist_ok=True)

    # Base PyInstaller command with the icon option.
    command = [
        'pyinstaller',
        '--onefile',  # Bundle everything into a single executable.
        '--clean',  # Clean PyInstaller cache and temporary files before building.
        '--distpath', 'build/dist',  # Output directory for the final executable.
        '--workpath', 'build/build',  # Directory for PyInstaller’s temporary work files.
        '--specpath', 'build/spec',  # Directory to store the generated spec file.
        # Use the icon as specified in image_collector.spec.
        '--icon', '../../core/gui/images/camera.png'
    ]

    # Add the necessary data file:
    # For Windows, use a semicolon as separator.
    # For Linux/Mac, replace the semicolon with a colon.
    command.extend([
        '--add-data', '../../core/utilities/u2net/u2net.onnx;core/utilities/u2net',
        '--add-data', '../../core/gui/images/camera.png;core/gui/images',
    ])

    # Optionally, add flags to exclude unnecessary modules to minimize file size.
    # For example:
    # command.extend(['--exclude-module', 'unittest'])

    # Specify the main script to compile.
    command.append('image_collector/image_collector.py')

    print("Building the image_collector executable with PyInstaller...")
    subprocess.run(command, check=True)
    print("Build complete. The executable can be found in build/dist.")

if __name__ == '__main__':
    main()
