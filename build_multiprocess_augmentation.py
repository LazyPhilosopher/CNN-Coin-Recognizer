import subprocess
import os


def main():
    # Create necessary directories if they don't exist.
    os.makedirs('build/dist', exist_ok=True)
    os.makedirs('build/build', exist_ok=True)
    os.makedirs('build/spec', exist_ok=True)

    # Base PyInstaller command.
    command = [
        'pyinstaller',
        '--onefile',  # Bundle everything into a single executable.
        '--clean',  # Clean PyInstaller cache and remove temporary files before building.
        '--distpath', 'build/dist',  # Output directory for the final .exe.
        '--workpath', 'build/build',  # Directory for PyInstaller’s temporary work files.
        '--specpath', 'build/spec',  # Directory to store the generated spec file.
    ]

    # (Optional) Exclude modules you know are not needed to reduce size.
    # For example, if you do not use any testing modules:
    # command.extend(['--exclude-module', 'unittest'])
    # You can add more --exclude-module flags as required.

    # Specify the main script to compile.
    command.append('multiprocess_augmentation/multiprocess_augmentation.py')

    print("Building the executable with PyInstaller...")
    subprocess.run(command, check=True)
    print("Build complete. The executable can be found in build/dist.")


if __name__ == '__main__':
    main()