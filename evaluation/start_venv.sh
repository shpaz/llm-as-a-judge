#!/bin/bash
# -----------------------------------------------------------------------------
# setup_and_activate.sh - Creates and/or activates a venv for macOS & Linux
# -----------------------------------------------------------------------------
# USAGE:
# You MUST run this script using the 'source' command for it to work correctly.
# In your terminal, type:
#
#   source setup_and_activate.sh
#
# DO NOT run it like './setup_and_activate.sh', as that will not affect
# your current terminal session.
# -----------------------------------------------------------------------------

# Define the path to the virtual environment directory
VENV_DIR=".venv"

# Check if the activation file exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment found. Activating..."
    
    # Use 'source' to run the activation script in the current shell
    source "$VENV_DIR/bin/activate"
    
    echo "Virtual environment activated. You should see '(.venv)' in your prompt."
else
    echo "Virtual environment not found. Creating one now..."
    
    # Create the virtual environment
    python3 -m venv "$VENV_DIR"
    
    # Check if creation was successful before activating
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Virtual environment created successfully. Activating..."
        source "$VENV_DIR/bin/activate"
        echo "Virtual environment activated. You should see '(.venv)' in your prompt."
    else
        echo "Error: Failed to create the virtual environment."
    fi
fi
```batch
@echo off
rem --------------------------------------------------------------------------
rem setup_and_activate.bat - Creates and/or activates a venv for Windows
rem --------------------------------------------------------------------------
rem USAGE:
rem Simply run this batch file from your command prompt by typing:
rem
rem   setup_and_activate.bat
rem
rem --------------------------------------------------------------------------

set VENV_DIR=.venv

rem Check if the activation batch file exists
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment found. Activating...
    
    rem Call the activation script
    call "%VENV_DIR%\Scripts\activate.bat"
    
    echo Virtual environment activated.
) else (
    echo Virtual environment not found. Creating one now...
    
    rem Create the virtual environment
    python -m venv %VENV_DIR%
    
    rem Check if creation was successful before activating
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Virtual environment created successfully. Activating...
        call "%VENV_DIR%\Scripts\activate.bat"
        echo Virtual environment activated.
    ) else (
        echo Error: Failed to create the virtual environment.
    )
)

