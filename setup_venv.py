import os
import subprocess
import sys

# Chemin de l'application
app_path = os.path.dirname(os.path.abspath(__file__))

# Créer l'environnement virtuel
subprocess.call([sys.executable, '-m', 'venv', 'venv'])

# Chemin vers pip dans l'environnement virtuel
venv_pip = os.path.join(app_path, 'venv', 'bin', 'pip')

# Installer les dépendances
subprocess.call([venv_pip, 'install', '-r', 'requirements.txt'])

print("Environnement virtuel créé et dépendances installées avec succès!")