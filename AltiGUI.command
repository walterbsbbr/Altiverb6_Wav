#!/bin/bash
# Abre a interface do conversor (duplo clique no Finder)
cd "$(dirname "$0")"
for venv_name in venv .venv env; do
    [ -f "$venv_name/bin/activate" ] && source "$venv_name/bin/activate" && break
done
python3 AltiGUI.py
