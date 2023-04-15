# How to install required packages

## Installing pip
for systems with apt package manager 
```
sudo apt-get update
sudo apt-get install python3
sudo apt-get install python3-pip
```
for macOS users
```
brew install python
```

check version using
```
python --version
pip --version
```

## Installing required packages
Required packages for preprocessing data are stored in the requirements.txt file. To install all of them run:
```
pip install -r requirements.txt
```
Keep in mind that in the exact moment that you are installing them, new versions can be available. Consider executing this command before (you have to be in the project's root directory):
```
pipreqs --force .
```

## Pep8
To automatically reformat all Python files in the current directory to meet PEP8 requirements you can use **black** formatter

```
pip install black
black .
```