# Makefile for SetiAstroSuite on MacOS
# written by Jackson Datkuliak
# tested on MacOS 15.1.1

PY = python3.12 # python version for venv

run: venv # setup python virtual environment and run program
	$(VENV)/python setiastrosuitemacQT6.py

build: venv # build executable with pyinstaller
	$(VENV)/pyinstaller setiastrosuitemac.spec

clean: # clean directory
	rm -rf ./dist ./build ./__pycache__ ./.venv

include Makefile.venv # need this to make python venv happy
