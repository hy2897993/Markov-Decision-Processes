@echo on
jython -J-Xms6000m -J-Xmx6000m easyGW.py 
jython -J-Xms6000m -J-Xmx6000m hardGW.py 
python3 ploteasy.py
python3 plothard.py