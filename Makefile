TAG=2022-07-30_caustique
J=jupyter nbconvert  --ExecutePreprocessor.timeout=0 --allow-errors --execute
# J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace
#JN=$(J) --to markdown --stdout


default: run html
run:
	$(JN) $(TAG).ipynb
html:
	jupyter-nbconvert $(TAG).ipynb --to html --output index.html

install_local:
	python3 -m pip install --upgrade --user -r requirements.txt

install_global:
	python3 -m pip install --upgrade -r requirements.txt

pull_fortytwo:
	rsync -av -u  -e "ssh  -i ~/.ssh/id-ring-ecdsa"  laurentperrinet@10.164.6.136:quantic/EtienneRey/2020_caustiques/$(TAG) .

clean:
	rm -fr /tmp/$(TAG)/*
