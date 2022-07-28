TAG=2022-07-27_caustique
J=jupyter nbconvert  --ExecutePreprocessor.timeout=0 --allow-errors --execute
# J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace
#JN=$(J) --to markdown --stdout


default:
	$(JN) $(TAG).ipynb
	jupyter-nbconvert $(TAG).ipynb --to html --output index.html

install_local:
	python3 -m pip install --upgrade --user -r requirements.txt

install_global:
	python3 -m pip install --upgrade -r requirements.txt

clean:
	rm -fr /tmp/$(TAG)/*
