
J=jupyter nbconvert  --ExecutePreprocessor.timeout=0 --allow-errors --execute
# J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace
#JN=$(J) --to markdown --stdout


default:
	$(JN) 2022-07-19_caustique.ipynb


install_local:
	python -m pip install --user -r requirements.txt

install_global:
	python -m pip install -r requirements.txt

clean:
	rm -fr /tmp/2022-02-08_*
