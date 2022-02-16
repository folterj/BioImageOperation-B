# BioImageOperation-B
### BioImageOperation-B (BIO-B) is a toolkit for behavioural analysis

<b>Quick guide</b>

- From BIO-B location, run from command line:

pip install -r requirements.txt

<b>Relabelling</b>
- Define desired parameters in .yml file (see example in resources/params.yml)
- From BIO-B location, run python script:

python run_relabelling.py --params path/to/params.yml

<b>Activity features</b>
- From BIO-B location, run python script:

python run_activity_features.py --params path/to/params.yml