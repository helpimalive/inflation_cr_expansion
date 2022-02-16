
:: national
python %cd%\stagflation_cap_rate_model.py analyze -mn national -v both
:: msa
python %cd%\stagflation_cap_rate_model.py analyze -mn msa -v both
:: msa individual
python %cd%\stagflation_cap_rate_model.py analyze -mn msa_individual -v both

pause