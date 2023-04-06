Le code a été conçu selon une structure en blocs. Chaque étape importante du traitement du signal est isolée dans un fichier .py dédié.
Pour installer les paquets nécessaires, exécutez la ligne de commande :

pip3 install -r requirements.txt

--|
  |-main.py/test_algo.py
  |   |
  |   |-block_1 [PreprocessingBlock] from preprocessing.py
  |   |     |- prétraitement et filtrage initial du signal
  |   |-block_2 [PrimaryQrsBlock] from qrs.py
  |   |     |- détection des qrs maternels
  |   |-block_3 [PcaBlock] from bss.py
  |   |     |- Primary Component Regression
  |   |-block_4 [SecondaryQrsBlock] from qrs.py
  |   |     |- détection des qrs fœtales
  |   |-block_5 [IcaBlock] from bss.py
  |   |     |- FastICA algorithme
  |   |-block_6 [CompareBlock] from compare.py
  |   |     |- la sélection du canal fœtal
