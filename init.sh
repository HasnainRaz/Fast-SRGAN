sudo apt install nano aria2 unzip

aria2c -x8 http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
aria2c -x8 http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip

mkdir DIV2K
mv DIV2K_valid_HR/* DIV2K/
mv DIV2K_train_HR/* DIV2K/

rm -rf DIV2K_train_HR DIV2K_train_HR.zip DIV2K_valid_HR DIV2K_valid_HR.zip

pip install pipenv
pipenv install --system --deploy
