set -e
mkdir -p data; cd data
wget https://dataset-bj.cdn.bcebos.com/qianyan/duie_sample.json.zip
wget https://dataset-bj.cdn.bcebos.com/qianyan/duie_schema.zip
wget https://dataset-bj.cdn.bcebos.com/qianyan/duie_train.json.zip
wget https://dataset-bj.cdn.bcebos.com/qianyan/duie_dev.json.zip
wget https://dataset-bj.cdn.bcebos.com/qianyan/duie_test2.json.zip
for f in *.zip;
do 
unzip -o -d . $f
done 