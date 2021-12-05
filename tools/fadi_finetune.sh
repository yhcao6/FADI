split=$1
shot=$2
GPUS=${GPUS:-8}

asso_config=configs/voc_split${split}/fadi_split${split}_shot${shot}_association.py
disc_config=configs/voc_split${split}/fadi_split${split}_shot${shot}_discrimination.py

# association
./tools/dist_train.sh $asso_config $GPUS &&

# discrimination
./tools/dist_train.sh $disc_config $GPUS --load-from work_dirs/fadi_split${split}_shot${shot}_association/latest.pth
