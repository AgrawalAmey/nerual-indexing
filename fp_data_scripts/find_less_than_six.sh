# set -x

dir='./nabard-data-lum-fingerprint/p2-Lum'

for dirname in $dir/*;
do
    if [[ -d $dirname ]]; then
        count=`ls $dirname | wc -l`
        if [[ $count -ne "6" ]]; then
            # echo File: $dirname Count: $count
            rm -rf $dirname
        fi
    fi
done