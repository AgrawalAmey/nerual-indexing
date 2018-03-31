# set -x

dir='./nabard-data-lum-fingerprint/p1-Lum'

for filename in $dir/*.bmp;
do
    without_dir=`basename $filename`
    without_ext=`echo $without_dir | cut -f1 -d '.'`
    id=`echo $without_ext | cut -f1 -d '_'` 
    image_num=`echo $without_ext | cut -f2 -d '_'`
    mkdir -p $dir/$id
    mv $filename $dir/$id/$image_num.bmp
done