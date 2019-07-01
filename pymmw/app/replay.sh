resolution="0.3021"
size="256"

echo -n $1
if [[ "$1" == "8.64" ]]; then
    resolution="0.4238"
elif [[ "$1" == "4.8" ]]; then
    resolution="0.7630"
elif [[ "$1" == "6" ]]; then
    resolution="0.6042"
elif [[ "$1" == "40" ]]; then
    resolution="0.0916"
elif [[ "$1" == "44" ]]; then
    resolution="0.0832"
elif [[ "$1" == "10" ]]; then
    resolution="0.3664"
elif [[ "$1" == "17" ]]; then
    resolution="0.2154"
fi
echo -n " "
echo $2
if [[ "$2" == "0.75" ]]; then
    size="192"
elif [[ "$2" == "0.5" ]]; then
    size="128"
elif [[ "$2" == "0.375" ]]; then
    size="96"
elif [[ "$2" == "0.875" ]]; then
    size="224"
elif [[ "$2" == "0.625" ]]; then
    size="160"
fi

echo "===== sleep 5 seconds for ramp up application ====="
sleep 5
echo "resolution: $resolution m / size: $size samples"
#echo "sleep 10 seconds for recording"
#sleep 10

echo "start drawing..."
python3 plot_range_azimuth_heat_map.py 2 4 $size 50 $resolution 0 0.3 $2 read
