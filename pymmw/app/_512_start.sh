resolution="0.3021"
size="512"

echo -n $1
if [[ "$1" == "8.64" ]]; then
    resolution="0.2119"
elif [[ "$1" == "4.8" ]]; then
    resolution="0.3815"
elif [[ "$1" == "6" ]]; then
    resolution="0.3021"
elif [[ "$1" == "40" ]]; then
    resolution="0.0458"
elif [[ "$1" == "10" ]]; then
    resolution="0.1832"
elif [[ "$1" == "17" ]]; then
    resolution="0.1077"
fi
echo -n " "
echo $2
if [[ "$2" == "0.75" ]]; then
    size="384"
elif [[ "$2" == "0.5" ]]; then
    size="256"
elif [[ "$2" == "0.875" ]]; then
    size="448"
elif [[ "$2" == "0.625" ]]; then
    size="320"
fi

echo "resolution: $resolution m / size: $size samples"
#echo "sleep 10 seconds for recording"
#sleep 10

echo "start drawing..."
python3 plot_range_azimuth_heat_map.py 2 4 $size 50 $resolution 0
