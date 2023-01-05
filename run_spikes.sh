python=~/P3.10GPU/bin/python3

if [ -f "./def.json" ]
then
    echo "json file found"
else
    echo "No json file exists"
    exit
fi


$python learn_it.py def.json
status=$?
if [ $status != "0" ]
then
    echo "Problem detected (learn_it.py)!!!"
    exit 1
fi

