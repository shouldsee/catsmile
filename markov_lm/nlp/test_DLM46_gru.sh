set -x
ACT="python3 -mpdb -cc -cq train.py"
x="
-1
0 1 2 3 4 5 6 7"
for w in $x
do
time $ACT --save 10 --model.window_size $w --target 30 --LOAD 20
done
