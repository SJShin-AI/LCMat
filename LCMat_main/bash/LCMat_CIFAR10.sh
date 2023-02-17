cd ..
for frac in 0.001
do
for mod in FacilityLocation
do
for eps in 0.01
do
for seed in 0 1 2
do
for se in 10
do
CUDA_VISIBLE_DEVICES=3 python3 -u main.py --fraction ${frac} --dataset CIFAR10 --data_path ~/datasets --model ResNet18 -se ${se} -so SGD --optimizer SGD --lr 0.1 -sp ./result --batch 128 --selection Submodular --submodular ${mod} --submodular_greedy NaiveGreedy --backpack True --kernel worst --eps ${eps} --seed ${seed}
done
done
done
done
done
