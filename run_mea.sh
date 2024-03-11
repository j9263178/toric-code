
for i in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0
do
python measurement.py --ta_ ${i} --tb_ 1 --L 12 --insample sample_L12_${i}.npy --out EA_L12_${i}
done


# for i in 0.9

# do
# python measurement.py --ta_ ${i} --tb_ 1 --L 6 --insample sample_${i}.npy --out EA_${i}
# don