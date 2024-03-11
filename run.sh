
for i in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0
do
python toriccode.py --ta_ ${i} --tb_ 1 --L 8 --num_samples 12000 --outsample sample_L8_${i} --outp p_L8_${i}
done

for i in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0
do
python measurement.py --ta_ ${i} --tb_ 1 --L 8 --insample sample_L8_${i}.npy --out EA_L8_${i}
done


# python toriccode.py --ta_ 0.8 --tb_ 1 --L 12 --num_samples 1000 --outsample test --outp test_p
