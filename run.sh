
for i in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0
do
python toriccode.py --ta_ ${i} --tb_ 1 --L 12 --num_samples 12000 --outsample sample_L12_${i} --outp p_L12_${i}
done


# python toriccode.py --ta_ 0.8 --tb_ 1 --L 12 --num_samples 100 --outsample test --outp test
