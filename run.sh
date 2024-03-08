
for i in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0
do
python toriccode.py --ta_ ${i} --tb_ 1 --L 6 --num_samples 12000 --outsample sample_${i} --outp p_${i}
done

