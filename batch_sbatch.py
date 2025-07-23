import os

def Gen_slurm(name, *args):
	print(name)
	with open(name,'w') as f:
		f.write("\n".join(args))

model = ["A1N2","A2N4","A3N8","B1N6","B2N6","B3N6","C1N6","C2N6","C3N6"]
wq = ["64","64","64","128","32","16","64","64","64"]
wk = ["64","64","64","128","32","16","64","64","64"]
wv = ["64","64","64","128","32","16","64","64","64"]
heads = ["12","12","12","6","24","48","12","12","12"]
eN = ["2","4","8","6","6","6","6","6","6"]
lr = ["0.00002","0.00002","0.00002","0.00002","0.00002","0.00002","0.00003","0.000015","0.00001"]
epochs = ["1500","1500","1500","1500","1500","1500","1000","2000","3000"]

for i in range(len(model)):
	name = f"{model[i]}.slurm"
	Gen_slurm(name,
		"#!/bin/bash",
		f"#BATCH -J bp_{model[i]}",
		"#SBATCH -p NVgpu",
		"#SBATCH -N 1",
		"#SBATCH --ntasks-per-node=1",
		"#SBATCH --cpus-per-task=2",
		"#SBATCH --mem=64G",
		"#SBATCH --gres=gpu:1",
		"#SBATCH -t 3-00:00:00",
		f"#SBATCH -o ./log/ABC_{model[i]}.out",
		f"#SBATCH -e ./log/ABC_{model[i]}.err",
		"export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64",
		"export PATH=$PATH:/usr/local/cuda-11.7/bin",
		"source activate pytorch",
		f'python model_batch_training.py -data "train datast (all pigments) --all.pkl" -n {model[i]} -wq {wq[i]} -wk {wk[i]} -wv {wv[i]} -eN {eN[i]} -dN {eN[i]} -lr {lr[i]} -epochs {epochs[i]}')
	# os.system(f"sbatch {name}")
	os.remove(name)