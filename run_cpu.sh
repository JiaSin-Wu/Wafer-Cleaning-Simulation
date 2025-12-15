rm -r outputs
mkdir outputs
make clean
make cpu
srun -A ACD114118 ./final_cpu
python3 validation.py
