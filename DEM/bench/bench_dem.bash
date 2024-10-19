#!/usr/bin/env bash

set -e
set -u

exe1=../build/original_dem_discr
exe2=../build/sdiscr_dem
exe3=../build/sdiscr_dem_parallel

genexe=../build/sdiscr_pointset
if [[ ! -f "$genexe" ]]; then
    echo "Could not find generator script"
fi

outf="times_dem.csv"
echo "test,algorithm,cores,time" > $outf

outdir="outputs"
mkdir -p $outdir

data_dir="/mnt/c/Users/Jaspar/Documents/Generalized_Hammersley_Points"

benchexe() {
    local inp=$1
    local exe=$2
    local cores=$3
    local inpname="$(basename "$inp" .txt)"
    local exename="$(basename "$exe")"
    local stdoutf="${outdir}/${inpname}_${exename}_${cores}_stdout.txt"
    local stderrf="${outdir}/${inpname}_${exename}_${cores}_stderr.txt"
    local timef="${outdir}/${inpname}_${exename}_${cores}_time.txt"

    if [ ! -f $exe ]; then
        echo "Could not find $exe binary, skipping.."
    else
        OMP_NESTED=true \
        OMP_NUM_THREADS="$cores" \
        /usr/bin/env time -f "%e" -o "$timef" \
            "$exe" < "$inp" > "$stdoutf" 2> "$stderrf"
        
        local exetime=$(head -n 1 $timef)
        echo "${inpname},${exename},${cores},${exetime}" >> $outf
        echo "$(printf "%-20s %2s %s" $exename $cores $exetime)"
    fi
}

for file in ${data_dir}/hammersley_point_a_*_b_*_num_digits_*.txt; do
    echo "# Benchmarking '$file'"
    # benchexe "$file" "$exe1" 1
    benchexe "$file" "$exe2" 1
    # benchexe "$file" "$exe3" 1
    # benchexe "$file" "$exe3" 2
    # benchexe "$file" "$exe3" 4
    # benchexe "$file" "$exe3" 8
    # benchexe "$file" "$exe3" 16
    # benchexe "$file" "$exe3" 32
done