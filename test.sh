proc="./ge"

type=("normal" "simd128" "simd512")

for test in 5 6 7; do
    for type in "normal" "simd128" "simd512"; do  
        for pp in 1 4 8;do
            echo $proc $test $type $pp > log
            $proc $test $type $pp
        done
    done
done
