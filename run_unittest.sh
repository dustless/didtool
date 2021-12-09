
test_files=$(ls tests/ | grep _test.py)
for file in ${test_files[@]}; do
  echo "======= unittest for tests/$file ======="
  python -m unittest tests/$file
  if [ $? -ne "0" ];then
    echo "unittest failed"
    exit 1
  fi
done
