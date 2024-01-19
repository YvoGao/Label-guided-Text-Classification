dataset=huffpost
data_path="../data/huffpost.json"
n_train_class=20
n_val_class=5
n_test_class=16
n_train_domain=1
n_val_domain=1
n_test_domain=1
alpha_pro=1
seed=2023
temp=5
lr=1e-6



python ../src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] news: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr 

python ../src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 5 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] news: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr


dataset=reuters
data_path="../data/reuters.json"
n_train_class=15
n_val_class=5
n_test_class=11
n_train_domain=1
n_val_domain=1
n_test_domain=1



python ../src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] news: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr 

python ../src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 5 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] news: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr





dataset=20newsgroup
data_path="../data/20news.json"
n_train_class=8
n_val_class=5
n_test_class=7
n_train_domain=1
n_val_domain=1
n_test_domain=1



python ../src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] news: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr 

python ../src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 5 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] news: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr


dataset=amazon
data_path="../data/amazon.json"
n_train_class=10
n_val_class=5
n_test_class=9
n_train_domain=1
n_val_domain=1
n_test_domain=1


python ../src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] review: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr 

python ../src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 5 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] review: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr





dataset=clinc150
data_path="../data/clinc150.json"
# Cross domain = True
n_train_class=60
n_val_class=15
n_test_class=75
n_train_domain=4
n_val_domain=1
n_test_domain=5
# n_train_domain=1
# n_val_domain=1
# n_test_domain=1
alpha_pro=1
temp=5

python ../src/main.py \
        --cuda 0 \
        --way 10 \
        --shot 1 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] intent: [sentence]' \
        --add_pro \
        --add_instance \
        --cross_domain \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr

python ../src/main.py \
        --cuda 0 \
        --way 10 \
        --shot 5 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] intent: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --cross_domain \
        --T $temp \
        --SG single \
        --lr=$lr



python ../src/main.py \
        --cuda 0 \
        --way 15 \
        --shot 1 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] intent: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --cross_domain \
        --T $temp \
        --SG single \
        --lr=$lr


python ../src/main.py \
        --cuda 0 \
        --way 15 \
        --shot 5 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] intent: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --cross_domain \
        --T $temp \
        --SG single \
        --lr=$lr


dataset=banking77
data_path="../data/banking_data/"
n_train_class=30
n_val_class=15
n_test_class=32
n_train_domain=1
n_val_domain=1
n_test_domain=1



python ../src/main.py \
        --cuda 0 \
        --way 10 \
        --shot 1 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] intent: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr

python ../src/main.py \
        --cuda 0 \
        --way 10 \
        --shot 5 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] intent: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr


python ../src/main.py \
        --cuda 0 \
        --way 15 \
        --shot 1 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] intent: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr

python ../src/main.py \
        --cuda 0 \
        --way 15 \
        --shot 5 \
        --query 25 \
        --mode train \
        --classifier mbc \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --pool prompt \
        --template 'This is a [MASK] intent: [sentence]' \
        --add_pro \
        --add_instance \
        --protype single \
        --seed=$seed \
        --T $temp \
        --SG single \
        --lr=$lr
