# 主要环境依赖
```
pyhton 3.10.11
torch 2.1.2
transformers 4.36.2
gensim 4.3.2
```

# 运行
下载源码并安装好依赖后，执行下述代码
```shell
python run.py --max_length 100 --hidden_channels 256 --dropout 0.3 --lr 2e-5 --weight_decay 5e-5 --batch_size 16 --epoch 100 
```