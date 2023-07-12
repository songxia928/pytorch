

export CUDA_VISIBLE_DEVICES=0


dataset=onePerson_3d    # mnist, cifar10, slice,            onePerson_3d, onePerson_3d_ROI
net=resnet3d_vae           # resnet, capsnet,            resnet_3d, capsnet_3d, capsnet_3d_resnet, resnet3d_vae
with_reconstruction=0
lr=0.005

nohup python3 main.py   \
		  --dataset ${dataset}   \
		  --net ${net}           \
		  --with_reconstruction ${with_reconstruction}    \
		  --lr ${lr}    \
          >log/${dataset}_${net}_${with_reconstruction}_${lr}.log 2>&1 &    # 后台运行data.sh，并保存log

tail -f log/${dataset}_${net}_${with_reconstruction}_${lr}.log    # 实时打印log



