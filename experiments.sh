# python pipline.py config_random_order_1.json  
# python pipline.py config_random_order_2.json

CUDA_VISIBLE_DEVICES=1 python pipline.py config_olmo_order_1.json  
CUDA_VISIBLE_DEVICES=1 python pipline.py config_olmo_order_2.json

CUDA_VISIBLE_DEVICES=1 python pipline.py config_notorder_olmo_all.json
# python pipline.py config_notorder_random.json