**Getting Started**
Don't have sam3 hugging-faces token?
-> visit: https://github.com/facebookresearch/sam3 -> README -> Getting Started

1. create .env  file in folder with Dockerfile
```
	   HF_TOKEN=your_hf_token
```

Replece 'your_hf_token' with actual token from https://huggingface.co/settings/tokens 

**sam3 contener initialization:**
new terminal:
```
docker run -it --rm \
--gpus all \
-v $(pwd):/root/workspace \
--env-file .env \
sam3-local
```

*"docker run iteractvely, remove after,
use all GPUS (dedicated),
work on actual system files instead of copy,
pointer to .env file with password
contener_name" 

1	*Script #1* test_sam3.py 
same terminal (root@xxx:~/workspace# ) -> ```
```
python test_sam3.py images/tools.jpg "syringe" --output-path output/syringe.png
```
**LEGEND:
- ```images/tools.jpg```  -> your input image
- ```"syringe"``` -> prompt, what sam3 shall long for
- ```output/syringe.png``` -> your segmented output

2 *Script #2 test_batch_sam3.py

```
python3 test_batch_sam3.py  ./dataset_01 "a syringe" ./results --threshold 0.65 --mask-threshold 0.4
```

**LEGEND:
- ```./dataset_01```  -> your input image folder
- ```"a syringe"``` -> prompt, what sam3 shall long for
- ```./results``` -> folder for output
-  ```--threshold 0.65``` ->  the minimum confidence the model needs to accept and detect an object at all
- ```--mask-threshold 0.4``` ->  determines the width and precise edges of the applied mask