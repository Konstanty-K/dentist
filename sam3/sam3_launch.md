**Getting Started**
Dont't have sam3 hugging-faces token?
-> visit: https://github.com/facebookresearch/sam3 -> README -> Getting Started

1. create .env  file in folde with Dockerfile
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
*docker run iteractvely, remove after,
use all GPUS (dedicated),
work on actual system files instead of copy,
pointer to .env file with password
contener_name 

*Script #1* test_sam3.py 
same terminal (root@xxx:~/workspace# ) -> ```
```
python test_sam3.py images/tools.jpg "syringe" --output-path output/syringe.png
```
**LEGEND:
- ```images/tools.jpg```  -> your input image
- ```"syringe"``` -> prompt, what sam3 shall long for
- ```output/syringe.png``` -> your segmented output
