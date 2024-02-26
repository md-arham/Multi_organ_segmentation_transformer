# Multi_organ_segmentation_transformer
"Explore my GitHub repo showcasing multi-organ segmentation with transformers. Cutting-edge AI meets medical imaging for precise organ delineation. Join the revolution in healthcare AI!"
Usage
1. use the model.py file to get the UNETR model
  UNETR model consist of Vision transformer as encoder and CNN decoder.
2. Prepare data
Please go to "./datasets/README.md" for details.

3. Environment
Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

4. Train/Test
Run the train script on synapse dataset. The batch size can be reduced to 8 or 16 to save memory and put batch size 32 or 64 to get higher accuracy.

Reference
• Dataset : https://paperswithcode.com/dataset/miccai-2015-multi-atlas-abdomen-labeling
