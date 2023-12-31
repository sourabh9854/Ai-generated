Overview:

This is an image classification model built using deep learning techniques to classify images as either "real" or "AI generated." The model is trained on a dataset consisting of real images and images generated by AI algorithms.

Dataset :

The dataset used for training and validation contains two classes: "real" and "AI generated." The images in the "real" class are real-world photographs, while the images in the "AI generated" class are artificially generated using various AI techniques.

Dataset Link: (https://huggingface.co/datasets/competitions/aiornot)

Model Architecture :

The model architecture is based on a pre-trained neural network, specifically ResNet, MobileNet, InceptionV3. The pre-trained model's weights are used as a starting point, and additional layers are added to fine-tune the model for our specific classification task.

Training :

The final model is trained using MobileNetV2 .

Training Accuracy: 90.5
Validation Accuracy: 92

Future Improvements :

The model's performance can be further improved by experimenting with different architectures, adjusting hyperparameters, and increasing the diversity and size of the dataset.

Acknowledgments :

We acknowledge Huggingface dataset, which contributed to the development and training of the AI image classifier.

Contact :

If you have any questions or suggestions, please feel free to reach out.
