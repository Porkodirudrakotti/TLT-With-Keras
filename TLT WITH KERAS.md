<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TLT WITH KERAS</title>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__html"><h1 id="transfer-learning-with-resnet50">TRANSFER LEARNING WITH RESNET50</h1>
<p>Transfer learning is simplying taking the knowledge learned from a particular training process and using them in a new, similar process. This method is very handy when the user particularly has very <strong>less dataset</strong> to train a bigger model. By following this method the <strong>training time</strong> can be saved big-time, since it is not carried out from scratch and the <strong>accuracy</strong> level attained at the end is <strong>better</strong>.  Through this method one can save a lot of computation power.</p>
<p>Here we will be discussing as to, how one can train a model to detect weather the given image is a cat or dog. This is done using the <strong>Transfer learning method</strong> with ** ResNet50** using <strong>Keras</strong> .</p>
<h4 id="what-is-numpy">What is NumPy?</h4>
<p>NumPy is a python library and is very beneficial when one is working with arrays. It also contains functions, which helpsthe developer work in domains such as lnear algebra, fourier transform, matrix. NumPy is a short form of <strong>Numerical Python</strong>.</p>
<h4 id="what-is-tensorflow">What is tensorflow?</h4>
<p>For faster compution of complex numerical problems related to deep learning, an open source python library was developed by google which is called as tensorflow.</p>
<h4 id="what-is-keras-">What is Keras ?</h4>
<p>Keras is an deep learning API, which is developed on tensorflow.</p>
<h2 id="step-1--setup">STEP 1 : Setup</h2>
<ul>
<li>
<p>Download the dataset with respect to your desired model development from the internet for the training purpose.</p>
</li>
<li>
<p>Download the pre-trained model according to your choice of the training model.</p>
</li>
<li>
<p>Install and and import basic and essential libraries for easy computations.</p>
</li>
<li>
<p>Below mentioned packages mostly contains the  required libraries for this project, according to your model requirement you can import other libraries.</p>
</li>
<li>
<p>Install matplotlib, tensorflow, pandas, seaborn.</p>
</li>
<li>
<p>Import the below mentioned packages <em>absolute_import, division, print_function, unicode_literals, os, time, numpy, keras, trt, tag_constants, ResNet50, import image, preprocess_input,ImageDataGenerator<br>
Dense, Activation, Flatten, Dropout, Sequential, Model, Input, load_img, img_to_array, SGD, Adam, ModelCheckpoint, AveragePooling2D , Dropout, Flatten, random</em></p>
</li>
</ul>
<h2 id="step-2--set-the-dimensions">STEP 2 : Set the dimensions</h2>
<ul>
<li>The dimension of the training image data and validation image should be equal, so we resize the dowloaded images.</li>
</ul>
<pre><code>HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 8 
FC_LAYERS = [1024, 1024] 
DROPOUT = 0.5
CLASSES = ['CAT' , 'DOG'] 
NUM_EPOCHS = 2 
BATCH_SIZE = 8 
INIT_LR = 1e-4 

BASE_PATH = "/workspace/datasets/data" !ls {BASE_PATH}
</code></pre>
<ul>
<li>
<p><strong>Height and weight</strong> simply indicates the dimension of our target image dataset.</p>
</li>
<li>
<p><strong>Batch size</strong> the maximum number of claases that can be taken in will be mentioned here.</p>
</li>
<li>
<p><strong>Dropout</strong> to manage the overfitting.</p>
</li>
<li>
<p><strong>Classes</strong> are the categories under which we want the inpput image dataset to be classified.</p>
</li>
<li>
<p><strong>Num epochs</strong> indicates the number iterations to carried out in-order to attain the maximum possible accuracy. In this method, the accuracy level is attained with in a few number of epochs when compared to other training methods.</p>
</li>
<li>
<p><strong>INIT_LR</strong> is the initial learning rate. Learning rate is something which determines how much a particular step should affect the weights and biases.</p>
</li>
</ul>
<h2 id="step-3---data-augmentation">STEP 3 :  Data Augmentation</h2>
<ul>
<li>As mentioned before, when one chooses to use transfer learning method, we start with very less data set. In-order make the training process more efficient we process the input  images data set through various augmenttion parameters. This way we create possible varitions with the inputb image data set.</li>
</ul>
<pre><code>trainAug = ImageDataGenerator( 
rotation_range=25,
zoom_range=0.1, 
width_shift_range=0.1, 
height_shift_range=0.1, 
shear_range=0.2, 
horizontal_flip=True, 
fill_mode="nearest")
</code></pre>
<ul>
<li>
<p>The input  image is rotated to 25 degrees(which can be anything from 0 to 360 degree), through the <strong>rotation range</strong> parameter.</p>
</li>
<li>
<p>The input image is zoomed at the 0.1 range using the <strong>zoom range</strong> parameter.</p>
</li>
<li>
<p>The width of the input image is shifted to 0.1 units using the  <strong>width shift range</strong> parameter.</p>
</li>
<li>
<p>The height of the input image is varied using the  <strong>height shift range</strong> parameter.</p>
</li>
<li>
<p>When the image dimensions are kept as it is and the pixels are towards one direction, may veritically or horizontally then that is called as shift.</p>
</li>
<li>
<p>The input image brightness is changed by varing the  <strong>brightness range</strong> parameter.</p>
</li>
<li>
<p>The augmented images are used for training purpose beacuse the parameter of these images are randomly varied, so the accuracy of the prediction will be considerably better.</p>
</li>
<li>
<p>The augmented images are slpit into <strong>two ategories</strong>  80% is alloted for training process and the rest of the 20%  is utilized for testing process.</p>
</li>
</ul>
<h2 id="step-4--tranfer-learning">STEP 4 : Tranfer Learning</h2>
<ul>
<li>
<p>As mentioned in the step 1, we need to download te pre-trained model .</p>
</li>
<li>
<p>Here we are using <strong>ResNet50</strong> model.</p>
</li>
<li>
<p>Therefore the base model here is <strong>ResNet50</strong>.</p>
</li>
<li>
<p>The weights is considered as <strong>ImageNet</strong></p>
</li>
<li>
<p>So, in transfer learing method we remove the top layers of the base model and add our custom layers with respect to our model of interest and then the training process is carried out.</p>
</li>
<li>
<p>Inoder to add top custom layers, the <strong>include top</strong>  parameter has to be set false.</p>
</li>
</ul>
<pre><code>baseModel = ResNet50(
weights = "imagenet",
include_top = False, 
input_tensor = Input(shape=(224, 224,3))) 
</code></pre>
<h2 id="step-5--add-custom-layers">STEP 5 : Add Custom Layers</h2>
<p><img src="https://improductive-storage.s3.amazonaws.com/5a0534028d6a847ea9f180ab/chatuploads/1608802728-tlt.jpeg" alt="image"></p>
<ul>
<li>
<p>Inoder to make use of the pre-trained ResNet 50 model for our customised model we need to add layers of our interest on top.</p>
</li>
<li>
<p>The custom model added in this project are    <strong>AveragePooling2D</strong>, <strong>Flatten</strong>, <strong>Dense</strong>, <strong>Dropout</strong>.</p>
</li>
</ul>
<pre><code>CustomModel = baseModel.output 
CustomModel = AveragePooling2D(pool_size (7, 7))(CustomModel) 
CustomModel = Flatten(name="flatten (CustomModel) 
CustomModel = Dense(256,activation="relu")(CustomModel) 
CustomModel = Dropout(0.5)(CustomModel) 
CustomModel = Dense(len(CLASSES), activation="softmax")(CustomModel) 

model = Model(inputs=baseModel.input, outputs=CustomModel) 

for layer in baseModel.layers: 
layer.trainable = False

</code></pre>
<ul>
<li>
<p>The mathematical equation which determines the output of a neural network is called as <strong>Activation function</strong></p>
</li>
<li>
<p>Here we have used relu and softmax</p>
</li>
<li>
<p>Commonly used Pooling methods are of two types, one is <strong>average pooling</strong> and the other is <strong>max pooling</strong> .</p>
</li>
<li>
<p>Pooling basically summarizes  the <strong>average presence</strong> and  <strong>most active presence</strong> of the features present.</p>
</li>
<li>
<p>Pooling is required to down sample the detection of features in feature maps.</p>
</li>
<li>
<p><strong>Flatten</strong> is used to flatten the input.</p>
</li>
<li>
<p>A deeply connected neural network is called as a <strong>Dense layer</strong>.</p>
</li>
<li>
<p><strong>Dropout layer</strong> fixes the Overfitting</p>
</li>
</ul>
<h2 id="step-6--compile-model">STEP 6 : Compile Model</h2>
<pre><code>model.compile(
loss = "categorical_crossentropy", 
optimizer = opt,
metrics = ["accuracy"])
</code></pre>
<ul>
<li>Since we have multiple classes in our project so, the loss is considered as the categorical cross-entropy.</li>
</ul>
<h2 id="step-7--train-model">STEP 7 : Train Model</h2>
<ul>
<li>
<p>In the previous steps the custom layers were added and our personalised model was created and compiled.</p>
</li>
<li>
<p>The training process is done in this specific step.</p>
</li>
</ul>
<pre><code>history = model.fit( 
trainGen, 
steps_per_epoch=trainGen.n, 
validation_data=valGen, 
validation_steps=valGen.n, 
epochs=NUM_EPOCHS)

</code></pre>
<pre><code>model.summary()
</code></pre>
<ul>
<li>
<p>To see the entire model sumary, run the above command.</p>
</li>
<li>
<p>The graphical representation of the loss and accuracy of the model can be visualised by running the below code.</p>
</li>
</ul>
<pre><code>plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 
plt.title('model accuracy') 
plt.ylabel('accuracy') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='upper left') 
plt.show() 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss') 
plt.ylabel('loss') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'],loc='upper left') 
plt.show()
</code></pre>
<pre><code>model.save('PersonalModel')
</code></pre>
<ul>
<li>To save the <strong>PersonalModel</strong> run the above command.</li>
</ul>
<pre><code>model = load_model('mymodel1')

</code></pre>
<ul>
<li>To load the <strong>PersonalModel</strong> for the prediction process, one has to run the above command.</li>
</ul>
<pre><code>validation_img_paths = 
[BASE_PATH+'/predict/CAT/1.jpg', 
BASE_PATH+'/predict/DOG/2.jpg']

img_list = [Image.open(img_path) for img_path in validation_img_paths]
</code></pre>
<ul>
<li>The validation images are listed under ima_list with the dimensions of the image.</li>
</ul>
<pre><code>validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224)))) for img in img_list])
</code></pre>
<ul>
<li>In-oder to create an <strong>array of the image list</strong>, the above command is executed.</li>
</ul>
<h2 id="step-8--prediction-workflow">STEP 8 : PREDICTION WORKFLOW</h2>
<ul>
<li>The image which is to be predicted in selected.</li>
</ul>
<pre><code>predicting = BASE_PATH+'/predict/CAT/1.jpg' 
</code></pre>
<ul>
<li>The original image is loaded and then resized similar to the dimensions of the training image dataset.</li>
</ul>
<pre><code>original = load_img(predicting, target_size = (224, 224)) 
print('PIL image size',original.size)
</code></pre>
<ul>
<li>
<p>It is easy to introduce numpy when the image is in array format</p>
</li>
<li>
<p>The image is converted to the array format by running the below code.</p>
</li>
</ul>
<pre><code>numpy_image = img_to_array(original) 
image_batch = np.expand_dims(numpy_image, axis = 0) 
np.shape(image_batch)
</code></pre>
<ul>
<li>
<p>The image to be predicted is uploaded and the resizing process takes place, followed by which the image specification are printed in array format.</p>
</li>
<li>
<p>By running the below code the image is predicted and the accuracy is mentioned in the aaray format.</p>
</li>
<li>
<p>The class with highest accuracy is the class prediction result.</p>
</li>
<li>
<p>The prediction can be observed by running the below code snippet.</p>
</li>
</ul>
<pre><code>processed_image = preprocess_input(image_batch.copy()) 
preds = model.predict(processed_image) print('Predictions:', preds) 
predindex = np.argmax(preds, axis=-1) 
predclass = CLASSES[predindex[0]] 
print('Predicted:{}'.format(predclass))
</code></pre>
<h1 id="conclusion">CONCLUSION</h1>
<p>Thus using this method the model can be trained in an efficient manner, with lesser number of dataset and the user can observe a decent accuracy level!</p>
</div>
</body>

</html>
