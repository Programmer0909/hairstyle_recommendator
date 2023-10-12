from flask import Flask, request, render_template
import tensorflow as tf
import tensorflow_hub as hub
import os 
import numpy as np
app = Flask(__name__)
app.static_folder = 'static'
file_paths_array = [
  ['archive/images/Train/Asymetrical/1.jpg',
   'archive/images/Train/Asymetrical/2.jpg',
   'archive/images/Train/Asymetrical/3.jpg',
   'archive/images/Train/Asymetrical/4.jpg',
   'archive/images/Train/Asymetrical/5.jpg',
   'archive/images/Train/Asymetrical/6.jpg',
   'archive/images/Train/Asymetrical/7.jpg',
   'archive/images/Train/Asymetrical/8.jpg',
   'archive/images/Train/Asymetrical/9.jpg',
   'archive/images/Train/Asymetrical/10.jpg',
   'archive/images/Train/Asymetrical/11.jpg',
   'archive/images/Train/Asymetrical/12.jpg',
   'archive/images/Train/Asymetrical/13.jpg',
   'archive/images/Train/Asymetrical/14.jpg',
   'archive/images/Train/Asymetrical/15.jpg'],
  ['archive/images/Train/Bangs/1.jpeg',
   'archive/images/Train/Bangs/2.jpeg',
   'archive/images/Train/Bangs/3.jpeg',
   'archive/images/Train/Bangs/4.jpeg',
   'archive/images/Train/Bangs/5.jpeg',
   'archive/images/Train/Bangs/6.jpeg',
   'archive/images/Train/Bangs/7.jpeg',
   'archive/images/Train/Bangs/8.jpeg',
   'archive/images/Train/Bangs/9.jpeg',
   'archive/images/Train/Bangs/10.jpeg',
   'archive/images/Train/Bangs/11.jpeg',
   'archive/images/Train/Bangs/12.jpeg',
   'archive/images/Train/Bangs/13.jpeg',
   'archive/images/Train/Bangs/14.jpeg',
   'archive/images/Train/Bangs/15.jpeg'],
  ['archive/images/Train/Blunt/1.jpeg',
   'archive/images/Train/Blunt/2.jpeg',
   'archive/images/Train/Blunt/3.jpeg',
   'archive/images/Train/Blunt/4.jpeg',
   'archive/images/Train/Blunt/5.jpeg',
   'archive/images/Train/Blunt/6.jpeg',
   'archive/images/Train/Blunt/7.jpeg',
   'archive/images/Train/Blunt/8.jpeg',
   'archive/images/Train/Blunt/9.jpeg',
   'archive/images/Train/Blunt/10.jpeg',
   'archive/images/Train/Blunt/11.jpeg',
   'archive/images/Train/Blunt/12.jpeg',
   'archive/images/Train/Blunt/13.jpeg',
   'archive/images/Train/Blunt/14.jpeg',
   'archive/images/Train/Blunt/15.jpeg'],
  ['archive/images/Train/Bob/1.jpg',
   'archive/images/Train/Bob/2.jpg',
   'archive/images/Train/Bob/3.jpg',
   'archive/images/Train/Bob/4.jpg',
   'archive/images/Train/Bob/5.jpg',
   'archive/images/Train/Bob/6.jpg',
   'archive/images/Train/Bob/7.jpg',
   'archive/images/Train/Bob/8.jpg',
   'archive/images/Train/Bob/9.jpg',
   'archive/images/Train/Bob/10.jpg',
   'archive/images/Train/Bob/11.jpg',
   'archive/images/Train/Bob/12.jpg',
   'archive/images/Train/Bob/13.jpg',
   'archive/images/Train/Bob/14.jpg',
   'archive/images/Train/Bob/15.jpg'],
  ['archive/images/Train/Bun/1.jpeg',
   'archive/images/Train/Bun/2.jpeg',
   'archive/images/Train/Bun/3.jpeg',
   'archive/images/Train/Bun/4.jpeg',
   'archive/images/Train/Bun/5.jpeg',
   'archive/images/Train/Bun/6.jpeg',
   'archive/images/Train/Bun/7.jpeg',
   'archive/images/Train/Bun/8.jpeg',
   'archive/images/Train/Bun/9.jpeg',
   'archive/images/Train/Bun/10.jpeg',
   'archive/images/Train/Bun/11.jpeg',
   'archive/images/Train/Bun/12.jpeg',
   'archive/images/Train/Bun/13.jpeg',
   'archive/images/Train/Bun/14.jpeg',
   'archive/images/Train/Bun/15.jpeg'],
  ['archive/images/Train/Curly/1.jpeg',
   'archive/images/Train/Curly/2.jpeg',
   'archive/images/Train/Curly/3.jpeg',
   'archive/images/Train/Curly/4.jpeg',
   'archive/images/Train/Curly/5.jpeg',
   'archive/images/Train/Curly/6.jpeg',
   'archive/images/Train/Curly/7.jpeg',
   'archive/images/Train/Curly/8.jpeg',
   'archive/images/Train/Curly/9.jpeg',
   'archive/images/Train/Curly/10.jpeg',
   'archive/images/Train/Curly/11.jpeg',
   'archive/images/Train/Curly/12.jpeg',
   'archive/images/Train/Curly/13.jpeg',
   'archive/images/Train/Curly/14.jpeg',
   'archive/images/Train/Curly/15.jpeg'],
  ['archive/images/Train/Layered/1.jpg',
   'archive/images/Train/Layered/2.jpg',
   'archive/images/Train/Layered/3.jpg',
   'archive/images/Train/Layered/4.jpg',
   'archive/images/Train/Layered/5.jpg',
   'archive/images/Train/Layered/6.jpg',
   'archive/images/Train/Layered/7.jpg',
   'archive/images/Train/Layered/8.jpg',
   'archive/images/Train/Layered/9.jpg',
   'archive/images/Train/Layered/10.jpg',
   'archive/images/Train/Layered/11.jpg',
   'archive/images/Train/Layered/12.jpg',
   'archive/images/Train/Layered/13.jpg',
   'archive/images/Train/Layered/14.jpg',
   'archive/images/Train/Layered/15.jpg'],
  ['archive/images/Train/Long/1.jpeg',
   'archive/images/Train/Long/2.jpeg',
   'archive/images/Train/Long/3.jpeg',
   'archive/images/Train/Long/4.jpeg',
   'archive/images/Train/Long/5.jpeg',
   'archive/images/Train/Long/6.jpeg',
   'archive/images/Train/Long/7.jpeg',
   'archive/images/Train/Long/8.jpeg',
   'archive/images/Train/Long/9.jpeg',
   'archive/images/Train/Long/10.jpeg',
   'archive/images/Train/Long/11.jpeg',
   'archive/images/Train/Long/12.jpeg',
   'archive/images/Train/Long/13.jpeg',
   'archive/images/Train/Long/14.jpeg',
   'archive/images/Train/Long/15.jpeg'],
  ['archive/images/Train/Pixie/1.jpeg',
   'archive/images/Train/Pixie/2.jpeg',
   'archive/images/Train/Pixie/3.jpeg',
   'archive/images/Train/Pixie/4.jpeg',
   'archive/images/Train/Pixie/5.jpeg',
   'archive/images/Train/Pixie/6.jpeg',
   'archive/images/Train/Pixie/7.jpeg',
   'archive/images/Train/Pixie/8.jpeg',
   'archive/images/Train/Pixie/9.jpeg',
   'archive/images/Train/Pixie/10.jpeg',
   'archive/images/Train/Pixie/11.jpeg',
   'archive/images/Train/Pixie/12.jpeg',
   'archive/images/Train/Pixie/13.jpeg',
   'archive/images/Train/Pixie/14.jpeg',
   'archive/images/Train/Pixie/15.jpeg'],
  ['archive/images/Train/Side/1.jpeg',
   'archive/images/Train/Side/2.jpeg',
   'archive/images/Train/Side/3.jpeg',
   'archive/images/Train/Side/4.jpeg',
   'archive/images/Train/Side/5.jpeg',
   'archive/images/Train/Side/6.jpeg',
   'archive/images/Train/Side/7.jpeg',
   'archive/images/Train/Side/8.jpeg',
   'archive/images/Train/Side/9.jpeg',
   'archive/images/Train/Side/10.jpeg',
   'archive/images/Train/Side/11.jpeg',
   'archive/images/Train/Side/12.jpeg',
   'archive/images/Train/Side/13.jpeg',
   'archive/images/Train/Side/14.jpeg',
   'archive/images/Train/Side/15.jpeg']
]


Heart_num = [1,3,6]
Oblong_num = [1,5,7]
Oval_num = [2,4,8]
Round_num = [0,4,6]
Square_num = [6,8,9]

Heart_cat = ["Bangs","Bob","Layered"]
Oblong_cat = ["Long" , "Bangs" , "Curly"]
Oval_cat = ["Blunt" , "Bun" , "Pixie"]
Round_cat = ["Layered" , "Bun" , "Asymetrical"]
Square_cat = ["Layered" , "Side" , "Pixie"]

loaded_full_model = tf.keras.models.load_model('20231004-15451696434346-full-image-set-2-mobilenetv2-Adam.h5',
                                               custom_objects={"KerasLayer": hub.KerasLayer})
def process_image(image_path):
    IMG_SIZE = 224
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

def func(custom_image_paths):
    data = tf.data.Dataset.from_tensor_slices((tf.constant(custom_image_paths))) 
    custom_data = data.map(process_image).batch(32)
    custom_preds = loaded_full_model.predict(custom_data)
    unique_breeds = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
    custom_pred_labels = [unique_breeds[np.argmax(custom_preds[i])] for i in range(len(custom_preds))]
    return custom_pred_labels
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Check if a file was uploaded
        uploaded_files = []
        if "image" in request.files:
            image = request.files["image"]
            if image.filename != "":
                # Save the image to the "uploads" folder
                upload_folder = os.path.join("static", "uploads")
                os.makedirs(upload_folder, exist_ok=True)
                file_path = os.path.join(upload_folder, image.filename)
                image.save(file_path)
                
                # Process the uploaded image and get predictions
                uploaded_files.append(file_path)  # Create a list with the uploaded file path
                res = func(uploaded_files)
                if(res[0] == "Heart"): return render_template("Heart.html", pred=res , file_paths_array = file_paths_array , num = Heart_num, cat = Heart_cat )  
                elif (res[0]=="Oblong"): return render_template("Oblong.html", pred=res , file_paths_array = file_paths_array , num = Oblong_num, cat = Oblong_cat )  
                elif (res[0]=="Oval"): return render_template("Oval.html", pred=res , file_paths_array = file_paths_array , num = Oval_num, cat = Oval_cat )  
                elif (res[0]=="Square"): return render_template("Square.html", pred=res , file_paths_array = file_paths_array , num = Square_num, cat = Square_cat )  
                elif (res[0]=="Round"): return render_template("Round.html", pred=res , file_paths_array = file_paths_array , num = Round_num, cat = Round_cat )  

    # If it's a GET request or no file was uploaded, render the upload.html template
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
