
from flask import render_template, flash, redirect , request, url_for
from app import app
import numpy as np
from app.forms import LoginForm
from app.output import *

@app.route('/',methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        text = form.review.data
        matrix = tokenizer.texts_to_matrix([text])
        pred = np.argmax(model_rnn.predict(matrix.reshape((1,1,n_words))) ) + 1
        if pred == 1:
            return render_template('star.html' , user_image = "https://dublinbookworm.files.wordpress.com/2015/07/1-star2.jpg?w=300&h=60")
        elif pred == 2:
            return render_template('star.html' , user_image = "https://dublinbookworm.files.wordpress.com/2015/07/2-star2.jpg?w=300&h=60")
        elif pred == 3:
            return render_template('star.html' , user_image = "https://dublinbookworm.files.wordpress.com/2015/07/3-star2.jpg?w=300&h=60")
        elif pred == 4:
            return render_template('star.html' , user_image = "https://dublinbookworm.files.wordpress.com/2015/07/4-star2.jpg?w=300&h=60")
        else:
            return render_template('star.html' , user_image = "https://dublinbookworm.files.wordpress.com/2015/07/5-star2.jpg?w=300&h=60")
    return render_template('index.html', form=form)
