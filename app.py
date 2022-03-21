from flask import Flask, render_template, redirect, url_for, request, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from chatbot import * 


# import our model from folder
from face_recognition_and_liveness.face_liveness_detection.face_recognition_liveness_app import recognition_liveness

app = Flask(__name__)
app.secret_key = 'web_app_for_face_recognition_and_liveness' # something super secret
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Users(db.Model):
    username = db.Column(db.String(100), primary_key=True)
    name = db.Column(db.String(100))
    password = db.Column(db.String(100))

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        session.pop('name', None)
        username = request.form['username']
        password = request.form['password']
        user = Users.query.filter_by(username=username).first()        
        print(user)
        if user is not None and user.password == password:
            session['name'] = user.name # store variable in session
            detected_name, label_name = recognition_liveness('face_recognition_and_liveness/face_liveness_detection/liveness.model',
                                                    'face_recognition_and_liveness/face_liveness_detection/label_encoder.pickle',
                                                    'face_recognition_and_liveness/face_liveness_detection/face_detector',
                                                    'face_recognition_and_liveness/face_recognition/encoded_faces.pickle',
                                                    confidence=0.5)
            if user.name == detected_name and label_name == 'real':
                return redirect(url_for('main'))
            else:
                return render_template('login_page.html', invalid_user=True, username=username)
        else:
            return render_template('login_page.html', incorrect=True)

    return render_template('login_page.html')

@app.route('/main', methods=['GET','POST'])
def main():
    name = session['name']
    return render_template('script.html', name=name)

@app.route('/setph', methods=['GET','POST'])
def setph():
    #init_plc()
    setpoint=get_ph()
    #print(setpoint)
    #setpoint= float(setpoint)
    #print(type(setpoint))
    #setpoint  = 6.0
    write_data(plc, 'DB3.DBD0', setpoint)
    
    print('INFO:setpoint set to {}'.format(setpoint))
    engine.say('setpoint set to {}'.format(setpoint))
    #engine.runAndWait()
    
    print('INFO:starting process')
    engine.say('starting process')
    #engine.runAndWait()
    write_data(plc, 'M0.3', False)#set stop to False
    write_data(plc, 'M0.2', True)#set start to True   
    
    return render_template('script.html')

@app.route('/stop', methods=['GET','POST'])
def stop():
    print('This is the client')
    stop_words=['stop','end','halt']
    text=input("You: ")
    #text=voice_data()
    if any(i in text for i in stop_words ):
        print("stop word found")
        write_data(plc, 'M0.3', True)#stop to true
        write_data(plc, 'M0.2', False)#start to false      
    
    return render_template('script.html')
      
if __name__ == '__main__':
    db.create_all()

    # add users to database

    '''new_user = Users(username='wilfred_marete', password='123456789', name='Marete')
    db.session.add(new_user)

    new_user_5 = Users(username='kelvin_voke', password='123456789', name='Kelvin')
    new_user_6 = Users(username='joe_kasaine', password='123456789', name='Joe')
    db.session.add(new_user_5)
    db.session.add(new_user_6)'''

    app.run(debug=True)