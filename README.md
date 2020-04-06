# Automatic Music Transcription listening tests website

This repository holds the code for the Automatic Music Transcription listening tests website.
The website uses Flask. The database is managed using SQLAlchemy.

## Requirements

This project uses Python 2.7.15. The list of required dependencies is available in `requirements.txt`
To install the list of dependencies, run:
`pip install -r requirements.txt`

## Data

All the audio examples should be placed in: `./app/static`
They should be in a folder containing one subfolder per example, and inside this subfolder, one mp3 per system + a reference mp3 named `target.mp3`

You can set the relative path of the parent of all the examples inside the `./app/static` folder by modifying the variable `DATA_PATH` in `./config.py`

## Running the website

Once the environment is set up, create the database with: `flask db init`

Then, start the website with: `flask run`

You can navigate the website at [http://localhost:5000](http://localhost:5000)

## More info

This website's design was heavily inspired by Miguel Grinberg's [Flask Mega Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world).
This is a very useful resource to understand Flask in general, and in particular how this specific website was designed.
